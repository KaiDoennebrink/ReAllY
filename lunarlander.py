import logging, os
from pathlib import Path

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
import tensorflow_probability as tfp
import numpy as np
import gym
import ray
from really import SampleManager  # important !!
from really.utils import dict_to_dict_of_datasets

# ===================== DEFINE GLOBAL (HYPER)PARAMETERS ====================== #

# general parameters
ENV_NAME = 'LunarLanderContinuous-v2'
SAVING_DIRECTORY = '/logging/lunarlander'

# parameters for training
GAMMA = 0.99
CLIPPING_VALUE = 0.2
EPOCHS = 1000
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.0001
SAMPLE_SIZE = 1000
BATCH_SIZE = 64

# parameters for testing and saving
MAX_TEST_STEPS = 500
TEST_EPISODES = 20
RENDER_EPISODES = 100
SAVE_EPISODES = 20

# parameters for the model
ACTOR_HIDDEN_LAYER_NEURONS = [32, 32, 32]
ACTOR_HIDDEN_LAYER_ACTIVATION_FUNCTION = 'relu'
CRITIC_HIDDEN_LAYER_NEURONS = [32, 32, 32]
CRITIC_HIDDEN_LAYER_ACTIVATION_FUNCTION = 'relu'

# parameters for loading models
LOAD_MODELS = True
LOADING_PATH_ACTOR = '/homework_solutions/homework_3/models/actor/epoch_640'
LOADING_PATH_CRITIC = '/homework_solutions/homework_3/models/critic/epoch_640'
START_EPOCH = 641

JUST_TESTING = False

# ====================== DEFINE THE ACTOR-CRITIC-MODEL ======================= #

class AdvantageActorCritic(Model):

    def __init__(self, observation_space, action_space):
        super(AdvantageActorCritic, self).__init__()

        # initialize the spaces
        self.observation_space = observation_space
        self.action_space = action_space

        # create the actor and critic
        self.actor = self._create_actor()
        self.critic = self._create_critic()


    def __call__(self, state_batch):
        mu_batch, sigma_batch = self.actor(state_batch)
        value_estimate_batch = self.critic(state_batch)
        output = {}
        output['mu'] = tf.squeeze(mu_batch)
        output['sigma'] = tf.squeeze(sigma_batch)
        output['value_estimate'] = tf.squeeze(value_estimate_batch)
        return output


    def _create_actor(self):
        state_input = Input(shape=self.observation_space)

        # create the hidden layers
        next_input = state_input
        for i in range(len(ACTOR_HIDDEN_LAYER_NEURONS)):
            units = ACTOR_HIDDEN_LAYER_NEURONS[i]
            activation = ACTOR_HIDDEN_LAYER_ACTIVATION_FUNCTION
            name = f'actor_dense_{i}'
            next_input = Dense(units, activation=activation, name=name)(next_input)

        # create the output layers
        mu = Dense(self.action_space, activation='tanh', name='actor_mu')(next_input)
        sigma = tf.exp(Dense(self.action_space, activation=None, name='actor_sigma')(next_input))
        return Model(inputs=state_input, outputs=[mu,sigma])


    def  _create_critic(self):
        state_input = Input(shape=self.observation_space)

        # create the hidden layers
        next_input = state_input
        for i in range(len(CRITIC_HIDDEN_LAYER_NEURONS)):
            units = CRITIC_HIDDEN_LAYER_NEURONS[i]
            activation = CRITIC_HIDDEN_LAYER_ACTIVATION_FUNCTION
            name = f'critic_dense_{i}'
            next_input = Dense(units, activation=activation, name=name)(next_input)

        # create the output layers
        value_estimate = Dense(1, name='critic_value_estimate')(next_input)
        return Model(inputs=state_input, outputs=value_estimate)

# ============================== UPDATE METHODS ============================== #

@tf.function
def train_policy_network(agent, state, action, log_prob, advantage, optimizer):
    """Trains the policy network with PPO clipped loss."""

    with tf.GradientTape() as tape:
        # calculate the probability ratio
        new_log_prob = agent.flowing_log_prob(state, action)
        prob_ratio = tf.exp(new_log_prob - tf.cast(log_prob, dtype=tf.float32))

        # calculate the loss - PPO
        unclipped_loss = prob_ratio * tf.expand_dims(advantage, 1)
        clipped_loss = tf.clip_by_value(prob_ratio, 1 - CLIPPING_VALUE, 1 + CLIPPING_VALUE) * tf.expand_dims(advantage, 1)
        loss = -tf.reduce_mean(tf.minimum(unclipped_loss, clipped_loss))
        gradients = tape.gradient(loss, agent.model.actor.trainable_variables)

    optimizer.apply_gradients(zip(gradients, agent.model.actor.trainable_variables))
    return loss


@tf.function
def train_value_network(agent, state, value, optimizer):
    """Trains the value network with the mean squared error between the true and estimated value."""

    with tf.GradientTape() as tape:
        value_pred = agent.v(state)
        loss = MSE(value, value_pred)
        gradients = tape.gradient(loss, agent.model.critic.trainable_variables)

    optimizer.apply_gradients(zip(gradients, agent.model.critic.trainable_variables))
    return loss

# ============================== TRAINING LOOP =============================== #

if __name__ == '__main__':
    ray.init(log_to_driver=False)

    # define the environment, model, and sample manager arguments
    env = gym.make(ENV_NAME)

    model_kwargs = {
        'observation_space': env.observation_space.shape[0],
        'action_space': env.action_space.shape[0]
    }

    kwargs = {
        'model': AdvantageActorCritic,
        'environment': ENV_NAME,
        'num_parallel': 2,
        'total_steps': SAMPLE_SIZE,
        'returns': ['value_estimate', 'log_prob'],
        'action_sampling_type': 'continuous_normal_diagonal',
        'model_kwargs':model_kwargs
    }

    manager = SampleManager(**kwargs)

    # specify where to save results and ensure that the folder exists
    saving_path = Path(os.getcwd() + SAVING_DIRECTORY)
    saving_path.mkdir(parents=True, exist_ok=True)
    saving_path_actor = Path(os.getcwd() + SAVING_DIRECTORY + '/model/actor')
    saving_path_actor.mkdir(parents=True, exist_ok=True)
    saving_path_critic = Path(os.getcwd() + SAVING_DIRECTORY + '/model/critic')
    saving_path_critic.mkdir(parents=True, exist_ok=True)

    # initialize manager
    aggregator_keys=['actor_loss', 'critic_loss', 'time_steps', 'reward']
    manager.initialize_aggregator(path=saving_path, saving_after=5, aggregator_keys=aggregator_keys)

    # initialize the optimizers
    actor_optimizer = Adam(learning_rate=LEARNING_RATE_ACTOR)
    critic_optimizer = Adam(learning_rate=LEARNING_RATE_CRITIC)

    # get initial agent
    agent = manager.get_agent(test=JUST_TESTING)

    # load previous model
    if LOAD_MODELS:
        print('# ================ LOAD MODELS  ================= #')
        agent.model.actor = tf.keras.models.load_model(os.getcwd() + LOADING_PATH_ACTOR)
        agent.model.critic = tf.keras.models.load_model(os.getcwd() + LOADING_PATH_CRITIC)

        # update the weights of the manager
        manager.set_agent(agent.get_weights())
    else:
        START_EPOCH = 1

    if not JUST_TESTING:
        print('# =============== INITIAL TESTING =============== #')
        time_steps, rewards = manager.test(MAX_TEST_STEPS, 5, evaluation_measure='time_and_reward', render=True, do_print=True)

        print('# =============== START TRAINING ================ #')
        for e in range(START_EPOCH, EPOCHS + 1):
            print(f'# ============== EPOCH {e}/{EPOCHS} ============== #')
            print('# ============= collecting samples ============== #')
            # sample trajectories
            sample_dict = manager.sample(SAMPLE_SIZE, from_buffer=False)
            for k in sample_dict.keys():
                sample_dict[k] = tf.convert_to_tensor(sample_dict[k], dtype=tf.float32)

            # create a dataset from the samples
            data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=BATCH_SIZE)
            dataset = tf.data.Dataset.zip((data_dict['state'], data_dict['action'], data_dict['state_new'], data_dict['reward'], data_dict['not_done'], data_dict['log_prob'], data_dict['value_estimate']))

            print('# ================= optimizing ================== #')
            actor_losses = []
            critic_losses = []
            for s, a, ns, r, nd, lp, ve in dataset:

                # ensure that the datasets have at least 10 elements
                # otherwise we run into problems with the MSE loss
                if len(s) >= 10:
                    # calculate the advantage
                    value = r + nd * GAMMA * agent.v(ns)
                    advantage = value - ve

                    # train the networks
                    actor_loss = train_policy_network(agent, s, a, lp, advantage, actor_optimizer)
                    critic_loss = train_value_network(agent, s, value, critic_optimizer)

                    # save the losses
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

            print(f'actor loss:  {np.mean(actor_losses)}')
            print(f'critic loss: {np.mean(critic_losses)}')

            # update the weights of the manager
            manager.set_agent(agent.get_weights())

            print('# ================= validation ================== #')
            render = e % RENDER_EPISODES == 0
            time_steps, rewards = manager.test(MAX_TEST_STEPS, TEST_EPISODES, evaluation_measure='time_and_reward', render=render, do_print=False)
            manager.update_aggregator(actor_loss=actor_losses, critic_loss=critic_losses, time_steps=time_steps, reward=rewards)
            print(f'average reward:     {np.mean(rewards)}')
            print(f'average time steps: {np.mean(time_steps)}')

            if e % SAVE_EPISODES == 0:
                print('# ================= save models ================= #')
                agent.model.actor.save(os.path.join(saving_path_actor, f'epoch_{e}'))
                agent.model.critic.save(os.path.join(saving_path_critic, f'epoch_{e}'))

        print('# ============== TRAINING FINISHED ============== #')
        print('# ============== SAVE FINAL MODELS ============== #')
        agent.model.actor.save(os.path.join(saving_path_actor, f'final'))
        agent.model.critic.save(os.path.join(saving_path_critic, f'final'))

    print('# ================= FINAL TEST ================== #')
    manager.test(MAX_TEST_STEPS, 10, render=True, do_print=True, evaluation_measure='time_and_reward')
