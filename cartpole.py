import logging, os
from pathlib import Path

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym
import numpy as np
import ray
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
import time
from really import SampleManager
import os
from really.utils import dict_to_dict_of_datasets

# ===================== DEFINE GLOBAL (HYPER)PARAMETERS ====================== #
# general parameters
ENV_NAME = 'CartPole-v0'
SAVING_DIRECTORY = '/logging/cartpole'

# parameters for training
EPOCHS = 100
LEARNING_RATE = 0.001
SAMPLE_SIZE = 1000
BATCH_SIZE = 64
GAMMA = 0.9
BUFFER_SIZE = 10000
EPSILON = 0.95

# parameters for testing
TEST_EPISODES = 20
MAX_TEST_STEPS = 500
RENDER_EPISODES = 5
SAVE_EPISODES = 5

class VanillaDeepQNetwork(Model):

    def __init__(self, observation_space, action_space):
        super(VanillaDeepQNetwork, self).__init__()

        # initialize the spaces
        self.observation_space = observation_space
        self.action_space = action_space

        # create the deep q network
        self.deep_q_net = self._create_deep_q_model()


    def __call__(self, state, **kwargs):
        q_vals = self.deep_q_net(state)
        output = {}
        output['q_values'] = q_vals
        return output


    def _create_deep_q_model(self):
        state_input = Input(shape=self.observation_space)

        # create the hidden layers
        drive_1 = Dense(64, activation='relu', kernel_initializer='he_uniform')(state_input)
        drive_2 = Dense(32, activation='relu', kernel_initializer='he_uniform')(drive_1)

        # create the output layer
        q_vals = Dense(self.action_space, activation='linear', kernel_initializer='he_uniform')(drive_2)
        return Model(inputs=state_input, outputs=q_vals)


# ============================== UPDATE METHODS ============================== #

@tf.function
def train_q_network(agent, state, action, reward, next_state, not_done, optimizer):
    """Trains the Q-network."""

    q_target = tf.cast(reward, tf.float32) + tf.cast(not_done, tf.float32) * GAMMA * agent.max_q(next_state)

    with tf.GradientTape() as tape:
        q_vals = agent.q_val(state, action)
        loss = MSE(q_target, q_vals)
        gradients = tape.gradient(loss, agent.model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))
    return loss


# ============================== TRAINING LOOP =============================== #

if __name__ == '__main__':
    ray.init(log_to_driver=False)

    # define the model and sample manager arguments
    model_kwargs = {
        'observation_space': 4,
        'action_space': 2
    }

    kwargs = {
        'model': VanillaDeepQNetwork,
        'environment': ENV_NAME,
        'num_parallel': 2,
        'total_steps': SAMPLE_SIZE,
        'model_kwargs':model_kwargs,
        "action_sampling_type": "epsilon_greedy",
        "epsilon": EPSILON
    }

    manager = SampleManager(**kwargs)

    # specify where to save results and ensure that the folder exists
    saving_path = Path(os.getcwd() + SAVING_DIRECTORY)
    saving_path.mkdir(parents=True, exist_ok=True)
    saving_path_model = Path(os.getcwd() + SAVING_DIRECTORY + '/model')
    saving_path_model.mkdir(parents=True, exist_ok=True)

    # initialize manager
    optim_keys = ['state', 'action', 'reward', 'state_new', 'not_done']
    manager.initilize_buffer(BUFFER_SIZE, optim_keys)
    aggregator_keys=['loss', 'time_steps', 'reward']
    manager.initialize_aggregator(saving_path, 5, aggregator_keys)

    # initialize the optimizers
    optimizer = Adam(learning_rate=LEARNING_RATE)

    print('# =============== INITIAL TESTING =============== #')
    manager.test(MAX_TEST_STEPS, 5, evaluation_measure='time_and_reward', do_print=True, render=True)

    # get the initial agent
    agent = manager.get_agent()

    print('# =============== START TRAINING ================ #')
    for e in range(1, EPOCHS+1):
        print(f'# ============== EPOCH {e}/{EPOCHS} ============== #')
        print('# ============= collecting samples ============== #')
        # collect experience and save it to ERP buffer
        data = manager.get_data(do_print=False)
        manager.store_in_buffer(data)

        # get some samples from the ERP buffer and create a dataset
        sample_dict = manager.sample(sample_size=SAMPLE_SIZE)
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=BATCH_SIZE)
        dataset = tf.data.Dataset.zip((data_dict['state'], data_dict['action'], data_dict['reward'], data_dict['state_new'], data_dict['not_done']))

        print('# ================= optimizing ================== #')
        losses = []
        for s, a, r, ns, nd in dataset:

            # ensure that the datasets have at least 10 elements
            # otherwise we run into problems with the MSE loss
            if len(s) >= 10:
                loss = train_q_network(agent, s, a, r, ns, nd, optimizer)
                losses.append(loss)

        print(f'average loss: {np.mean(losses)}')

        # update the weights of the manager
        manager.set_agent(agent.get_weights())

        print('# ================= validation ================== #')
        render = e % RENDER_EPISODES == 0
        time_steps, rewards = manager.test(MAX_TEST_STEPS, TEST_EPISODES, evaluation_measure='time_and_reward', render=render, do_print=False)
        manager.update_aggregator(loss=losses, time_steps=time_steps, reward=rewards)
        print(f'average reward:     {np.mean(rewards)}')
        print(f'average time steps: {np.mean(time_steps)}')

        if e % SAVE_EPISODES == 0:
            print('# ================= save model ================== #')
            agent.model.deep_q_net.save(os.path.join(saving_path_model, f'epoch_{e}'))

    print('# ============== TRAINING FINISHED ============== #')
    print('# ============== SAVE FINAL MODELS ============== #')
    agent.model.deep_q_net.save(os.path.join(saving_path_model, f'final'))

    print('# ================= FINAL TEST ================== #')
    manager.test(MAX_TEST_STEPS, 10, render=True, do_print=True, evaluation_measure='time_and_reward')
