import gym
import numpy as np
import ray
import tensorflow as tf
import time
from really import SampleManager
from gridworlds import GridWorld

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""


class TabularQ(object):
    def __init__(self, h, w, action_space):
        self.action_space = action_space
        self.h = h
        self.w = w
        self.q_table = np.zeros([self.h ,self.w, self.action_space])
        pass

    def __call__(self, state):
        output = {}
        output["q_values"] = np.expand_dims(self.q_table[tuple(np.squeeze(state.astype(np.int32)))], axis=0)
        return output

    def get_weights(self):
        return self.q_table.copy()

    def set_weights(self, q_vals):
        self.q_table = q_vals


if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 10,
        "width": 10,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
    }

    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 2,
        "total_steps": 100,
        "model_kwargs": model_kwargs,
        "env_kwargs" : env_kwargs
        # and more
    }

    # initilize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    print("test before training: ")
    manager.test(
        max_steps=100,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )

    # some parameters
    epochs = 100
    gamma = 0.85
    learning_rate = 0.2

    # get the initial agent
    agent = manager.get_agent()

    print("training: ")
    for e in range(epochs):
        print(f"start epoche {e+1}")
        start_time = time.time()
        step_counter = 0
        # select the start state
        env = manager.env_instance
        state_new = np.expand_dims(env.reset(), axis=0)
        done = False

        while not done:
            state = state_new
            env.render()
            step_counter += 1

            # choose the next action
            action = agent.act(state)
            if tf.is_tensor(action):
                action = action.numpy()

            # get the next state and the reward
            state_new, reward, done, _ = env.step(action)
            state_new = np.expand_dims(state_new, axis=0)

            # calculate the new q values
            state_action_q_value = np.squeeze(agent.model(state)['q_values'])[action]
            state_new_max_action_q_value = np.max(agent.model(state_new)['q_values'])
            new_state_action_q_value = state_action_q_value + learning_rate * (reward + gamma * state_new_max_action_q_value - state_action_q_value)

            # update the weights of the agent
            new_weights = agent.get_weights()
            new_weights[tuple(np.squeeze(state))][action] = new_state_action_q_value

            manager.set_agent(new_weights)
            agent.set_weights(new_weights)
            # we set the weights directly at the agent because creating
            # a new agent every timestep slows down the training a lot
            # agent = manager.get_agent()

        time_delta = time.time() - start_time
        if ((e+1)%10 == 0):
            print(f"test after epoche {e+1}:")
            manager.test(max_steps=100, test_episodes=10, render=True, do_print=True, evaluation_measure="time_and_reward")
        print(f"finished epoche {e+1} \nDuration: {time_delta} \nSteps: {step_counter}")
        print("---")

    print("done")
    print("testing optimized agent")
    manager.test(max_steps=100, test_episodes=10, render=True, do_print=True, evaluation_measure="time_and_reward")
