import io
import gym
import time
import math
import glob
import retro
import base64
import random
import matplotlib
import numpy as np
import retrowrapper
import tensorflow as tf
import multiprocessing as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from gym import wrappers
from itertools import count
from collections import deque
from gym.wrappers import Monitor
# from IPython.display import HTML
from gym import logger as gymlogger
from baselines.common.retro_wrappers import *


# set parameters
class ModelParameters():
    def __init__(self):
        self.GAME_NAME = "SonicTheHedgehog-Genesis"
        self.SEED = 42
        self.GAMMA = 0.99                                  # Discount factor for past rewards
        self.EPSILON = 1.0                                 # Epsilon greedy parameter
        self.EPSILON_MIN = 0.1                             # Minimum epsilon greedy parameter
        self.EPSILON_MAX = 1.0                             # Maximum epsilon greedy parameter
        self.EPSILON_INTERVAL = self.EPSILON_MAX - self.EPSILON_MIN  # Rate at which to reduce chance of random action being taken
        self.BATCH_SIZE = 32                               # Size of batch taken from replay buffer
        self.MAX_STEPS_PER_EPISODE = 10000
        self.EPSILON_RANDOM_FRAMES = 50000                 # Number of frames to take random action and observe output
        self.EPSILON_GREEDY_FRAMES = 1000000.0             # Number of frames for exploration
        self.MAX_MEMORY_LENGTH = 100000                    # Maximum replay length (Note: The Deepmind paper suggests 1000000 
                                                           # however this causes memory issues)
        self.UPDATE_AFTER_ACTIONS = 4                      # Train the model after 4 actions
        self.UPDATE_TARGET_NETWORK = 10000                 # How often to update the target network
        self.NUM_ACTIONS = 7                               # number of actions
        self.FRAME_COUNT = 0                               # total played frame


param = ModelParameters()


# memory and memory buffer
class Memory():
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplyBuffer():
    def __init__(self, maxlen=param.MAX_MEMORY_LENGTH):
        self.buffer = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.buffer)

    def get_batch_sample(self):
        # Get indices of samples for replay buffers
        indices = np.random.choice(range(len(self.buffer)), size=param.BATCH_SIZE)

        # Using list comprehension to sample from replay buffer
        state_sample = np.array([self.buffer[i].state for i in indices])
        next_state_sample = np.array([self.buffer[i].next_state for i in indices])
        rewards_sample = [self.buffer[i].reward for i in indices]
        action_sample = [self.buffer[i].action for i in indices]
        done_sample = tf.convert_to_tensor(
            [float(self.buffer[i].done) for i in indices]
        )
        return state_sample, action_sample, rewards_sample, next_state_sample, done_sample


reply_buffer = ReplyBuffer()


# models
def get_model_and_target_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(84, 84, 4)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=8, strides=4, padding="same", use_bias=True, activation="relu"
            ),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=4, strides=2, padding="same", use_bias=True, activation="relu"
            ),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=1, padding="same", use_bias=True, activation="relu"
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=512, activation="relu", use_bias=True),
            tf.keras.layers.Dense(units=param.NUM_ACTIONS)
        ]
    )
    target_model = tf.keras.models.clone_model(model)
    return model, target_model


model, target_model = get_model_and_target_model()


# env
env = None


def get_env():
    global env
    if env is not None:
        env.close()
    env = retro.make(game=param.GAME_NAME)
    env = wrap_deepmind_retro(env, scale=True, frame_stack=4)
    env = SonicDiscretizer(env)
    # env = AllowBacktracking(env)
    return env


env = get_env()


def select_action(state):
    if param.FRAME_COUNT < param.EPSILON_RANDOM_FRAMES or param.EPSILON > np.random.rand(1)[0]:
        # Take random action
        action = np.random.choice(param.NUM_ACTIONS)
    else:
        # Predict action Q-values
        # From environment state
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()
    return action


# loss and optimizer
huber_loss = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)


def train_step():
    state_sample, action_sample, rewards_sample, next_state_sample, done_sample = reply_buffer.get_batch_sample() 

    # Build the updated Q-values for the sampled future states
    # Use the target model for stability
    future_rewards = target_model.predict(next_state_sample)

    # Q value = reward + discount factor * expected future reward
    updated_q_values = rewards_sample + param.GAMMA * tf.reduce_max(future_rewards, axis=1)

    # If final frame set the last value to -1
    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

    # Create a mask so we only calculate loss on the updated Q-values
    masks = tf.one_hot(action_sample, param.NUM_ACTIONS)

    with tf.GradientTape() as tape:
        # Train the model on the states and updated Q-values
        q_values = model(state_sample)

        # Apply the masks to the Q-values to get the Q-value for action taken
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        # Calculate loss between new Q-value and old Q-value
        # Clip the deltas using huber loss for stability
        loss = huber_loss(updated_q_values, q_action)

    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def decay_epsilon():
    param.EPSILON -= param.EPSILON_INTERVAL / param.EPSILON_GREEDY_FRAMES
    param.EPSILON = max(param.EPSILON, param.EPSILON_MIN)


def train_loop():
    episode_reward_history = []
    episode_count = 0
    running_reward = 0

    checkpoint = tf.train.Checkpoint(model=model, target_model=target_model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, "./checkpoints", max_to_keep=10)
    train_summary_writer = tf.summary.create_file_writer("./log")

    while True:  # Run until solved
        state = np.array(env.reset())
        episode_reward = 0
        for timestep in range(1, param.MAX_STEPS_PER_EPISODE):
            param.FRAME_COUNT += 1
            action = select_action(state)
            decay_epsilon()

            # Apply the sampled action in our environment
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state)

            episode_reward += reward

            # add to reply buffer
            reply_buffer.buffer.append(
                Memory(state=state, action=action, reward=reward, next_state=next_state, done=done)
            )

            state = next_state

            if param.FRAME_COUNT % param.UPDATE_AFTER_ACTIONS == 0 and len(reply_buffer) > param.BATCH_SIZE:
                loss = train_step()

            if param.FRAME_COUNT % 10000 == 0:
                # update the the target network with new weights
                target_model.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, param.FRAME_COUNT))
                if param.FRAME_COUNT % 1000000 == 0:
                    manager.save()
                    model.save("./latest_checkpoint", overwrite=True, include_optimizer=True, save_format="tf")

            if done:
                break
        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        with train_summary_writer.as_default():
            tf.summary.scalar("moving mean reward", running_reward, step=episode_count)

        episode_count += 1

        if running_reward > 40:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
    return


def render_trained_model():
    # if you want render model on server follow these steps:
    # 1) pip install gym pyvirtualdisplay > /dev/null 2>&1
    #    apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
    # 2) https://github.com/openai/gym/issues/637#issuecomment-315601151
    global env
    env = wrappers.Monitor(env, './videos/', force=True)

    pre_trained_model = tf.keras.models.load_model("./latest_checkpoint")
    print(model.summary())

    state = np.array(env.reset())
    cnt = 1
    while True:  # Run until solved
        episode_reward = 0
        for timestep in range(1, param.MAX_STEPS_PER_EPISODE):
            param.FRAME_COUNT += 1

            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = pre_trained_model(state_tensor, training=False)
            # Take best action
            if cnt < 400:
                # action = np.random.choice(param.NUM_ACTIONS)
                action = 1
            else:
                action = tf.argmax(action_probs[0]).numpy()
            cnt += 1

            next_state, reward, done, _ = env.step(action)
            # print(action)
            next_state = np.array(next_state)

            episode_reward += reward

            state = next_state

            if done:
                break

            if timestep % 100 == 0:
                print(f"timestep: {timestep}, episode_reward: {episode_reward}")

            env.render()
        break

    env.close()


def main():
    train_loop()
    # render_trained_model()


if __name__ == "__main__":
    main()
