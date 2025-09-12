#!/usr/bin/env python

import random
import logging
import sys
from collections import deque
from functools import partial
from math import floor, ceil

import tensorflow as tf
import keras as keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.simulation.simulation import Simulation, Vector2
from src.simulation.board_generator import make_simple_board

from src.model.model_utils import get_model_moves, get_model_scores, get_rotated_state

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)


# Parameters
view_type = "diagonal"
view_range = 7

replay_buffer_size = 50_000

model_name = "r7_simple_rb_1_3_e_100000_lr_001_timeout_diag_v2"

def calculate_score(ate_food: bool, died: bool):
    if died:
        return -10

    if ate_food:
        return 10

    return 0


# Model architecture
Dense = partial(keras.layers.Dense, activation="relu", kernel_initializer="he_normal")

online_model = keras.Sequential([
    keras.layers.Input((5 * 8,)),
    Dense(64),
    Dense(64),
    Dense(64),
    keras.layers.Dense(1)
])

target_model = keras.models.clone_model(online_model)
target_model.set_weights(online_model.get_weights())


# Logging config
logging.basicConfig(level=logging.DEBUG)

log_formatter = logging.Formatter("%(message)s")

std_handler = logging.StreamHandler(sys.stdout)
std_handler.setLevel(logging.INFO)
std_handler.setFormatter(log_formatter)

file_handler = logging.FileHandler(f"logs_{model_name}.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.handlers.clear()
#logger.addHandler(std_handler)
logger.addHandler(file_handler)


# 
simulation = Simulation(calculate_score, False)
simulation.reset(make_simple_board(np.array([view_range * 2 + 1, view_range * 2 + 1])), 2, 1, 1)

online_model.predict(np.array([simulation.get_snake_view(0, view_type, view_range)]))


# In[12]:


def epsilon_greedy(snakes_states: list[np.ndarray], snakes_possible_moves: list[list[Vector2]], snakes_alive: list[bool], epsilon: float=0.):
    n_snakes = len(snakes_states)

    process_mask = [True for _ in range(n_snakes)]
    
    for i, alive in enumerate(snakes_alive):
        if not alive or np.random.rand() < epsilon:
            process_mask[i] = False

    moves = get_model_moves(online_model, snakes_states, snakes_possible_moves, process_mask, softmax=True)

    for i, move in enumerate(moves):
        if move == (0, 0):
            moves[i] = random.choice(simulation.get_legal_moves(i))

    return moves


# In[13]:




# In[14]:


#priority_replay_buffer = deque(maxlen=replay_buffer_size // 2)
#replay_buffer = deque(maxlen=replay_buffer_size // 2)

replay_buffer = ReplayBuffer(replay_buffer_size // 2, replay_buffer_size // 2)


# In[15]:


def play_one_step(simulation: Simulation, states_before: list[np.ndarray], possible_moves_before: list[Vector2], snakes_alive_before: list[bool], epsilon: float):
    # Save states
    moves = epsilon_greedy(states_before, possible_moves_before, snakes_alive_before, epsilon)

    scores, running = simulation.next(moves)

    states_after = [simulation.get_snake_view(i, view_type, view_range) for i in range(simulation.n_snakes)]
    possible_moves_after = [simulation.get_legal_moves(i) for i in range(simulation.n_snakes)]
    snakes_alive_after = simulation.snakes_alive

    for i in range(len(states_before)):
        if snakes_alive_before[i]:
            replay_buffer.append((states_before[i], moves[i], scores[i], states_after[i], possible_moves_after[i], snakes_alive_after[i]))

    return states_after, possible_moves_after, snakes_alive_after, scores, running


# In[16]:


def sample_experiences(batch_size: int):
    idxs = np.random.randint(len(replay_buffer), size=batch_size)
    
    batch = [replay_buffer[idx] for idx in idxs]

    return [
        [experience[field_idx] for experience in batch]
        for field_idx in range(6)
    ]


# In[17]:


batch_size = 32
discount_factor = 0.9
loss_fn = keras.losses.mse
optimizer = keras.optimizers.Nadam(learning_rate=0.001)

losses = []


def training_step(buffer_split: float):
    #experiences = sample_experiences(batch_size)
    experiences = replay_buffer.sample(split=buffer_split, batch_size=batch_size)
    states, moves, scores, states_after, possible_moves_after, snakes_alive_after = experiences

    best_next_moves = [[move] for move in get_model_moves(online_model, states_after, possible_moves_after, snakes_alive_after)]
    next_q_values = get_model_scores(target_model, states_after, best_next_moves, snakes_alive_after)
    
    #next_q_values = get_model_scores(target_model, states_after, possible_moves_after, snakes_alive_after)
    runs = 1.0 - np.array(snakes_alive_after)
    
    target_q_values = scores + runs * discount_factor * next_q_values

    X = tf.constant([get_rotated_state(state, move) for state, move in zip(states, moves)])
    
    with tf.GradientTape() as tape:
        predicted_q_values = online_model(X)
        individual_losses = loss_fn(target_q_values, predicted_q_values)
        loss = tf.reduce_mean(individual_losses)

    replay_buffer.update_loss(individual_losses, loss)

    losses.append(loss.numpy())
    grads = tape.gradient(loss, online_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, online_model.trainable_variables))


# In[ ]:


simulation = Simulation(calculate_score, True)
rewards = []
steps = []
replay_buffer.clear()

best_model_window = 7
best_model_rewards = float("-inf")

n_episodes = 100_000

for episode in range(n_episodes):
    simulation.reset(make_simple_board(np.array([15, 15])), snake_count=1, food_count=1, tail_len=2)

    rewards.append(0)

    states = [simulation.get_snake_view(i, view_type, view_range) for i in range(simulation.n_snakes)]
    possible_moves = [simulation.get_legal_moves(i) for i in range(simulation.n_snakes)]
    snakes_alive = simulation.snakes_alive
    
    epsilon = max(1 - (episode / (n_episodes * 0.8)), 0.005)
    priority_buffer_split = min(episode / (n_episodes * 0.8) * 0.2, 0.2)

    step = 0
    
    while True:
        states, possible_moves, snakes_alive, scores, running = play_one_step(simulation, states, possible_moves, snakes_alive, epsilon)

        rewards[-1] += sum(scores)

        if not running:
            break

        step += 1

    steps.append(step)

    logging.info(f"Episode {episode} played")
    logging.info(f"\tSteps: {step}")
    logging.info(f"\tRewards: {rewards[-1]}")

    if episode > 200:
        curr_model_rewards = sum(rewards[-best_model_window:])
        if curr_model_rewards > best_model_rewards:
            avg_rewards = curr_model_rewards / best_model_window
            best_model_rewards = curr_model_rewards
            logging.info(f"\tSnaphshot saved (avg. {avg_rewards})")
            
            if model_name:
                online_model.save(f"models/{model_name}_{round(avg_rewards)}_snapshot.keras")

        training_step(priority_buffer_split)
        
        logger.info(f"\tPriority: {len(replay_buffer.priority_buffer)}")
        
        if episode % 100 == 0:
            target_model.set_weights(online_model.get_weights())
            logging.info("\tTarget model updated")


# In[ ]:


#sns.lineplot(steps, label="steps")
fig, ax = plt.subplots(figsize=(16,6))
sns.lineplot(np.convolve(rewards, [1 for _ in range(10)]))
plt.xlabel("Episode")
plt.ylabel("Reward")


# In[ ]:


fig, ax = plt.subplots(figsize=(16,6))
sns.lineplot(np.convolve(losses, [1 for _ in range(10)]), ax=ax)
plt.xlabel("Episode")
plt.ylabel("Loss")


# In[ ]:


fig, ax = plt.subplots(figsize=(16,6))
sns.lineplot(np.cumsum(rewards), label="rewards", ax=ax)
plt.xlabel("Episode")
plt.ylabel("Reward")


# In[ ]:


online_model.save(f"models/{model_name}.keras")

