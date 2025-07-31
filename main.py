import boolean_env
import agent
from boolean_env import BooleanSimplificationEnv
from agent import DQNAgent

import numpy as np
import torch
import collections
from collections import deque
import time

""" HYPER-PARAMETERS 
"""

# training hyperparams
EPISODES = 2000
MAX_STEPS_PER_EPISODE = 100
SOLVE_SCORE = 4.0
PRINT_EVERY = 100

# env hyperparams
MAX_EXPRESSION_DEPTH = 3
MAX_LITERALS = 4 

# agent hyperparams
AGENT_SEED = 0
HIDDEN_SIZE = 64
LEARNING_RATE = 5e-4
GAMMA = 0.99
TAU = 1e-3
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
UPDATE_EVERY = 4

def train_DQN():
    env = BooleanSimplificationEnv(max_expression_depth = MAX_EXPRESSION_DEPTH, 
                                   max_literals = MAX_LITERALS,
                                   max_steps = MAX_STEPS_PER_EPISODE)
    
    state_size = env.get_state_size() 
    action_size = env.get_action_size() 

    agent = DQNAgent(
        state_size = state_size,
        action_size = action_size,
        seed = AGENT_SEED,
        hidden_size = HIDDEN_SIZE,
        learning_rate = LEARNING_RATE,
        gamma = GAMMA,
        tau = TAU, 
        buffer_size = BUFFER_SIZE, 
        batch_size = BATCH_SIZE,
        update_every = UPDATE_EVERY
    )

    # training loop variables
    scores = deque(maxlen = 100)
    all_scores = []
    start_time = time.time()

    for episode in range(1, EPISODES + 1): 
        # TODO: complete main episode loop
        pass

    end_time = time.time()
    print(f'\nTraining finished in {end_time - start_time:.2f} seconds')
    # TODO: save model

if __name__ == "main":
    train_DQN()




