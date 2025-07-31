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
EPISODES = 2000          # Number of training episodes
MAX_STEPS_PER_EPISODE = 100 # Max steps an agent can take in one episode
SOLVE_SCORE = 4.0        # Target score to consider the environment solved (adjust based on rewards)
PRINT_EVERY = 100        # How often to print progress

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
    # Removed: state = env._get_state() - initial state comes from env.reset() inside the loop

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
        state = env.reset() # Get initial state for the episode
        episode_reward = 0.0
        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action) # info is not used
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                break
        
        # adding episode reward to scores deque
        scores.append(episode_reward)

        # adding episode reward to all_scores list
        all_scores.append(episode_reward)

        # mean over last 100 scores
        avg_score = np.mean(scores)
        
        # Print progress
        print(f"Episode {episode}/{EPISODES} | Avg Score: {avg_score:.2f} | Epsilon: {agent.epsilon:.2f} | Initial Complexity: {env.initial_complexity}", end="")
        if episode % PRINT_EVERY == 0:
            print("") # New line for cleaner output

        # Check for solving condition
        if avg_score >= SOLVE_SCORE:
            print(f"Environment solved in {episode} episodes! Average score: {avg_score:.2f}")
            break # Stop training early

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    # TODO: You can add code here to save the trained model (e.g., torch.save(agent.qnet_policy.state_dict(), 'checkpoint.pth'))

if __name__ == "__main__":
    train_DQN()