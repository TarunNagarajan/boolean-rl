#-----------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
#-----------------------------------------------------------------------
EPISODES = 2000
# the total number of episodes to train the agent for.

MAX_STEPS_PER_EPISODE = 200
# the maximum number of steps the agent can take within a single episode.

SOLVE_SCORE = 60.0
# the average score over 100 consecutive episodes required to consider the
# environment solved.

#-----------------------------------------------------------------------
# ENVIRONMENT HYPERPARAMETERS
#-----------------------------------------------------------------------
MAX_EXPRESSION_DEPTH = 7
# the maximum depth of the randomly generated boolean expressions.

MAX_LITERALS = 7
# the number of unique literals (e.g., a, b, c) available for generation.

#-----------------------------------------------------------------------
# AGENT HYPERPARAMETERS
#-----------------------------------------------------------------------
AGENT_SEED = 0
# random seed for ensuring reproducibility.

HIDDEN_SIZE = 64
# the number of neurons in the hidden layers of the neural networks.

LEARNING_RATE = 5e-4
# the learning rate for the adam optimizer. it controls how much to change the
# model in response to the estimated error each time the weights are updated.

GAMMA = 0.99
# the discount factor for future rewards. a value close to 1 means the agent
# will care more about long-term rewards.

TAU = 1e-3
# the interpolation parameter for the soft update of the target network.
# it controls how much the target network is updated towards the policy network.

BUFFER_SIZE = int(1e5)
# the maximum number of experiences to store in the replay buffer.

BATCH_SIZE = 64
# the number of experiences to sample from the replay buffer for each learning step.

UPDATE_EVERY = 4
# the frequency (in steps) at which the agent learns from the replay buffer.
