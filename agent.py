import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_size):
        """
        state_size(int): dimension of each state (input features)
        action_size(int): dimension of each action (output Q-vals)
        seed(int): random seed for reproducibility
        hidden_size(int): number of nodes in the hidden layer
        
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        # network that maps state -> action values
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    fixed size buffer to store experience tuples
    """

    def __init__(self, buffer_size, batch_size, seed, device):
        """
        buffer_size (int): max size of buffer
        batch_size (int): size of each training batch   
        seed(int): random seed for reproducibility
        device (torch.device)
        """

        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = collections.namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k = self.batch_size)
        
        # vstack: concatenation along the first axis after 1-D arrays of shape (N,) have been reshaped to (1,N). 
        # rebuilds arrays divided by vsplit
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    # agent that interacts with and learns from the environment
    def __init__(self,
                 state_size,
                 action_size,
                 seed, 
                 hidden_size = 64,
                 learning_rate = 5e-4,
                 gamma = 0.99,
                 tau = 1e-3,
                 buffer_size = int(1e5), 
                 batch_size = 64,
                 update_every = 4):
        """
        Initializes a DQN Agent

        state_size(int): dimension of each state (from env) 
        action_size(int): dimension of each action (from env)
        seed(int): random seed for reproducibility
        hidden_size(int): hidden layers of the Q-network
        learning_rate(float): optimizer's lr    
        gamma(float): discount factor for future rewards
        tau(float): for soft update of target network params
        buffer_size(int): maximum size of the replay buffer
        batch_size(int): size of the training batch
        update_every(int): how often to update the target network (steps)
        
        """

        self.state_size = state_size
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # qnet_target provides stable targets for the policy network 
        # weights are updated less frequently and more smoothly (via tau)
        self.qnet_policy = QNetwork(state_size, action_size, seed, hidden_size).to(self.device)
        self.qnet_target = QNetwork(state_size, action_size, seed, hidden_size).to(self.device)

        # initialize qnet_target's weights to be same as policy networks.
        self.qnet_target.load_state_dict(self.qnet_policy.state_dict())
        self.qnet_target.eval()

        # update the weights of the policy network based on the calculated loss
        self.optimizer = optim.Adam(self.qnet_policy.parameters(), lr = self.learning_rate)

        # replay buffer 
        # stores (state, action, reward, next_state, done) tuples
        self.memory = ReplayBuffer(buffer_size, batch_size, seed, self.device)

        # epsilon-greedy params
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # tracks the number of steps taken
        self.t_step = 0