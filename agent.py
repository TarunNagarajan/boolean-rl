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

    # given a state, return an action (the index of the rule, that needs to be applied)
    def act(self, state):
        # state (np.array): current state observation from the environment
        if (random.random() < self.epsilon): # exploration case
            return random.randrange(self.action_size)
        
        else: 
            # exploitation case (use qnet_policy for greedy selection)
            state_tensor = torch.from_numpy(state).float()

            # expects a batch of inputs -> add an extra dimension to the state_tensor
            state_tensor = state_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                # tensor of Q values for all actions
                qval_tensor = self.qnet_policy(state_tensor)

            return qval_tensor.argmax(1).item() # selects the best action and returns it
      
    # experiences = (states, actions, rewards, next_states, dones)
    def learn(self, experiences):
        # update the weights of policy network to better predict Q-values, based on bellman eqn.
        # also updates the target network
        states, actions, rewards, next_states, dones = experiences
        next_states = next_states.to(self.device)

        with torch.no_grad():
            # we don't want to compute gradients for the target network
            qval_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
            qval_targets_curr = rewards + (self.gamma * qval_targets_next * (1 - dones))

        qval_expected = self.qnet_policy(states).gather(1, actions)
        # gather (1, actions is to select the Q-value for the specific action taken from the output of the network)

        loss = F.mse_loss(qval_expected, qval_targets_curr)

        # zero out old gradients
        self.optimizer.zero_grad()

        # backprop to compute gradients of the loss wrt policy network weights
        loss.backward()

        # take an optimizer step to update the policy network weights
        self.optimizer.step() 

        # update target network parameters with polyak averaging
        for target_param, policy_param in zip(self.qnet_target.parameters(), self.qnet_policy.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

        # gradual epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # step method called by the training loop after the agent has taken its action in the environment and received reward
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1

        # check if
        # 1. there is enough data in the replay buffer to sample another batch of training
        # 2. have reached the threshold for updation (remember self.every...)

        if (self.t_step % self.update_every == 0) and (len(self.memory) >= self.batch_size):
            experiences = self.memory.sample()
            self.learn(experiences)
            self.t_step = 0
    

            