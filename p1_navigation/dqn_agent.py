import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size=4, action_size=2, hidden_layers=None, drop_p=0, buffer_size=100, batch_size=10, gamma=0.99, tau=1e-3, lr=5e-4, update_every=4, seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            hidden_layers(array of ints): array used for the sizes the hidden layers
            drop_p (float): probability for the dropout layers
            buffer_size (int): size of the replay memory
            batch_size (int): size for the batch retrieved from the replay memory
            gamma (float): discount factor
            tau (float): for the target Q-Network soft update
            lr (float): learning rate
            update_every (int): update frequency for the load from the replay memory
            seed (int): random seed
        """

        self.action_size=action_size
        self.hidden_layers = [1] if hidden_layers is None else hidden_layers
        self.batch_size=batch_size
        self.gamma=gamma
        self.tau=tau
        self.update_every=update_every
        random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, self.hidden_layers, drop_p).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.hidden_layers, drop_p).to(device)
        
        # The target Q-Network is initialized with the values initialized in the local one
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
                
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayMemory(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Put the model in evaluation mode and deactivate the gradient calculation so that the model applies inference
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state)          
        # After the inference process, switch back to training mode
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def step(self, state, action, reward, next_state, done):
        """Add experiences to the replay memory.
        
        Params
        ======
            state (array_like): current state
            action (array_like): current action
            reward (float): obtained reward
            next_state (array_like): next state
            done (boolean): indicates if the episode has ended
        """        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step += 1
        if self.t_step % self.update_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
            self.t_step = 0 # Counter is reset to avoid overflows
                
                
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (tuple): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
                        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update for the target model parameters.
        
        Params
        ======
            local_model (PyTorch model): weights of the local Q-Network
            target_model (PyTorch model): weights of the target Q-Network
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayMemory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayMemory object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = seed
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)