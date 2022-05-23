import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import math
import random
import copy
from datetime import datetime

from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, path, name_prefix):
    """
    Save the model to disk as a checkpoint with the specified path and name prefix, automatically add the timestamp as part of the name to avoid overwritings 
        
    Params
    ======
        model (dictionary): state of the model
        path (string): path to the file
        name_prefix (string): Name prefix
    """        
    
    checkpoint = {'state_size': model.state_size,
                'action_size': model.action_size,
                'hidden_layers': model.hidden_layers,
                'seed': model.seed,
                'state_dict': model.state_dict()}
    torch.save(checkpoint, os.path.join(path, name_prefix+str(math.trunc(datetime.timestamp(datetime.now())))+".pth"))
    
    
def load_checkpoint(model, path, file_name):
    """ Load the model from the specified file path
                
    Params
    ======
        model (dictionary): state of the model
        path (string): path to the file
        file_name (string): file name        
    """        
    checkpoint = torch.load(os.path.join(path, file_name))
    model.__init__(checkpoint['state_size'], checkpoint['action_size'], checkpoint['hidden_layers'], checkpoint['seed'])
    model.load_state_dict(checkpoint['state_dict'])    
        
class OUNoise():
    """
    Adds noise to continuous actions using the Ornstein-Uhlenbeck process.

    Params
    ======
        size (array_like): actions vector dimensions
        seed (int): random seed
        mu (float): is the asymptotic mean for the process, the value wich the process tends towards
        theta (float): is the decay-rate or growth-rate
        sigma (float): is the variation, or the "size" of the noise
    """
        
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """
        Initialize parameters and noise process.
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = seed
        random.seed(seed)
        self.reset()

    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu).
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        Update internal state and return it as a noise sample.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

def plot_scores(scores, rolling_window=100):
    '''
    Plot the scores and their moving average on the same chart
    
    Params
    ======
        scores (array_like): array with the scores
        rolling_window (Integer): Window size for the rolling mean
    '''
    
    plt.plot(np.arange(len(scores)), scores, '-c', label='Episode score')
    plt.title('Episodic Score')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(np.arange(len(scores)), rolling_mean, '-r', label='Rolling mean')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
       
class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed for the sampling
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)