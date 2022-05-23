import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_layers, seed=777):
    
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers: list of integers with the sizes of the hidden layers, allow us to generate hidden layers in a flexible way
            seed (int): Random seed
        """
        super(Actor, self).__init__()             
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        self.seed = seed
        torch.manual_seed(seed)
        
        # Input layer
        self.layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])        
        
         # Hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
               
        # Output layer
        self.output = nn.Linear(hidden_layers[-1], action_size)        
               
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()                             
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        for linear in self.layers:
            x = F.leaky_relu(linear(x))        
        return F.tanh(self.output(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_layers, seed=777):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers: list of integers with the sizes of the hidden layers, allow us to generate hidden layers in a flexible way
            seed (int): Random seed
        """
        super(Critic, self).__init__()        
        self.state_size = state_size
        self.action_size = action_size        
        self.hidden_layers = hidden_layers
        self.seed = seed
        torch.manual_seed(seed)
        
        # Input layer (The input will be the state plus the action)
        self.layers = nn.ModuleList([nn.Linear(state_size+action_size, hidden_layers[0])])        
        
        # Hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        print(f"critic_layers:{self.layers}")
               
        # Output layer
        self.output = nn.Linear(hidden_layers[-1], 1)            
               
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()                
        self.output.weight.data.uniform_(-3e-3, 3e-3)       

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=1)
        for linear in self.layers:
            x = F.leaky_relu(linear(x))    
        return self.output(x)
