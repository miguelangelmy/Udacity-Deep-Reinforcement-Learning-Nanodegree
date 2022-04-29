import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Model for the Q-Network"""

    def __init__(self, state_size, action_size, hidden_layers, drop_p=0.5):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers: list of integers with the sizes of the hidden layers, allow us to generate hidden layers in a flexible way
            drop_p: float between 0 and 1, probabiliby for the dropout layers
        """
        
        # Pytorch initializes by default the weights and biases of the tensors. In the case of Linear type, for example, the following applies:
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)
        
        super(QNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        
        # Input layer
        self.layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # Output layer
        self.output = nn.Linear(hidden_layers[-1], action_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=drop_p)


    def forward(self, state):
        """Build a network that maps state -> action values.
    
        Params
        ======
            state (array_like): current state
        """        
        
        # The incoming state is passed through the successive layers of the network
        # Added interleaved dropout layers to avoid overfitting
        x = state
        for linear in self.layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
            
        # Finally the forward function returns what is generated at the output layer
        return self.output(x)
    
    
    def save(self, file_name):
        """ Save the model to disk as a checkpoint with the specified name
            
        Params
        ======
            file_name (string): file name
        """        
        
        checkpoint = {'state_size': self.state_size,
                      'action_size': self.action_size,
                      'hidden_layers': self.hidden_layers,
                      'state_dict': self.state_dict()}
        torch.save(checkpoint, file_name)
        
        
    def load(self, file_path):
        """ Load the model from the specified file path
                    
        Params
        ======
            file_path (string): file path
        """        
        
        checkpoint = torch.load(file_path)
        self.__init__(checkpoint['state_size'], checkpoint['action_size'], checkpoint['hidden_layers'])
        self.load_state_dict(checkpoint['state_dict'])