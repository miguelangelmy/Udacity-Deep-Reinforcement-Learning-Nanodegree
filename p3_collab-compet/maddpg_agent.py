import numpy as np
import random

from maddpg_model import Actor, Critic
from utils import OUNoise, save_model_checkpoint, load_model_checkpoint

import torch
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, agent_index, num_agents, state_size, action_size, hidden_layers, batch_size, gamma, tau, lr_actor, lr_critic, weight_decay=0.0, random_seed=777):
        """Initialize an Agent object.
        
        Params
        ======
            num_agents (int): Number of agents (for the critic input size)
            agent_index (int): Agent index        
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            hidden_layers: tuple of integers with the sizes of the hidden layers, allow us to generate hidden layers in a flexible way
            batch_size (int): size for the training batch
            gamma (float): Discount factor
            tau (float): Parameter for soft update of Q_target             
            lr_actor (float): learning rate for the actor
            lr_critic (float): learning rate for the critic
            random_seed (int): random seed
        """
        self.num_agents = num_agents
        self.agent_index = agent_index
        self.agent_name = str(agent_index)
        self.state_size = state_size
        self.action_size = action_size
        self.critic_input_size = num_agents * (state_size + action_size)
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
                
        self.seed = random_seed
        random.seed(random_seed)

        # Each actor will have its own actor network but all of them will share the same critic network (in the manager)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, hidden_layers, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, hidden_layers, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)       
            
        # Critic Network (w/ Target Network)
        
        self.critic_local = Critic(self.critic_input_size, hidden_layers, random_seed).to(device)
        self.critic_target = Critic(self.critic_input_size, hidden_layers, random_seed).to(device)        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)           
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        
        self.reset()

        
    def reset(self):
        """
        Resets the weights for the network of the actor (local and target).
        Resets the noise function.
        """        
        self.actor_local.reset_parameters();
        self.actor_target.reset_parameters();
        self.noise.reset()             


    def act(self, state, add_noise=False, n_weight=1.0):
        """
        Returns the values for the action array given a specific state. The values returned will be clipped by (-1, 1)

        Params
        ======
            state (array_like): state
            add_noise (boolean): specifies if noise will be added
            n_weight = Noise weighting factor           
        Returns
        =======              
        Actions for given state as per current policy.
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += n_weight * self.noise.sample()
        return np.clip(action, -1, 1)
    
    
    def learn(self, agent_id, experiences, all_actions, all_next_actions):
        """
        Update the Q table for the critic and the policy for the actor using given batch of experience tuples.
 
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, full_obs) tuples 
        """
        states, actions, rewards, next_states, dones = experiences # Each one is an array with the values for each agent

        # get predicted next-state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id]).to(device)
        actions_next = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, actions_next)


        # Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # q_targets = reward of this timestep + discount * Q(st+1,at+1) from target network
        Q_targets = rewards.index_select(1, agent_id) + (self.gamma * Q_targets_next * (1 - dones.index_select(1, agent_id)))        
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())       
        # Minimize the loss0
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        self.actor_optimizer.zero_grad()
                
        actions_pred = [action if i == self.agent_index else action.detach() for i, action in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(states, actions_pred).mean()        
        
        
        # Minimize the loss
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)           


    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def save_checkpoint(self, path):
        """
        Save the models for the agent to disk as a checkpoint with the specified path and name prefix, automatically add the timestamp as part of the name to avoid overwritings 
            
        Params
        ======
            self: agent
            path (string): path to the file
        """ 
        
        save_model_checkpoint(self.critic_local, path=path, name_prefix=f"{self.agent_name}_critic_local_")
        save_model_checkpoint(self.critic_target, path=path, name_prefix=f"{self.agent_name}_critic_target_")
        
        save_model_checkpoint(self.actor_local, path=path, name_prefix=f"{self.agent_name}_actor_local_")
        save_model_checkpoint(self.actor_target, path=path, name_prefix=f"{self.agent_name}_actor_target_")
        
        
    def load_checkpoint(self, path, timestamp):
        """ Load the models for the agent from the specified file path and timestamp
                    
        Params
        ======
            self (dictionary): agent
            path (string): path to the file
            time_stamp (string): timestamp sufix
        """        
        
        load_model_checkpoint(self.critic_local, path=path, file_name=f"{self.agent_name}_critic_local_{timestamp}")
        load_model_checkpoint(self.critic_target, path=path, file_name=f"{self.agent_name}_critic_target_{timestamp}")
        
        load_model_checkpoint(self.actor_local, path=path, file_name=f"{self.agent_name}_actor_local_{timestamp}")
        load_model_checkpoint(self.actor_target, path=path, file_name=f"{self.agent_name}_actor_target_{timestamp}") 