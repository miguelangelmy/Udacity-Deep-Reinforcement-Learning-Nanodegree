import numpy as np
import random
import os

from maddpg_agent import MADDPGAgent
from maddpg_replaybuffer import ReplayBuffer

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NOISE_DECAY = 1.0               # Decay for the noise   
NOISE_START = 0.5               # Initial value for the noise weighting factor 
NOISE_STOP_POINT = 30000        # T-step for the noise to stop   
    
class MADDPGManager():

    """Manage the multi-agent scenario."""
    
    def __init__(self, num_agents, state_size, action_size, hidden_layers, buffer_size, batch_size, update_frequency, gamma, tau, lr_actor, lr_critic, weight_decay,  random_seed):
        """Initialize an Agent object.
            
        Params
        ======
            num_agents (int): Number of agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            hidden_layers: tuple of integers with the sizes of the hidden layers, allow us to generate hidden layers in a flexible way
            buffer_size (int): size for the replay memory
            batch_size (int): size for the training batch
            update_frequency (int): how often the learning process is launched
            gamma (float): Discount factor
            tau (float): Parameter for soft update of Q_target 
            lr_actor (float): learning rate for the actor
            lr_critic (float): learning rate for the critic            
            weight_decay (float): L2 weight decay         
            random_seed (int): random seed
        """
        
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.gamma = gamma
        self.tau = tau
        self.n_start = NOISE_START
        self.n_weight = NOISE_START
        self.n_decay = NOISE_DECAY
        self.n_stop_point = NOISE_STOP_POINT
        self.n_counter = 0
        self.seed = random_seed
        random.seed(random_seed)
        
        self.agents = [MADDPGAgent(x, self.num_agents, self.state_size, self.action_size, self.hidden_layers, self.batch_size, self.gamma, self.tau, lr_actor, lr_critic, weight_decay, self.seed ) for x in range(self.num_agents)]
        
        # print(f"Actor model:\n{self.agents[0].actor_local}")
        # print(f"Critic model:\n{self.agents[0].critic_local}")
        
        self.memory=ReplayBuffer(action_size=action_size, buffer_size=buffer_size, batch_size=batch_size, seed=random_seed)
                    
        self.reset()
            
                
    def reset(self):
        """
        Resets the agents.
        """
        self.n_weight = self.n_start
        self.n_counter = 0
        for agent in self.agents:
            agent.reset
            
    
    def learn(self, experiences):
        """Performs batch learning for multiple agents"""
                
        actions =[]
        next_actions = []
        
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            actions.append(action)
            
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            next_actions.append(next_action)
            
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], actions, next_actions)
            

    def save_experience(self, states, actions, rewards, next_states, dones):
        """
        Save experience in replay memory, and use random sample from buffer to learn.

        Params
        ======
            states (array_like): states
            actions (array_like): actions
            rewards (array_like): rewards
            next_states: (array_like): next states
            done: indicates whether the episodes have ended
        """        
        # Save experience / reward
        states = states.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        self.memory.add(states, actions, rewards, next_states, dones)
              
        # Update the step counter for freezing the noise weighting factor
        self.n_counter += 1

        # Learn, if enough samples are available in memory and it's time to
        if self.n_counter % self.update_frequency == 0:
            if len(self.memory) > self.batch_size:
                experiences = [self.memory.sample() for _ in range(self.num_agents)]
                self.learn(experiences)


    def act(self, states):
        """
        Returns the values for the action array given a specific state. The values returned will be clipped by (-1, 1)

        Params
        ======
            states (array_like): state
            add_noise (boolean): specifies if noise will be added
        Returns
        =======              
        Actions for given state as per current policy.
        """
        actions=[]
        add_noise = True if self.n_counter < self.n_stop_point else False
        for state, agent in zip(states, self.agents):
            action=agent.act(state=state, add_noise=add_noise, n_weight=self.n_weight)
            self.n_weight = self.n_weight * self.n_decay
            actions.append(action)
        return np.array(actions).reshape(1, -1)
    
        
    def save_checkpoint(self, path, is_final):
        """
        Save the models for all the agents to disk as a checkpoint with the specified path and name prefix, automatically add the timestamp as part of the name to avoid overwritings 
            
        Params
        ======
            self: MADDPGmanager
            path (string): path to the files
        """
        path = os.path.join(path, "final") if is_final == True else os.path.join(path, "training")
        
        for agent in self.agents:
            agent.save_checkpoint(path)
            
            
    def load_checkpoint(self, path, timestamp):
        """ Load the models for all the agents from the specified file path and timestamp
                    
        Params
        ======
            self (dictionary): MADDPGmanager
            path (string): path to the file
            time_stamp (string): timestamp sufix
        """               
            
        for agent in self.agents:
            agent.load_checkpoint(path, timestamp)