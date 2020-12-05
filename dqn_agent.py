import numpy as np
import random
from collections import namedtuple, deque

from models import QNetwork, Encoder

import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import mkdir


class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(config["seed"])
        self.seed = config["seed"]
        self.gamma = 0.99
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.tau = config["tau"]
        self.fc1 = config["fc1_units"]
        self.fc2 = config["fc2_units"]
        self.device = config["device"]
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, self.fc1, self.fc2, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.fc1, self.fc2, self.seed).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.encoder = Encoder(config).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), self.lr)


        # Replay memory

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, memory, writer):
        self.t_step += 1 
        if len(memory) > self.batch_size:
            if self.t_step % 4 == 0:
                experiences = memory.sample(self.batch_size)
                self.learn(experiences, writer)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        state = state.type(torch.float32).div_(255)
        self.qnetwork_local.eval()
        self.encoder.eval()
        with torch.no_grad():
            state = self.encoder.create_vector(state)
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        self.encoder.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, writer):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        states = states.type(torch.float32).div_(255)
        states = self.encoder.create_vector(states)
        next_states = next_states.type(torch.float32).div_(255)
        next_states = self.encoder.create_vector(next_states)
        actions = actions.type(torch.int64)
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * dones)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        writer.add_scalar('Q_loss', loss, self.t_step)
        # Minimize the loss
        self.optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
        self.encoder_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau)*target_param.data)
    
    def save(self, filename):
        """
        """
        mkdir("", filename)
        torch.save(self.qnetwork_local.state_dict(), filename + "_q_net.pth")
        torch.save(self.optimizer.state_dict(), filename + "_q_net_optimizer.pth")
        torch.save(self.encoder.state_dict(), filename + "_encoder.pth")
        torch.save(self.encoder_optimizer.state_dict(), filename + "_encoder_optimizer.pth")
        print("Save models to {}".format(filename))
