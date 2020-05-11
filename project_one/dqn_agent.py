import numpy as np
import random
import torch
import torch.nn.functional as F

from torch import optim
from typing import Tuple, Optional

from project_one import QNetwork
from prioritized_memory import Memory


class Agent:
    """
    Interacts with and learns from the environment.
    Learns using a Deep Q-Network with prioritised experience replay.
    Two models are instantiated, one for use during evaluation and updating (qnetwork_local) and one to be used for the
    target values in the learning algorithm (qnetwork_target)
    """

    BUFFER_SIZE = int(1e5)  # prioritised experience replay buffer size
    BATCH_SIZE = 64  # minibatch size
    TAU = 1e-3  # for soft update of target parameters
    LR = 5e-4  # learning rate
    UPDATE_EVERY = 4  # how often to update the network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, state_size: int = 37, action_size: int = 4, seed: int = 44, gamma: float = 0.99,
                 tau: float = 1e-3):
        """
        Initialize an Agent object.

        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param seed: random seed for network initialisation
        :param gamma: discount factor
        :param tau: lag for soft update of target network parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau

        self.max_w = 0

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        # Prioritised Experience Replay memory
        self.memory = Memory(self.BUFFER_SIZE)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool,
             gamma: Optional[float] = None, tau: Optional[float] = None):
        """
        An agent step takes the current experience and stores it in the replay memory, then samples from the memory and
        calls the learning algorithm.

        :param state: the state vector
        :param action: the action performed on the state
        :param reward: the reward given upon performing the action
        :param next_state: the next state after doing the action
        :param done: True if the episode has ended
        :param gamma: discount factor
        :param tau: lag for soft update of target network parameters
        """
        gamma_value = gamma if gamma is not None else self.gamma
        tau_value = tau if tau is not None else self.tau

        self.memory.add((state, action, reward, next_state, done))  # Save experience in replay memory

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.tree.n_entries > self.BATCH_SIZE:
                experiences, idxs, importance_weights = self.memory.sample(self.BATCH_SIZE)
                self.learn(experiences, idxs, importance_weights, gamma_value, tau_value)

    def act(self, state: np.ndarray, eps: float = 0.0):
        """
        Returns actions for given state as per current policy. Uses the local copy of the model.

        :param state: current state
        :param eps: epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.int32(np.argmax(action_values.cpu().data.numpy()))
        else:
            return np.int32(random.choice(np.arange(self.action_size)))

    def learn(self, experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
              indices: np.ndarray, importance_weights: torch.Tensor, gamma: float, tau: float):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences: tuple of (s, a, r, s', done) tuples
        :param indices:
            indices of the SumTree that contain the priority values for these experiences. Used for updating the
            priority values after error has been found
        :param importance_weights: the weighting that each experience carries when used in updating the network
        :param gamma: discount factor
        :param tau: lag for soft update of target network parameters
        """
        states, actions, rewards, next_states, dones = experiences

        # For Double-DQN, get action with the highest q-value (for next_states) from the local model
        next_action = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.qnetwork_target(next_states).gather(1, next_action)
        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        error = torch.abs(q_targets - q_expected).detach().numpy()

        # update priorities
        self.memory.batch_update(indices, error)

        # Compute mse and loss with importance weights
        t_mse = F.mse_loss(q_expected, q_targets)
        loss = (importance_weights * t_mse).mean()
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network with model parameters approaching those of the local network.
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

    @staticmethod
    def soft_update(local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float):
        """
        Soft update model parameters. Every learning step the target network is updated to bring its parameters nearer
        by a factor TAU to those of the improving local network.

        If TAU = 1 the target network becomes a copy of the local network.
        If TAU = 0 the target network is not updated.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: weights will be copied from
        :param target_model: weights will be copied to
        :param tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
