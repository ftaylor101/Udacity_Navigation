import numpy as np
import torch

from typing import Tuple
from SumTree import SumTree


class Memory:
    """
    A class to add and access experience data and its associated priority.

    This code has been obtained and modified from:
    https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb

    Hyperparameters:
    - a
        Introduces some randomess to the sampling. a = 0 is equivalent to uniform sampling
    - b
        To control how much is learnt from the associated experience data. b anneals from initial value to 1, meaning
        the learning impact of the higher priority samples is reduced.
    - b_increment
        amount to anneal the b value by, moving towards 1 every sampling event
    """
    e = 0.1  # a constant to avoid some experiences having 0 probability and therefore never being sampled
    a = 0.0
    b = 0.0
    b_increment = 0.00000

    absolute_error_upper = 1.0  # priority to use when current max priority is 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, capacity: int):
        """
        Constructor to instantiate the SumTree.

        :param capacity: maximum number of leaf nodes in the SumTree
        """
        self.tree = SumTree(capacity=capacity)

    def add(self, experience: Tuple):
        """
        Store a new experience in the SumTree
        Each new experience has a priority equal to the maximum priority currently in the SumTree

        :param experience:
            experience data received from the agent and its interaction with the environment. Tuple of
            (state, action, reward, next_state, done)
        """
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])  # Find the current max priority

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(priority=max_priority, data=experience)

    def sample(self, n: int) -> Tuple[Tuple, np.ndarray, torch.Tensor]:
        """
        Sampling method to return n experience samples.

        :param n: the number of samples to be returned
        :return:
            tuple containing the experience data, the leaf node indices for the data and the data's importance weights
        """
        experience_sample = []  # Create a sample array that will contains the minibatch
        leaf_idx = np.empty((n,), dtype=np.int32)
        importance_weights = np.empty((n, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority / n

        self.b = np.min([1.0, self.b + self.b_increment])  # anneal b hyperparameter

        # Calculating the max_weight which comes from the smallest priority value
        if self.tree.n_entries < self.tree.capacity:
            min_priority = np.min(self.tree.tree[-self.tree.capacity:-(self.tree.capacity - self.tree.n_entries)]) / self.tree.total_priority
        else:
            min_priority = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (min_priority * n) ** (-self.b)

        for i in range(n):
            index = 0
            data = 0
            priority = 0.0
            while priority == 0.0:
                a, b = priority_segment * i, priority_segment * (i + 1)
                value = np.random.uniform(a, b)

                index, priority, data = self.tree.get_leaf(value=value)

            sampling_probabilities = priority / self.tree.total_priority

            importance_weights[i, 0] = np.power(n * sampling_probabilities, -self.b) / max_weight

            leaf_idx[i] = index
            experience = [data]
            experience_sample.append(experience)

        states_mb = torch.from_numpy(np.vstack([each[0][0] for each in experience_sample])).float().to(self.device)
        actions_mb = torch.from_numpy(np.vstack([each[0][1] for each in experience_sample])).long().to(self.device)
        rewards_mb = torch.from_numpy(np.vstack([each[0][2] for each in experience_sample])).float().to(self.device)
        next_states_mb = torch.from_numpy(np.vstack([each[0][3] for each in experience_sample])).float().to(self.device)
        dones_mb = torch.from_numpy(np.vstack([each[0][4] for each in experience_sample]).astype(np.uint8)).float().\
            to(self.device)
        imp_wgts = torch.from_numpy(np.array([ew[0] for ew in importance_weights])).float().to(self.device)

        return (states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb), leaf_idx, imp_wgts

    def batch_update(self, tree_idx: np.ndarray, abs_errors: np.ndarray):
        """
        Update the priorities in the SumTree for the recently evaluated batch of experiences. Clips the errors to have
        a maximum error of +1.

        :param tree_idx: indices of the SumTree to update
        :param abs_errors: the new error values that turn into priorities
        """
        abs_errors += self.e  # add constant priority to avoid 0 priority
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
