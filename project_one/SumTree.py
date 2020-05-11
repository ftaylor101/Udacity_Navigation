import numpy as np
from typing import Tuple


class SumTree:
    """
    This SumTree code has been obtained and modified from:
    https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb
    """
    data_pointer = 0  # used to keep calculate the leaf node at which to store priorities and where to store
    # experience data

    def __init__(self, capacity: int):
        """
        Constructor of SumTree of a specified capacity.
        All nodes in the tree are set to 0. Initialize the data with all values = 0

        :param capacity: number of priorities to store
        """
        self.capacity = capacity  # Number of leaf nodes that contains priorities
        self.n_entries = 0  # current number of priorities stored

        self.tree = np.zeros(2 * capacity - 1)  # array to represent tree and store priority values

        self.data = np.zeros(capacity, dtype=object)  # array to store experience data

    def add(self, priority: float, data: Tuple):
        """
        Add the experience data and its priority to the SumTree.

        :param priority: priority value
        :param data: experience data
        """
        tree_index = self.data_pointer + self.capacity - 1  # find next tree index for the experience

        self.data[self.data_pointer] = data  # store experience data

        self.update(tree_index, priority)  # update the leaf node with the priority

        self.data_pointer += 1  # keep track of next empty index in the tree for storing

        if self.data_pointer >= self.capacity:  # Overwrite data once capacity has been reached
            self.data_pointer = 0

        if self.n_entries < self.capacity:  # keep track of current number of priorities stored
            self.n_entries += 1

    def update(self, tree_index: int, priority: float):
        """
        Update the leaf priority score and propagate the change through the SumTree.

        :param tree_index: the index of the leaf node in the SumTree at which to update the priority
        :param priority: the new priority value
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:  # propagate the change through the SumTree
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value: float) -> Tuple[float, float, Tuple]:
        """
        Here we get the leaf_index, priority value of that leaf and experience associated with that index

        :param value: a random number within [0, total_priority]
        :return: the leaf node index, its priority value and the experience data for that leaf node
        """
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):  # If we reach the bottom, end the search
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        """
        Contains the value of the root node, the total priority.

        :return: sum of all priorities in the SumTree
        """
        return self.tree[0]
