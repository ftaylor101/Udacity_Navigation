import torch
import torch.nn as nn

from collections import OrderedDict


class QNetwork(nn.Module):
    """
    Actor (Policy) Model.
    """

    def __init__(self, state_size: int, action_size: int, seed: int, layer_size: int = 64):
        """Initialize parameters and build model.

        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param seed: Random seed
        :param layer_size: Dimension of the hidden layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=state_size, out_features=layer_size)),
            ('reluf1', nn.ReLU()),
            ('fc2', nn.Linear(in_features=layer_size, out_features=layer_size)),
            ('reluf2', nn.ReLU()),
            ('output', nn.Linear(in_features=layer_size, out_features=action_size))
        ]))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Build a network that maps state -> action values.

        :return: tensor of action values
        """
        output = self.model(state)
        return output
