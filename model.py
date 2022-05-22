import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DuelingQNetwork(nn.Module):
    """Network mapping state to action"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hidden_units: List[int] = [64, 32, 16, 8],
    ):
        """Initialize parameters and build model.

        https://arxiv.org/abs/1511.06581

        Parameters
        ----------
        state_size : int
            Dimension of each state
        action_size : int
            Dimension of each action
        seed : int
            Random seed
        hidden_units : List[int], optional
            Number of nodes in each hidden layer, by default [64, 32, 16, 8]

        """
        super().__init__()
        self.seed = torch.manual_seed(seed)

        # input and hidden
        units = [state_size, *hidden_units]
        layer_list = []
        for i, u in enumerate(units):
            if i != 0:  # skip first
                layer_list.append(nn.Linear(units[i - 1], u))
        self.hidden_layers = nn.ModuleList(layer_list)

        # output
        self.advantage_hidden = nn.Linear(units[-1], units[-1])
        self.output_advantage_values = nn.Linear(units[-1], action_size)

        # output state values scalar
        self.state_values_hidden = nn.Linear(units[-1], units[-1])
        self.output_state_values = nn.Linear(units[-1], 1)

    def forward(self, state):
        """Model inference"""
        x = state
        for layer in self.hidden_layers:
            x = F.elu(layer(x))

        # action value estimation
        av = F.elu(self.advantage_hidden(x))
        av = self.output_advantage_values(av)

        # state value estimation
        sv = F.elu(self.state_values_hidden(x))
        sv = self.output_state_values(sv)

        # originally formulated as a max, then converted to
        # mean for more stability. a softmax was also attempted
        # but yielded similar results to mean + was more complex
        out = sv + (av - av.mean())

        return out
