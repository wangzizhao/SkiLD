from typing import Any, Dict, Optional, Sequence, Tuple, Union, Callable, Type

import numpy as np
import torch
from torch import nn
import time

from tianshou.utils.net.discrete import NoisyLinear
ModuleType = Type[nn.Module]


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        input_dim: int,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int],
        device: Union[torch.device, None] = None,
        features_only: bool = False,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.device = device

        sizes = [input_dim] + list(hidden_sizes)
        self.net = nn.Sequential(
            *sum([ (list() if norm_layer is None else [norm_layer(in_dim)]) +
                [layer_init(nn.Linear(in_dim, out_dim)),
                   activation(inplace=True) if activation == nn.ReLU else activation()] 
                   for in_dim, out_dim in zip(sizes[:-1], sizes[1:])],
                 start=list())
        )
        self.output_dim = sizes[-1]

        if not features_only:
            self.net = nn.Sequential(
                self.net, layer_init(nn.Linear(self.output_dim, np.prod(action_shape)))
            )
            self.output_dim = np.prod(action_shape)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.get_default_dtype())
        out, state = self.net(obs), state
        return out, state


class Rainbow(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        input_dim: int,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int],
        num_atoms: int = 51,
        noisy_std: float = 0.5,
        device: Union[torch.device, None] = None,
        is_dueling: bool = True,
        is_noisy: bool = True,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
    ) -> None:
        super().__init__(input_dim, action_shape, hidden_sizes, device, features_only=True, norm_layer=norm_layer, activation=activation)
        self.action_num = np.prod(action_shape)
        self.num_atoms = num_atoms

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            else:
                return nn.Linear(x, y)

        self.Q = nn.Sequential(
            linear(self.output_dim, 512), nn.ReLU(inplace=True),
            linear(512, self.action_num * self.num_atoms)
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(
                linear(self.output_dim, 512), nn.ReLU(inplace=True),
                linear(512, self.num_atoms)
            )
        self.output_dim = self.action_num * self.num_atoms

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        q = self.Q(obs)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2)
        return probs, state
