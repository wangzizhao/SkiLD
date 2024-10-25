from typing import Any, Dict, Optional, Sequence, Tuple, Union, Callable, Type

import numpy as np
import torch
from torch import nn
import sys, copy
from gymnasium.spaces import Discrete

from tianshou.utils.net.discrete import NoisyLinear
from tianshou.data import Batch
ModuleType = Type[nn.Module]


# necessary args for these networks:
    # net_type
    # embed_dim
# output dim
    # factor: query_dim, key_dim, first_obj_dim, single_obj_dim, query_aggregate
# factor net: reduce_function, num_pair_layers, append_keys

class GenDQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        input_dim: int,
        action_shape: Sequence[int],
        # hidden_sizes: Sequence[int],
        net_args: dict, # the factor network information
        device: Union[torch.device, None] = None,
        features_only: bool = False,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.device = device

        self.hidden_sizes = net_args.hidden_sizes
        self.encoder = None
        self.extractor = None
        sys.path.append(sys.path[0] + "/Causal/ac_infer")
        from Causal.ac_infer.Network.General.Factor.key_query import KeyQueryEncoder
        from Causal.ac_infer.Network.Dists.net_dist_utils import init_key_args, init_query_args
        from Causal.ac_infer.Network.net_types import network_type, FACTOR_NETS
        if net_args.net_type in FACTOR_NETS:
            self.extractor = net_args.extractor
            key_args, query_args = init_key_args(net_args), init_query_args(net_args)
            net_args = copy.deepcopy(net_args)
            self.encoder= KeyQueryEncoder(net_args, key_args, query_args)
            net_args.factor.key_dim, net_args.factor.query_dim = self.encoder.key_dim, self.encoder.query_dim
            net_args.input_dim = net_args.embed_dim
            net_args.object_dim = net_args.embed_dim
            net_args.output_dim = self.hidden_sizes[-1]
            self.goal_dim = 0 if net_args.factor.start_dim == 0 else net_args.factor.first_obj_dim
        self.num_outputs = self.hidden_sizes[-1]
        self.net = network_type[net_args.net_type](net_args) # make sure that net_args gives the right shape for the output (self.output_dim = hidden_sizes[-1])
        self.output_dim = self.hidden_sizes[-1]

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
        if self.extractor is not None: obs = self.extractor.slice_targets(obs, flatten = True, skip_num=self.goal_dim) 
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.get_default_dtype())
        if self.encoder is not None: 
            keys, queries = self.encoder(obs) # TODO: allow networks to share encoders
            return self.net(keys, queries, None, ret_settings=[])[0], state  # TODO: assumes no support of state
        return self.net(obs), state


class GenRainbow(GenDQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    an exact copy of Networks.network.Rainbow except for the the difference in import
    TODO: pretty sure python can do multi inheritance, which should be used here instead of replicating the code
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
        obs, state = super().forward(obs) # TODO: assumes no support of state
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
