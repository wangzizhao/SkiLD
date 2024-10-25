from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)

import sys, copy
import numpy as np
import torch
from torch import nn

from tianshou.data.batch import Batch

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
                 Sequence[Dict[Any, Any]]]


class GenNet(nn.Module):
    """Wrapper of network for tianshou.
        copied from tianshou Net with modifications

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param state_shape: int or a sequence of int of the shape of state.
    :param action_shape: int or a sequence of int of the shape of action.
    :param hidden_sizes: shape of MLP passed in as a list.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: specify the device when the network actually runs. Default
        to "cpu".
    :param bool softmax: whether to apply a softmax layer over the last layer's
        output.
    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape only.
    :param int num_atoms: in order to expand to the net of distributional RL.
        Default to 1 (not use).
    :param bool dueling_param: whether to use dueling network to calculate Q
        values (for Dueling DQN). If you want to use dueling option, you should
        pass a tuple of two dict (first for Q and second for V) stating
        self-defined arguments as stated in
        class:`~tianshou.utils.net.common.MLP`. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.MLP` for more
        detailed explanation on the usage of activation, norm_layer, etc.

        You can also refer to :class:`~tianshou.utils.net.continuous.Actor`,
        :class:`~tianshou.utils.net.continuous.Critic`, etc, to see how it's
        suggested be used.
    """

    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        net_args: Dict,
        action_shape: Union[int, Sequence[int]] = 0,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        
        if not output_dim:
            output_dim = net_args.hidden_sizes[-1]
            net_args.hidden_sizes = net_args.hidden_sizes[:-1]

        net_args.num_inputs, net_args.num_outputs = input_dim, output_dim
        self.flatten_input = True # TODO: maybe this shouldn't always be true
        self.encoder = None
        self.extractor = None
        sys.path.append(sys.path[0] + "./Causal/ac_infer")
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
            net_args.output_dim = output_dim
            net_args.output_dim = output_dim
            self.goal_dim = 0 if net_args.factor.start_dim == 0 else net_args.factor.first_obj_dim
        self.model = network_type[net_args.net_type](net_args)
        self.output_dim = self.model.num_outputs
        if self.use_dueling:  # dueling DQN TODO: not sure the dictionaries play nice together
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms

            q_args = copy.deepcopy(net_args)
            for k in dueling_param[0].keys():
                q_args[k] = dueling_param[0][k]
            q_args["input_dim"] = self.output_dim,
            q_args["output_dim"] = q_output_dim,
            q_args["device"] = self.device

            v_args = copy.deepcopy(net_args)
            for k in dueling_param[1].keys():
                v_args[k] = dueling_param[1][k]
            v_args["input_dim"] = self.output_dim,
            v_args["output_dim"] = v_output_dim,
            v_args["device"] = self.device

            self.Q, self.V = network_type[net_args.net_type](q_args), network_type[net_args.net_type](v_args)
            self.output_dim = self.Q.output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        if self.extractor is not None: obs = self.extractor.slice_targets(obs, flatten = True, skip_num=self.goal_dim)
        if self.model.device is not None:
            obs = torch.as_tensor(obs, device=self.model.device, dtype=torch.float32)
        if self.flatten_input:
            obs = obs.flatten(1)
        if self.encoder is not None:
            keys, queries = self.encoder(obs) # TODO: allow networks to share encoders
            model_vals= self.model(keys, queries, None, ret_settings=[])
            logits = model_vals[0]
        else:
            logits = self.model(obs)
        bsz = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state
