import os
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.beta import Beta

from tianshou.data import Batch, to_numpy

from Causal.dynamics import Dynamics

from Causal.modules import ChannelMHAttention
from Causal.dynamics_utils import reset_layer, forward_network, expand_helper, flatten_helper, mixup_helper


class DynamicsGrad(Dynamics):
    def __init__(self, env, extractor, config, is_upper=False):
        super(DynamicsGrad, self).__init__(env, extractor)

        self.config = config
        self.extractor = extractor
        self.is_upper = is_upper
        self.device = device = config.device
        self.grad_config = config.dynamics.grad
        self.use_prio = config.data.prio.dynamics_prio

        self.init_model(env)
        self.reset_params()

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.grad_config.lr)
        self.train()

        self.updating = False
        self.first_forward = False

        # for post-processing graph
        self.no_change_graph = torch.eye(self.num_factors, self.num_factors + 1, dtype=torch.bool, device=self.device)

        self.cmi_cache = None

    def init_model(self, env):
        spaces = list(env.dict_obs_space.spaces.values())
        if all([isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)) for space in spaces]):
            self.continuous_state = False
        elif all([isinstance(space, gym.spaces.Box) for space in spaces]):
            self.continuous_state = True
        else:
            raise NotImplementedError

        if self.is_upper:
            raise NotImplementedError
        else:
            self.action_space = env.action_space

        self.continuous_action = isinstance(self.action_space, gym.spaces.Box)

        self.local_causality_type = self.grad_config.local_causality_type

        # set up cmi all-but-one mask
        if self.local_causality_type == "cmi":
            self.cmi_mask = None

        # set up mix-up for discrete space training
        self.mixup = self.grad_config.mixup_alpha > 0
        if self.continuous_state or self.continuous_action:
            self.mixup = False

        if self.mixup:
            alpha = torch.tensor(float(self.grad_config.mixup_alpha), device=self.device)
            self.beta = Beta(alpha, alpha)

        # get number of input factor and the size
        self.num_factors = num_factors = self.config.num_factors
        self.factor_longest = self.extractor.longest

        if isinstance(self.action_space, gym.spaces.Box):
            self.action_size = action_size = self.action_space.shape[0]
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self.action_size = action_size = self.action_space.n
        else:
            raise NotImplementedError

        self.use_diayn = self.is_upper and self.config.policy.upper.goal_type == "diayn"
        if self.is_upper:
            num_inputs = num_factors + 2
            if self.use_diayn:
                self.num_diayn_classes = self.config.policy.upper.diayn.num_classes
                input_size = max(action_size, self.factor_longest, self.num_diayn_classes)
            else:
                input_size = max(action_size, self.factor_longest)
            raise NotImplementedError
        else:
            num_inputs = num_factors + 1
            input_size = max(action_size, self.factor_longest)

        # get output shape and mask
        self.init_output_status(env)

        # dynamics model networks
        self.saz_feature_weights = nn.ParameterList()
        self.saz_feature_biases = nn.ParameterList()
        self.predictor_weights = nn.ParameterList()
        self.predictor_biases = nn.ParameterList()

        channel_size = self.num_output_heads

        # Instantiate the parameters of each module in each variable's dynamics model
        # state action feature extractor
        in_dim = input_size
        for out_dim in self.grad_config.feature_fc_dims:
            self.saz_feature_weights.append(nn.Parameter(torch.zeros(channel_size, num_inputs, in_dim, out_dim)))
            self.saz_feature_biases.append(nn.Parameter(torch.zeros(channel_size, num_inputs, 1, out_dim)))
            in_dim = out_dim

        # multi-head attention
        feature_embed_dim = self.grad_config.feature_fc_dims[-1]

        attn_config = self.grad_config.attn
        attn_out_dim = attn_config.attn_out_dim
        post_fc_dims = attn_config.post_fc_dims
        if post_fc_dims:
            assert attn_out_dim == post_fc_dims[-1]

        self.attns = nn.ModuleList()
        for i in range(attn_config.num_attns):
            in_dim = attn_out_dim if i else feature_embed_dim
            attn = ChannelMHAttention((channel_size, ),
                                      attn_config.attn_dim,
                                      attn_config.num_heads,
                                      num_inputs, in_dim, num_inputs, in_dim,
                                      out_dim=attn_config.attn_out_dim,
                                      use_bias=attn_config.attn_use_bias,
                                      residual=attn_config.residual,
                                      share_weight_across_kqv=attn_config.share_weight_across_kqv,
                                      post_fc_dims=attn_config.post_fc_dims)
            self.attns.append(attn)

        # predictor
        in_dim = attn_out_dim
        for out_dim in self.grad_config.predictor_fc_dims:
            self.predictor_weights.append(nn.Parameter(torch.zeros(channel_size, in_dim, out_dim)))
            self.predictor_biases.append(nn.Parameter(torch.zeros(channel_size, 1, out_dim)))
            in_dim = out_dim

        self.predictor_weights.append(nn.Parameter(torch.zeros(channel_size, in_dim, self.output_head_longest)))
        self.predictor_biases.append(nn.Parameter(torch.zeros(channel_size, 1, self.output_head_longest)))

    def init_output_status(self, env):
        self.pred_granularity = self.grad_config.pred_granularity

        spaces = list(env.dict_obs_space.spaces.values())

        # get number of variables and the longest length
        self.num_variables = 0
        self.variable_longest = 0
        self.variable_idx_to_factor_idx = []
        for i, space in enumerate(spaces):
            if isinstance(space, gym.spaces.Discrete):
                num_variables_in_factor = 1
                self.variable_longest = max(self.variable_longest, space.n)
            elif isinstance(space, gym.spaces.MultiDiscrete):
                num_variables_in_factor = space.nvec.shape[0]
                self.variable_longest = max(self.variable_longest, *space.nvec)
            elif isinstance(space, gym.spaces.Box):
                num_variables_in_factor = space.shape[0]
                self.variable_longest = max(self.variable_longest, 1)
            else:
                raise NotImplementedError
            self.num_variables += num_variables_in_factor
            self.variable_idx_to_factor_idx += [i] * num_variables_in_factor

        # used for
        #   1. masking out the invalid entries in the variable matrix of (bs, num_variables, variable_longest)
        #   2. converting concatenated state (bs, state_size) to variable (bs, num_variables, variable_longest)
        self.variable_entry_mask = torch.zeros(self.num_variables, self.variable_longest, dtype=torch.bool, device=self.device)
        var_idx = 0
        for space in spaces:
            if isinstance(space, gym.spaces.Discrete):
                self.variable_entry_mask[var_idx, :space.n] = True
                var_idx += 1
            elif isinstance(space, gym.spaces.MultiDiscrete):
                for n in space.nvec:
                    self.variable_entry_mask[var_idx, :n] = True
                    var_idx += 1
            elif isinstance(space, gym.spaces.Box):
                assert len(space.shape) == 1
                n = space.shape[0]
                self.variable_entry_mask[var_idx:var_idx + n, 0] = True
                var_idx += n
            else:
                raise NotImplementedError

        # get number of output heads and their longest size
        # self.output_entry_mask:
        #   converting prediction output of (bs, num_output_heads, output_head_longest) to
        #   variable (bs, num_variables, variable_longest)
        if self.pred_granularity == "variable":
            self.num_output_heads = self.num_variables
            self.output_head_longest = self.variable_longest
            self.output_idx_to_factor_idx = self.variable_idx_to_factor_idx
            self.output_entry_mask = self.variable_entry_mask
        elif self.pred_granularity == "factor":
            self.num_output_heads = self.num_factors
            self.output_head_longest = self.factor_longest
            self.output_idx_to_factor_idx = list(range(self.num_factors))
            self.output_entry_mask = torch.tensor(self.extractor.obs_mask, dtype=torch.bool, device=self.device)
        elif self.pred_granularity == "macro_variable":
            assert hasattr(env, "macro_variable_space")

            # dict_obs_space: Dict[str (factor_name), gym.Space]
            dict_obs_space = env.dict_obs_space
            # macro_variable_space: Dict[str (factor_name), gym.spaces.Dict[str (macro-variable name), gym.Space]]
            macro_variable_space = env.macro_variable_space

            self.num_output_heads = self.output_head_longest = 0
            self.output_idx_to_factor_idx = []

            # pre-assign a too large mask, and then trim it to the correct size
            self.output_entry_mask = torch.zeros(self.num_variables, self.factor_longest, dtype=torch.bool, device=self.device)
            for factor_idx, obs_k in enumerate(dict_obs_space.keys()):
                factor_obs_space = dict_obs_space[obs_k]
                if isinstance(factor_obs_space, gym.spaces.Box):
                    assert len(factor_obs_space.shape) == 1
                    factor_size = factor_obs_space.shape[0]
                elif isinstance(factor_obs_space, gym.spaces.Discrete):
                    factor_size = factor_obs_space.n
                elif isinstance(factor_obs_space, gym.spaces.MultiDiscrete):
                    factor_size = np.sum(factor_obs_space.nvec)
                else:
                    raise NotImplementedError

                macro_variable_obs_space = macro_variable_space[obs_k]      # Dict[str (macro-variable name), gym.Space]

                macro_variable_total_size = 0
                for macro_variable_name, macro_variable_obs_space in macro_variable_obs_space.spaces.items():
                    if isinstance(macro_variable_obs_space, gym.spaces.Box):
                        assert isinstance(factor_obs_space, gym.spaces.Box)
                        assert len(factor_obs_space.shape) == 1
                        assert len(macro_variable_obs_space.shape) == 1
                        macro_variable_size = macro_variable_obs_space.shape[0]
                    elif isinstance(macro_variable_obs_space, gym.spaces.Discrete):
                        assert isinstance(factor_obs_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete))
                        macro_variable_size = macro_variable_obs_space.n
                    elif isinstance(macro_variable_obs_space, gym.spaces.MultiDiscrete):
                        assert isinstance(factor_obs_space, gym.spaces.MultiDiscrete)
                        macro_variable_size = np.sum(macro_variable_obs_space.nvec)
                    else:
                        raise NotImplementedError

                    macro_variable_total_size += macro_variable_size
                    self.output_head_longest = max(self.output_head_longest, macro_variable_size)
                    self.output_idx_to_factor_idx.append(factor_idx)
                    self.output_entry_mask[self.num_output_heads, :macro_variable_size] = True
                    self.num_output_heads += 1

                if macro_variable_total_size != factor_size:
                    raise ValueError(f"micro variable space {macro_variable_obs_space} does not match "
                                     f"factor space {factor_obs_space}")

            self.output_entry_mask = self.output_entry_mask[:self.num_output_heads, :self.output_head_longest]
        else:
            raise NotImplementedError

    def reset_params(self):
        module_weights = [self.saz_feature_weights, self.predictor_weights]
        module_biases = [self.saz_feature_biases, self.predictor_biases]
        for weights, biases in zip(module_weights, module_biases):
            for w, b in zip(weights, biases):
                reset_layer(w, b)

    @staticmethod
    def compute_annealing_val(step, starts=0, start_val=0, ends=0, end_val=0):
        coef = np.clip((step - starts) / (ends - starts), 0, 1)
        return start_val + coef * (end_val - start_val)

    def setup_annealing(self, step):
        pass

    def extract_state_action_feature(self, factors, action, diayn_z=None):
        """
        :param factors: [(bs, num_output_heads, factor_longest)] * num_factors
            notice that bs must be 1D
        :param action: (bs, num_output_heads, action_dim)
        :param diayn_z: (bs, num_output_heads, num_diayn_classes)
        :return: (num_output_heads, num_inputs, bs, out_dim),
        """
        inputs, self.sa_inputs = [], []
        for input_i in factors + [action]:
            input_i = input_i.detach()
            input_i.requires_grad = True
            self.sa_inputs.append(input_i)          # (bs, num_output_heads, input_i_dim)

            input_i = input_i.permute(2, 1, 0)      # (input_i_dim, num_output_heads, bs)
            inputs.append(input_i)

        if self.use_diayn:
            raise NotImplementedError

        # (input_size, num_inputs, num_output_heads, bs)
        x = pad_sequence(inputs)

        # (num_output_heads, num_inputs, bs, input_size)
        x = x.permute(2, 1, 3, 0)

        sa_feature = forward_network(x, self.saz_feature_weights, self.saz_feature_biases)
        sa_feature = F.relu(sa_feature)

        return sa_feature  # (num_output_heads, num_inputs, bs, feature_embed_dim)

    def preprocess_inputs(self, data):
        """
        :param data.obs: (bs, state_size), bs can be multi-dimensional
        :param data.action: (bs, ) or (bs, action_size)
        :param data.diayn_z: (bs, num_diayn_classes)
        :param data.obs_next: (bs, state_size)
        :return:
            state: (flat_bs, num_output_heads, state_size), notice that flat_bs is flattened to 1D
            action: (flat_bs, num_output_heads, action_size)
            diayn_z: (flat_bs, num_output_heads, num_diayn_classes)
        """
        state = data.obs.observation if type(data.obs) == Batch else data.obs
        factors = self.extractor.slice_targets(state)         # (bs, state_size) -> (bs, num_factors, factor_longest)

        next_state = data.obs_next.observation if type(data.obs_next) == Batch else data.obs_next
        if self.is_upper:
            raise NotImplementedError
        else:
            action = data.act

        state = torch.as_tensor(state, device=self.device, dtype=torch.get_default_dtype())
        factors = torch.as_tensor(factors, device=self.device, dtype=torch.get_default_dtype())
        action = torch.as_tensor(action, device=self.device, dtype=torch.get_default_dtype())
        next_state = torch.as_tensor(next_state, device=self.device, dtype=torch.get_default_dtype())

        # one-hot encoding for discrete action
        if not self.continuous_action and action.ndim == state.ndim - 1:
            action = F.one_hot(action.type(torch.int64), self.action_space.n).float()

        diayn_z = None
        if self.use_diayn:
            raise NotImplementedError

        input_ndim, input_shape = action.ndim, action.shape
        assert input_ndim == state.ndim

        # expand the input to (bs, num_output_heads, input_dim)
        factors = factors.unbind(dim=-2)
        self.expand_output_dim = not (input_ndim == 3 and input_shape[-2] == self.num_output_heads)
        if self.expand_output_dim:
            factors, action, diayn_z = expand_helper([factors, action, diayn_z], -2, self.num_output_heads)

        # compress multi-dimensional bs to 1-d bs
        bs = action.shape[:-2]
        # (bs, num_output_heads, input_dim) -> (flat_bs, num_output_heads, input_dim)
        factors, action, diayn_z = flatten_helper([factors, action, diayn_z], -2)

        # (bs, state_size) -> (flat_bs, state_size)
        state, next_state = flatten_helper([state, next_state], -1)

        if self.mixup and self.first_forward:
            flat_bs = action.shape[0]
            mix_weight = self.beta.sample((flat_bs, 1, 1))
            rand_idx = torch.randperm(flat_bs)
            factors, action, diayn_z = mixup_helper([factors, action, diayn_z], mix_weight, rand_idx)
            next_state = mixup_helper(next_state, mix_weight[..., 0], rand_idx)

        return bs, state, factors, action, diayn_z, next_state

    def predict(self, data):
        """
        :param state: (bs, state_size), bs can be multi-dimensional
        :param action: (bs, ) or (bs, action_size)
        :param diayn_z: (bs, ) or (bs, num_diayn_classes)
        """
        bs, state, factors, action, diayn_z, next_state = self.preprocess_inputs(data)

        flatten_bs = len(bs) > 1
        flat_bs = np.prod(bs)

        # (num_output_heads, num_inputs, bs, feature_embed_dim)
        sa_feature = self.extract_state_action_feature(factors, action, diayn_z)

        # set up mask for cmi training or evaluation
        if self.local_causality_type == "cmi":
            if self.updating:
                assert self.cmi_mask is None
                num_choices = 2 * (self.num_factors + 1)
                cmi_mask = torch.randint(0, num_choices, (self.num_output_heads, flat_bs),
                                         device=self.device, dtype=torch.int64)
                # (num_output_heads, flat_bs, num_inputs)
                cmi_mask = F.one_hot(cmi_mask, num_choices)[..., :self.num_factors + 1].float()
                cmi_mask = cmi_mask.permute(0, 2, 1)                                    # (num_output_heads, num_inputs, flat_bs)
            else:
                assert self.cmi_mask is not None
                cmi_mask = self.cmi_mask[None, :, None]                                 # (1, num_inputs, 1)
            sa_feature = sa_feature * (1 - cmi_mask[..., None])

        for attn in self.attns:
            # (num_output_heads, num_inputs, bs, attn_out_dim)
            sa_feature = attn(sa_feature, sa_feature)

        # (num_output_heads, bs, attn_out_dim)
        sa_feature = sa_feature[np.arange(self.num_output_heads), self.output_idx_to_factor_idx]

        # (num_output_heads, bs, output_head_longest)
        prediction = forward_network(sa_feature, self.predictor_weights, self.predictor_biases)

        # (bs, num_output_heads, output_head_longest)
        prediction = prediction.permute(1, 0, 2)

        # (bs, num_output_heads, output_head_longest) -> (bs, num_variables, variable_longest)
        prediction_variable = torch.zeros(flat_bs, self.num_variables, self.variable_longest,
                                          device=self.device, dtype=prediction.dtype)
        prediction_variable[..., self.variable_entry_mask] = prediction[..., self.output_entry_mask]

        if self.continuous_state:
            state_in_variable = torch.zeros(flat_bs, self.num_variables, self.variable_longest,
                                            device=self.device, dtype=prediction.dtype)
            state_in_variable[..., self.variable_entry_mask] = state
            prediction_variable = prediction_variable + state_in_variable

        if flatten_bs:
            prediction_variable = prediction_variable.reshape(*bs, *prediction_variable.shape[-2:])
            state = state.reshape(*bs, state.shape[-1])
            next_state = next_state.reshape(*bs, next_state.shape[-1])

        return prediction_variable, state, next_state

    def __call__(self, data):
        if self.local_causality_type == "gradient":
            # forward pass
            prediction, state, next_state = self.predict(data)
            # (bs, num_factors), (bs, num_factors), (bs, num_factors)
            _, _, grad_target, pred_correct, changed = self.loss(prediction, state, next_state)
            # backward pass
            graph = self.gradient_graph(grad_target)
            graph = self.post_process_graph(data, graph, pred_correct, changed)
        elif self.local_causality_type == "cmi":
            with torch.no_grad():
                neg_logps = []
                for i in range(self.num_factors + 1):
                    # (num_inputs, )
                    self.cmi_mask = F.one_hot(torch.tensor(i, device=self.device, dtype=torch.int64),
                                              self.num_factors + 1).float()
                    prediction, state, next_state = self.predict(data)
                    # (bs, num_variables)
                    _, _, neg_logp_i, _, _ = self.loss(prediction, state, next_state)
                    neg_logps.append(neg_logp_i)
                neg_logps = torch.stack(neg_logps, dim=-1)                              # (bs, num_variables, num_factors + 1)

                self.cmi_mask = torch.zeros(self.num_factors + 1, device=self.device, dtype=torch.float32)
                prediction, state, next_state = self.predict(data)
                # (bs, num_factors), (bs, num_factors)
                self.pred_loss_cache, self.priority_cache, neg_logp, pred_correct, changed = \
                    self.loss(prediction, state, next_state)

                cmi = (neg_logps - neg_logp[..., None]).clip(min=0)                     # (bs, num_variables, num_factors + 1)

                # aggregate variable cmi to factor cmi
                bs = cmi.shape[:-2]
                cmi_in_factor = torch.zeros(*bs, self.num_factors, self.num_factors + 1, device=self.device, dtype=cmi.dtype)
                for var_idx, factor_idx in enumerate(self.variable_idx_to_factor_idx):
                    cmi_in_factor[..., factor_idx, :] = torch.maximum(cmi_in_factor[..., factor_idx, :], cmi[..., var_idx, :])

                self.cmi_cache = cmi_in_factor

                graph = cmi_in_factor > self.grad_config.cmi_threshold
                graph = self.post_process_graph(data, graph, pred_correct, changed)
                self.cmi_mask = None
        else:
            raise NotImplementedError
        return graph

    def gradient_graph(self, target):
        # target: (bs, feature_dim)
        sa_grads = torch.autograd.grad(target.sum(), self.sa_inputs,
                                       retain_graph=self.updating,
                                       create_graph=self.updating and self.grad_config.gradient.grad_reg_coef > 0)

        self.grad_norm = 0
        grad_var_mask = torch.zeros(*target.shape[:-1], self.num_output_heads, self.num_factors + 1,
                                    device=self.device, dtype=torch.get_default_dtype())
        grad_factor_mask = torch.zeros(*target.shape[:-1], self.num_factors, self.num_factors + 1,
                                       device=self.device, dtype=torch.get_default_dtype())
        for i, (grad, sa_input) in enumerate(zip(sa_grads, self.sa_inputs)):
            # grad: (bs, num_output_heads, action_dim / obj_i_dim)
            grad = grad.abs()
            self.grad_norm += grad.sum(dim=(-2, -1)).mean()
            if i < self.num_factors and not self.continuous_state:
                # only use grad of 1s in the one-hot encodings
                grad = grad * sa_input
            elif i == self.num_factors and not self.continuous_action:
                # only use grad of 1s in the one-hot encodings
                grad = grad * sa_input

            grad_var_mask[..., i] = grad.sum(dim=-1)

        # (bs, num_output_heads, num_factors + 1) -> (bs, num_factors, num_factors + 1)
        for output_idx, factor_idx in enumerate(self.output_idx_to_factor_idx):
            grad_factor_mask[..., factor_idx, :] += grad_var_mask[..., output_idx, :]

        graph = grad_factor_mask > self.grad_config.gradient.local_causality_threshold

        return graph

    def post_process_graph(self, data, graph, pred_correct, changed):
        """
        param graph: (bs, num_factors, num_factors + 1)
        param pred_correct: (bs, num_factors)
        param changed: (bs, num_factors)
        """
        # overwrite sub-graphs with inpred_correct predictions to all False
        # TODO: this assumes there is no all-False sub-graphs in ground truth graphs
        if not self.updating and hasattr(data, "true_graph"):
            assert np.all(data.true_graph.any(axis=-1))

        graph[~pred_correct, :] = False
        graph = torch.where(changed[..., None], graph, self.no_change_graph)
        graph = to_numpy(graph)
        return graph

    def update(self, batch_size, buffer):
        self.updating = True
        self.first_forward = True

        batch, indices = buffer.sample(batch_size,
                                       policy_prio=False,
                                       dynamics_prio=self.use_prio)
        prediction, state, next_state = self.predict(batch)
        # scalar, (bs, num_variables), (bs, num_factors), (bs, num_factors), (bs, num_factors)
        pred_loss, priority, grad_target, pred_correct, changed = self.loss(prediction, state, next_state)

        self.first_forward = False
        loss = pred_loss
        loss_detail = {}

        if self.local_causality_type == "gradient":
            if self.mixup:
                # use non-mixup data for graph and priority
                prediction, state, next_state = self.predict(batch)
                # (bs, num_factors), (bs, num_factors), (bs, num_factors)
                pred_loss, priority, grad_target, pred_correct, changed = self.loss(prediction, state, next_state)

            graph = self.gradient_graph(grad_target)
            graph = self.post_process_graph(batch, graph, pred_correct, changed)

            if self.grad_config.gradient.grad_reg_coef:
                loss_detail["input_grad_norm"] = self.grad_norm.item()
                loss += self.grad_norm * self.grad_config.gradient.grad_reg_coef
        elif self.local_causality_type == "cmi":
            self.updating = False
            graph = self(batch)
            pred_loss, priority = self.pred_loss_cache, self.priority_cache
        else:
            raise NotImplementedError

        loss_detail["pred_loss"] = pred_loss.item()
        self.backprop(loss, loss_detail)

        buffer.update_graph(indices, graph)
        if self.use_prio:
            assert indices.ndim == 1
            assert priority.ndim == 2
            buffer.update_weight(indices, priority)

        self.updating = False

        if hasattr(batch, "true_graph"):
            true_graph = batch.true_graph
            if self.cmi_cache is not None and true_graph.any() and (~true_graph).any():
                cmi = to_numpy(self.cmi_cache)
                pos_cmi = cmi[true_graph][:, None]
                neg_cmi = cmi[~true_graph][:, None]
                threshold = np.linspace(cmi.min(), neg_cmi.max(), 100)
                true_positive = (pos_cmi > threshold).sum(axis=0)
                false_positive = (neg_cmi > threshold).sum(axis=0)
                false_negative = (pos_cmi <= threshold).sum(axis=0)
                f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
                best_threshold = threshold[np.argmax(f1)]
                loss_detail["cmi_threshold"] = best_threshold
                loss_detail["best_f1"] = f1.max()

            changed = to_numpy(changed)
            graph_of_changed_factor = graph[changed]
            true_graph_of_changed_factor = true_graph[changed]
            true_positive = (graph_of_changed_factor & true_graph_of_changed_factor).sum()
            false_positive = (graph_of_changed_factor & ~true_graph_of_changed_factor).sum()
            false_negative = (~graph_of_changed_factor & true_graph_of_changed_factor).sum()
            if true_positive + false_positive + false_negative:
                f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
                loss_detail["graph_f1"] = f1
            if true_positive + false_positive:
                loss_detail["graph_precision"] = true_positive / (true_positive + false_positive)
            if true_positive + false_negative:
                loss_detail["graph_recall"] = true_positive / (true_positive + false_negative)
            if graph_of_changed_factor.size:
                loss_detail["graph_acc"] = (graph_of_changed_factor == true_graph_of_changed_factor).mean()

        if hasattr(self, "pred_acc"):
            loss_detail["pred_acc"] = self.pred_acc

        loss_detail = {"dyn/" + k: v for k, v in loss_detail.items()}
        return loss_detail

    def loss(self, prediction, state, next_state):
        """
        param prediction: tensor, (bs, num_variables, variable_longest)
        param state: tensor, (bs, state_size)
        param next_state: tensor, (bs, state_size)
        """
        bs = state.shape[:-1]

        # convert flattened state and next_state to (bs, num_variables, variable_longest)
        current_variable = torch.zeros(*bs, self.num_variables, self.variable_longest, device=self.device, dtype=state.dtype)
        next_variable = torch.zeros(*bs, self.num_variables, self.variable_longest, device=self.device, dtype=next_state.dtype)
        current_variable[..., self.variable_entry_mask] = state                         # (bs, num_variables, variable_longest)
        next_variable[..., self.variable_entry_mask] = next_state                       # (bs, num_variables, variable_longest)

        changed_var = torch.any(current_variable != next_variable, dim=-1)              # (bs, num_variables)

        if self.continuous_state:
            assert self.variable_longest == 1 and torch.all(self.variable_entry_mask)
            error = (prediction - next_variable).sum(dim=-1).abs()                      # (bs, num_variables)
            pred_loss = error.sum(dim=-1).mean()
            priority = grad_target = error                                              # (bs, num_variables)
            pred_correct_var = error < self.grad_config.regression_correct_threshold
        else:
            prediction[..., ~self.variable_entry_mask] = -np.inf
            log_softmax = F.log_softmax(prediction, dim=-1)                             # (bs, num_variables, variable_longest)
            log_softmax = log_softmax.where(self.variable_entry_mask, 0)
            ce = -(next_variable * log_softmax).sum(dim=-1)                             # (bs, num_variables)

            pred_loss = ce.sum(dim=-1).mean()
            self.pred_acc = (torch.argmax(prediction, dim=-1) == next_variable.argmax(dim=-1)).float().mean().item()

            # priority = 1 - torch.exp(-ce)
            priority = ce
            grad_target = ce                                                            # (bs, num_variables)
            pred_correct_var = priority < self.grad_config.classification_correct_threshold

        # check whether each factor is predicted pred_correctly, used for filtering out the local causality graph
        # (bs, num_variables) -> (bs, num_factors)
        pred_correct_factor = torch.ones(*bs, self.num_factors, device=self.device, dtype=torch.bool)
        for var_idx, factor_idx in enumerate(self.variable_idx_to_factor_idx):
            pred_correct_factor[..., factor_idx] = pred_correct_factor[..., factor_idx] & pred_correct_var[..., var_idx]

        changed_factor = torch.zeros(*bs, self.num_factors, device=self.device, dtype=torch.bool)
        for var_idx, factor_idx in enumerate(self.variable_idx_to_factor_idx):
            changed_factor[..., factor_idx] = changed_factor[..., factor_idx] | changed_var[..., var_idx]

        return pred_loss, priority, grad_target, pred_correct_factor, changed_factor

    def backprop(self, loss, loss_detail):
        self.optimizer.zero_grad()
        loss.backward()

        grad_clip_norm = self.grad_config.grad_clip_norm
        if not grad_clip_norm:
            grad_clip_norm = np.inf
        loss_detail["grad_norm"] = grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)

        if torch.isfinite(loss) and torch.isfinite(grad_norm):
            self.optimizer.step()

        return loss_detail

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("DynamicsGradAttn loaded from", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
