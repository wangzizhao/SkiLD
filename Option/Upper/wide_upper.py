from typing import Any, Dict, Union, Optional

import torch
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

import numpy as np
from omegaconf import DictConfig
from gymnasium.spaces import Box, MultiDiscrete
from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.policy import BasePolicy
from tianshou.policy import DQNPolicy, DDPGPolicy

from Causal import Dynamics
from Networks import GraphEncoding
from State.extractor import Extractor
from State.buffer import HierarchicalReplayBuffer, VectorHierarchicalReplayBufferManager
from Option.Upper import UpperPolicy, ModularUpper
from Option.Terminate import RewardTermTruncManager
from Option.utils import get_new_indices


EPS = 1e-2
SOFTMAX_TEMP = 0.2
softmax = torch.nn.Softmax(dim=-1)

class WideUpper(ModularUpper):
    """
    Graph:
        - factor + parents or the whole graph
        - sampled from historical or learned
    Goal:
        - for a factor or for the whole state (all factors)
        - sampled from historical or learned
        - sampled for diayn
    """

    def __init__(
            self,
            policy: Union[BasePolicy, None],
            dynamics: Dynamics,
            rewtermdone: RewardTermTruncManager,
            extractor: Extractor,
            config: DictConfig,
    ):
        super().__init__(policy, dynamics, rewtermdone, extractor, config)
        self.unique_factor_graph_from_hash = [dict() for i in range(self.num_factors)]
        self.factor_achieved_tracker = [dict() for i in range(self.num_factors)]

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        _obs, batch.obs = batch.obs, batch.upper_obs    # use the upper obs instead of the obs
        if self.policy is None:
            policy = Batch(act=np.zeros((len(batch), 1)), state=None)
        else:
            policy = self.policy.forward(batch, state)
        policy = Batch(act=to_numpy(policy.act))
        batch.obs = _obs
        policy.act = to_numpy(policy.act)
        # if policy.state is None:
        #     policy.pop("state")
        # sample action is a sample of act
        # could use policy.logits instead of policy.act
        # policy.act of shape [batch, num_factors * action_dim + num_factors]
        # the first num_factors components correspond to the option_choice, if not random 
        policy = self.sample_action(policy)
        return policy

    def sample_act(self, batch: Batch):
        bs = len(batch)
        if self.policy is None:
            act = np.zeros((bs, 1))
        else:
            act = np.array([self.policy.action_space.sample() for _ in range(bs)])
        policy = Batch(act=act)
        return policy

    def sample_history_graph(self, bs):
        # almost the same logic as modular upper sample history graph, except for using the unique_FACTOR_graph
        # and iterating over factors. 
        # bs: tuple (batch, num_factors)
        assert len(bs) == 2
        all_sampled_graphs = list()
        for i in range(self.num_factors):
            all_sampled_graphs.append(self.sample_single_history_graph((bs[0], ), self.unique_factor_graph_from_hash[i], force_factor=i))
        return np.stack(all_sampled_graphs, axis=1)

    def sample_learned_goal(self, act, graph):
        # for single graph modules, uses the super version, otherwise, applies per factor
        if graph.ndim == 3: # assume that we are getting a factored rep [batch, factor, graph]
            learned_goals = list()
            for i in range(self.num_factors):
                learned_goals.append(super().sample_learned_goal(act[:,i], graph[:,i]))
            return np.stack(learned_goals, axis=1)
        else: # graph ndim = 2
            return super().sample_learned_goal(act, graph)

    def sample_history_goal(self, graph):
        # for single graph modules, uses the achieved goal tracker
        if graph.ndim == 3: # assume that we are getting a factored rep [batch, factor, graph]
            history_goals = list()
            for i in range(self.num_factors):
                history_goals.append(self.sample_history_goal_single(graph[:,i], self.factor_achieved_tracker))
            return np.stack(history_goals, axis=1)
        else: # graph ndim = 2
            return self.sample_history_goal_single(graph, self.achieved_goal_tracker)
    
    def sample_action(self, policy: Batch, random: bool = False):
        policy.option_choice_logits = policy.act[...,:self.num_factors] 
        factor_logits = torch.tensor(policy.option_choice_logits) if type(policy.option_choice_logits) == np.ndarray else policy.option_choice_logits
        policy.option_choice = torch.distributions.Categorical(softmax(factor_logits)).sample().cpu().numpy()
        desired_logits = torch.tensor(policy.act[...,self.num_factors:]) if type(policy.act) == np.ndarray else policy.act[...,self.num_factors:]
        policy.act = policy.all_desired_logits = desired_logits.reshape(policy.act.shape[0], self.num_factors, -1)
        hots = torch.zeros(len(policy.act), self.num_factors, self.num_factors)
        if type(policy.act) == torch.Tensor and policy.act.is_cuda: hots = hots.cuda()
        for i in range(self.num_factors):
            hots[:,i,i] = 1
        policy.act = torch.cat([hots, policy.act], dim=-1)
        policy = super().sample_action(policy, random) # results in policy.sampled_act having sampled ALL desired goals
        # selects the one act and one sampled act corresponding to option_choice
        policy.sampled_act, policy.all_desired_goals = policy.sampled_act[np.arange(len(policy.act)), policy.option_choice], policy.sampled_act
        policy.act = policy.act[np.arange(len(policy.act)), policy.option_choice]
        return policy