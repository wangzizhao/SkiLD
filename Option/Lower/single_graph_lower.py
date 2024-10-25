from typing import Any, Optional, Union, List, Dict

import numpy as np
import gymnasium as gym
from omegaconf import DictConfig
from tianshou.data import Batch, to_numpy
from tianshou.policy import DQNPolicy, DDPGPolicy

from Causal import Dynamics
from State.extractor import Extractor
from Option.Terminate import RewardTermTruncManager
from Option.Lower.lower_policy import LowerPolicy
from Networks import DiaynDiscriminator


class SingleGraphLowerPolicy(LowerPolicy):
    """
    Wraps around a base policy (which handles the RL components) \
    to return the appropriate action space for the lower level \
    part of the hierarchy. This base model should generally return lists
    """
    def __init__(
        self,
        policies: List[Union[DQNPolicy, DDPGPolicy]],
        dynamics: Dynamics,
        diayn_discriminator: Union[DiaynDiscriminator, None],
        rewtermdone: RewardTermTruncManager,
        extractor: Extractor,
        action_space: gym.Space,
        config: DictConfig,
    ):
        """
        policies: Sequence[Union[DQNPolicy, DDPGPolicy]],
            a set of tianshou policies based on the algorithm of choice
        dynamics: Dynamics,
            dynamics model, used for estimating local dependency graph
        rewtermdone: RewardTermTruncManager,
            corresponding reward, termination and done objects to compute policy specific signals
        extractor: Extractor,
            handles factorization of state
        action_space: gym.Space,
            the environment action space, sometimes not stored in the TS policy so stored here
        """
        super().__init__(policies, dynamics, rewtermdone, extractor, action_space, config)

        self.diayn_discriminator = diayn_discriminator

        self.num_factors = num_factors = config.num_factors
        self.num_edge_classes = num_edge_classes = config.graph_encoding.num_edge_classes

        self.onehot_helper = np.eye(num_edge_classes).astype(np.float32)

        self.goal_type = config.policy.upper.goal_type
        self.graph_type = config.policy.upper.graph_type

        if self.graph_type == "graph":
            self.graph_info_size = num_factors * (num_factors + 1) * num_edge_classes
        elif self.graph_type == "factor":
            self.graph_info_size = num_factors + (num_factors + 1) * num_edge_classes
        elif self.graph_type == "none":
            self.graph_info_size = 0
        else:
            raise NotImplementedError(f"unknown graph type: {self.graph_type}")

        # a flag for HER to compute the achieved z for DIAYN
        self.compute_diayn_achieved_z = False

    def get_achieved_goal(self, data, use_next_obs, factor_choice=None):
        desired_graph = data.obs.desired_goal[..., :self.graph_info_size]
        bs = desired_graph.shape[:-1]
        num_factors = self.num_factors
        if self.graph_type == "graph":
            achieved_graph = self.onehot_helper[data.graph.astype(np.int64)]
            achieved_graph = achieved_graph.reshape(*bs, -1)
        elif self.graph_type == "factor":
            factor_onehot = desired_graph[..., :num_factors]                        # (bs, num_factors)
            if factor_choice is not None:
                factor_onehot = np.zeros_like(factor_onehot)
                factor_onehot[..., factor_choice] = 1
            factor_choice = np.nonzero(factor_onehot)
            achieved_parents = data.graph[factor_choice]                                # (bs, num_factors + 1)
            achieved_parents = achieved_parents.astype(np.int64)
            achieved_parents = self.onehot_helper[achieved_parents]                     # (bs, num_factors + 1, num_edge_classes)
            achieved_parents = achieved_parents.reshape(*bs, -1)
            achieved_graph = np.concatenate([factor_onehot, achieved_parents], axis=-1)
        elif self.graph_type == "none":
            achieved_graph = desired_graph
        else:
            raise NotImplementedError(f"unknown graph type: {self.graph_type}")

        desired_goal = data.obs.desired_goal[..., self.graph_info_size:]

        if self.goal_type == "value":
            assert self.graph_type == "factor"
            factored_state = data.get("next_target" if use_next_obs else "target",
                                      self.extractor.slice_targets(data, next=use_next_obs))        # (bs, num_factors, longest)
            achieved_goal = factored_state[factor_choice]                                           # (bs, longest)
        elif self.goal_type == "diayn":
            achieved_goal = desired_goal
            if self.compute_diayn_achieved_z:
                # only will enter this if condition when using HER to compute the achieved z
                state = data.get("target", self.extractor.slice_targets(data, next=False))          # (bs, num_factors, longest)
                next_state = data.get("target", self.extractor.slice_targets(data, next=False))     # (bs, num_factors, longest)
                factor_value = state[factor_choice]
                next_factor_value = next_state[factor_choice]
                achieved_goal = self.diayn_discriminator.get_achieved_z(desired_graph, factor_value, next_factor_value)
        else:
            raise NotImplementedError(f"unknown goal type: {self.goal_type}")

        return np.concatenate([achieved_graph, achieved_goal], axis=-1)

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any,
    ) -> Batch:
        state = state["lower"] if state is not None else None
        policy = self.policies[0](batch)
        policy.act = to_numpy(policy.act)
        policy.action_chain = np.expand_dims(policy.act, axis=1)  # batch.act is [batch_size, 1, environment action dim]
        policy.sampled_action_chain = policy.action_chain
        return policy

    def policy_obs(self, batch, policy_index, next=False):
        # no change to the batch, assumes that the correct data is stored
        return batch
