import numpy as np
import time

from typing import Any, Dict, Optional, Union, List
from tianshou.data import Batch, ReplayBuffer

from Option.Lower import LowerPolicy
from Causal import Dynamics
from Option.Terminate import RewardTermTruncManager
from State.extractor import Extractor
from State.buffer import VectorHierarchicalReplayBufferManager
from tianshou.policy import DQNPolicy, DDPGPolicy
from omegaconf import DictConfig
import gymnasium as gym
from tianshou.policy import DQNPolicy, DDPGPolicy, PPOPolicy
from Option.utils import get_new_indices


class WideLowerPolicy(LowerPolicy):
    """
    A lower policy consisting of one policy for each factor. \
    Each of the policies is contained in factored_lower_policy \
    However, sampling data is nontrivial (TODO)
    """

    def __init__(
        self,
        policies: List[Union[DQNPolicy, DDPGPolicy]],
        dynamics: Dynamics,
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
        TODO: possibly integrate with single_graph_lower, since certain features overlap
        """
        super().__init__(policies, dynamics, rewtermdone, extractor, action_space, config)
        self.num_factors = config.num_factors
        self.num_edge_classes = num_edge_classes = config.graph_encoding.num_edge_classes
        self.env_goal_based = config.goal_based
        self.graph_type = config.policy.upper.graph_type
        self.add_count_based_lower = config.policy.upper.add_count_based_lower

        self.onehot_helper = np.eye(num_edge_classes).astype(np.float32)
        self.graph_size = self.num_factors + (self.num_factors + 1) * self.num_edge_classes
        if self.graph_type == "graph":  # graph style goals with wide lower would require some creative redefinitions
            self.graph_info_size = self.num_factors * (self.num_factors + 1) * num_edge_classes
        elif self.graph_type == "factor":
            self.graph_info_size = self.num_factors + (self.num_factors + 1) * num_edge_classes
        else:
            raise NotImplementedError(f"unknown graph type: {self.graph_type}")

    def update(self, sample_size: int, buffer: Optional[VectorHierarchicalReplayBufferManager],
               **kwargs: Any) -> Dict[str, Any]:
        """ Logic is copied from tianshou.policy.base, but handles multiple policies
        Update the policy network and replay buffer.
        This needs a changed update policy to sample only from valid indices
        """
        results = {}
        for idx, policy in enumerate(self.policies):
            # start = time.time()
            result = policy.update(sample_size, buffer, **kwargs)  # implemented in subclass
            results.update({f"{idx}_{k}": v for k, v in result.items()})
            # print("per factor", time.time() - start)

        return results

    def reset_training(self, buffer):
        for policy in self.policies:
            policy.reset_training(buffer)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        # runs forward with every policy in lower
        # if lower is hierarchical, batch.time_lower and batch.
        state = state["lower"] if state is not None else None
        combined_output = list()
        self._obs = batch.obs
        for idx, policy in enumerate(self.policies):
            batch.obs = policy.construct_obs(batch)
            combined_output.append(policy(batch))
        batch.obs = self._obs
        policy = Batch.stack(combined_output, axis=1)
        policy.action_chain = policy.act
        policy.act = policy.act[np.arange(len(batch)), batch.option_choice]
        return policy

    def get_achieved_goal(self, data, use_next_obs):
        # return all_achieved_goals
        bs, desired_goal_dim = data.obs.desired_goal.shape[:-1], data.obs.desired_goal.shape[-1]
        achieved_goal = np.zeros(bs + (len(self.policies), desired_goal_dim))
        for idx, policy in enumerate(self.policies):
            achieved_goal[..., idx, :] = policy.get_achieved_goal(data, use_next_obs)
        achieved_goal = np.take_along_axis(achieved_goal, data.option_choice[..., None, None], axis=-2)[..., 0, :]
        return achieved_goal

    def update_lower_stats(self, lower_metrics, desired_goal):
        # assume wide_lower.rewtermdone == each factor_lower.rewtermdone, and it handles factor-specific / graph-specific updates
        self.rewtermdone.update_lower_stats(lower_metrics, desired_goal)

    def update_schedules(self):
        self.rewtermdone.update_schedules()
        # for policy in self.policies:
        #     policy.rewtermdone.update_schedules()

    def update_state_counts(self, data):
        self.rewtermdone.update_state_counts(data)
        # self.policies[0].rewtermdone.update_state_counts(data)
