from typing import Union, List, Optional, Any, Dict
import torch

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
from tianshou.data import Batch
from tianshou.policy import DQNPolicy, DDPGPolicy, PPOPolicy

from Causal import Dynamics
from Networks.diayn_discriminators import DiaynDiscriminator
from Option.Lower import SingleGraphLowerPolicy
from Option.Terminate import RewardTermTruncManager
from Option.utils import get_new_indices
from State.extractor import Extractor
from State.buffer import VectorHierarchicalReplayBufferManager


class FactorLowerPolicy(SingleGraphLowerPolicy):
    """
    The lower policy for a single factor. Rewrites the factor component of the goal with the 
    """

    def __init__(self,
                 policies: List[Union[DQNPolicy, DDPGPolicy]],
                 dynamics: Dynamics,
                 diayn_discriminator: Union[DiaynDiscriminator, None],
                 rewtermdone: RewardTermTruncManager,
                 extractor: Extractor,
                 action_space: gym.Space,
                 config: DictConfig,
                 idx: int,
                 ):
        assert len(policies) == 1
        super().__init__(policies, dynamics, diayn_discriminator, rewtermdone, extractor, action_space, config)
        assert self.graph_type == "factor", "FactorLowerPolicy can only be used with factor graphs"

        self.idx = idx
        self.is_state_coverage_policy = False
        if self.idx == self.num_factors:
            assert config.policy.upper.add_count_based_lower
            self.is_state_coverage_policy = True
            self.her_update_achieved_goal_fn = lambda x: x

        # a flag for HER to compute the achieved z for DIAYN
        self.compute_diayn_achieved_z = False

    def get_achieved_goal(self, data, use_next_obs):
        # for the state coverage factored lower, self.idx == self.num_factors, so we need to reset it to avoid out of index error
        # achieved goal is not used for state coverage, so it doesn't matter what we return
        idx = None if self.is_state_coverage_policy else self.idx
        return super().get_achieved_goal(data, use_next_obs, idx)

    def update(self, sample_size: int, buffer: Optional[VectorHierarchicalReplayBufferManager], **kwargs: Any) -> Dict[str, Any]:
        """ Logic is copied from tianshou.policy.base, but handles multiple policies
        Update the policy network and replay buffer.

        It includes 3 function steps: process_fn, learn, and post_process_fn. In
        addition, this function will change the value of ``self.updating``: it will be
        False before this function and will be True when executing :meth:`update`.
        Please refer to :ref:`policy_state` for more detailed explanation.

        :param int sample_size: 0 means it will extract all the data from the buffer,
            otherwise it will sample a batch with given sample_size.
        :param ReplayBuffer buffer: the corresponding replay buffer.

        :return: A dict, including the data needed to be logged (e.g., loss) from
            ``policy.learn()``.
        """
        if buffer is None:
            return {}

        policy, idx = self.policies[0], 0
        if isinstance(policy, PPOPolicy):
            self.buffer_last_index, indices = get_new_indices(self.buffer_last_index, buffer)
            if len(indices) == 0:
                return {}
            # filter out the samples that are not from the current policy
            mask = buffer.option_choice[indices] == self.idx
            if np.sum(mask) <= 1:
                return {}
            indices = indices[mask]
            batch = buffer[indices]

            kwargs = {"batch_size": sample_size, "repeat": self.config.policy.lower.ppo.repeat_per_collect}
            n_step = 1
        else:
            # state coverage factored lower uses self.idx == self.num_factors, and it doesn't need HER since it doesn't use goals
            # to turn off HER for this policy update, we set her_update_achieved_goal_fn to None
            if len(buffer.valid_graph_indices[self.idx]) <= 1:
                return {}

            her_update_achieved_goal_fn = None
            if self.use_her and not self.is_state_coverage_policy:
                her_update_achieved_goal_fn = self.her_update_achieved_goal_fn
                self.compute_diayn_achieved_z = self.her_use_diayn_posterior

            batch, indices = buffer.sample(
                sample_size,
                policy_prio=self.use_prio,
                dynamics_prio=False,
                her_update_achieved_goal=her_update_achieved_goal_fn,
                policy_idx=self.idx
            )

            n_step = self.n_step
            self.compute_diayn_achieved_z = False

        policy.updating = True
        self.rewtermdone.set_updating(True)

        # converts sampled batch.obs, .act, .obs_next, rew, returns to policy space
        batch = self.process_fn(batch, buffer, indices, idx, n_step)
        temp_obs, temp_next, batch.obs, batch.obs_next = batch.obs, batch.obs_next, self.preprocess(batch.obs), self.preprocess(batch.obs_next)
        batch.obs = torch.as_tensor(batch.obs, device=self.config.device, dtype=torch.get_default_dtype())
        batch.obs_next = torch.as_tensor(batch.obs_next, device=self.config.device, dtype=torch.get_default_dtype())
        result = policy.learn(batch, **kwargs)              # implemented in contained class: ex: tianshou policy
        batch.obs, batch.obs_next = temp_obs, temp_next
        self.post_process_fn(batch, buffer, indices, idx)
        if policy.lr_scheduler is not None:
            policy.lr_scheduler.step()

        policy.updating = False
        self.rewtermdone.set_updating(False)
        return result

    def construct_obs(self, batch):
        upper_type = self.config.policy.upper.type
        if upper_type == "modular":
            return batch.obs
        elif upper_type == "wide":
            # TODO: goals are assigned from batch.obs.all_desired_goals
            # for single_lower there is no distinction between obs.desired_goal and desired_target_goal
            # (only if there are multiple options)
            obs = Batch(desired_goal=batch.obs.all_desired_goals[:, self.idx],
                        observation=batch.obs.observation,
                        all_desired_goals=batch.obs.all_desired_goals)
            if "graph" in batch:
                # the achieved goal is only used for hindsight related values, so it isn't necessary for most contexts
                obs.achieved_goal = np.concatenate([batch.graph[:, self.idx], batch.target[:, self.idx]],
                                                   axis=-1),  # TODO: we may want to replace this with obs-dependent valeus only
            return obs
        else:
            raise NotImplementedError
