from typing import Any, Dict, Optional, Union, List
import time

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.policy import DQNPolicy, DDPGPolicy, PPOPolicy

from Causal import Dynamics
from Option.Terminate import RewardTermTruncManager
from Option.utils import get_new_indices
from State.extractor import Extractor
from State.buffer import VectorHierarchicalReplayBufferManager


class LowerPolicy(nn.Module):
    """
    Wraps around a base policy (which handles the RL components) \
    to return the appropriate action space for the lower level \
    part of the hierarchy. This base model should generally return lists
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
        """
        super().__init__()
        if len(policies):
            self.policies = nn.ModuleList(policies)     # policy.process_fn should handle
        else:
            self.policies = policies
        self.dynamics = dynamics
        self.rewtermdone = rewtermdone
        self.extractor = extractor
        self.action_space = action_space
        self.config = config

        self.n_step = config.policy.lower.n_step
        self.use_prio = config.data.prio.lower_prio
        self.use_her = config.data.her.lower.use_her
        self.her_use_diayn_posterior = config.data.her.lower.her_use_diayn_posterior
        self.her_update_achieved_goal_fn = lambda batch: batch
        if self.use_her:
            self.her_update_achieved_goal_fn = lambda batch: self.update_achieved_goal(batch)

        # For extracting on-policy indices
        self.buffer_last_index = None

    def reset_training(self, buffer):
        self.buffer_last_index = buffer.last_index

    def sample_action(self, option_choice):
        # samples the action chain with the option(s) chosen
        # TODO: this is incompatible with option-capable lower policies (lower policies calling other lower policies as options)
        # TODO: incompatible with resampled lower policies (action space != sampled action space)
        all_samples = list()
        option_samples = list()
        for oc in option_choice:
            samples = np.stack([self.action_space.sample() for _ in range(len(self.policies))], axis=0)
            option_samples.append(samples[oc])
            all_samples.append(samples)
        return np.stack(all_samples, axis=0), np.stack(option_samples, axis=0)

    def get_target(self, data, next=False):
        return self.extractor.slice_targets(data, next=next)

    def get_achieved_goal(self, data, use_next_obs):
        # get the achieved upper action goal for HER,
        # ideally only use data.obs.observation, data.obs.desired_goal, data.graph
        raise NotImplementedError

    def policy_obs(self, batch, policy_index, next=False):
        """
        converts batch.obs or batch.obs_next to the\
        sub policy space @param policy index.
        returns the adjusted batch. should be overriden in subclass
        """
        return self.policies[policy_index].policy_obs(batch, next=next)

    def update_achieved_goal(self, batch):
        # batch.graph = self.dynamics(batch)
        achieved_goal = self.get_achieved_goal(batch, use_next_obs=True)
        batch.obs.achieved_goal = batch.obs_next.achieved_goal = achieved_goal
        return batch

    def process_fn(self, batch, buffer, indices, idx, n_step):
        # TODO: if there is OOM error, we need to change this to sample n_step indices in a for loop
        assert indices.ndim == 1
        num_indices = len(indices)
        n_step_indices = np.empty(num_indices * n_step, dtype=indices.dtype)
        n_step_indices[:num_indices] = indices
        next_indices = indices
        for i in range(1, n_step):
            next_indices = buffer.next(next_indices)
            n_step_indices[num_indices * i:num_indices * (i + 1)] = next_indices

        n_step_batch = buffer[n_step_indices]
        n_step_batch = self.update_achieved_goal(n_step_batch)
        # achieved_goal and graph is not used anywhere except for RTT, so we don't need to write it back to the buffer
        # TODO: double-check if this is true
        # buffer.graph[n_step_indices] = n_step_batch.graph
        # buffer.obs.achieved_goal[n_step_indices] = n_step_batch.obs.achieved_goal
        # buffer.obs_next.achieved_goal[n_step_indices] = n_step_batch.obs.achieved_goal
        buffer.rew[n_step_indices] = self.rewtermdone.check_rew(n_step_batch, idx)

        batch = buffer[indices]
        return self.policies[idx].process_fn(batch, buffer, indices)

    def post_process_fn(self, batch, buffer, indices, idx):
        self.policies[idx].post_process_fn(batch, buffer, indices)

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

        result = {}
        for idx, policy in enumerate(self.policies):
            if isinstance(policy, PPOPolicy):
                self.buffer_last_index, indices = get_new_indices(self.buffer_last_index, buffer)
                if len(indices) == 0:
                    continue
                batch = buffer[indices]
                kwargs = {"batch_size": sample_size, "repeat": self.config.policy.lower.ppo.repeat_per_collect}
                n_step = 1
            else:
                # separate sample for all functions to support HER
                batch, indices = buffer.sample(sample_size,
                                               policy_prio=self.use_prio,
                                               dynamics_prio=False,
                                               her_update_achieved_goal=self.her_update_achieved_goal_fn)
                n_step = self.n_step
            policy.updating = True
            self.rewtermdone.set_updating(True)

            # converts sampled batch.obs, .act, .obs_next, rew, returns to policy space
            batch = self.process_fn(batch, buffer, indices, idx, n_step)
            result = policy.learn(batch, **kwargs)              # implemented in contained class: ex: tianshou policy
            self.post_process_fn(batch, buffer, indices, idx)
            if policy.lr_scheduler is not None:
                policy.lr_scheduler.step()

            policy.updating = False
            self.rewtermdone.set_updating(False)
        return result

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any,
    ) -> Batch:
        # runs forward with every policy in lower
        # if lower is hierarchical, batch.time_lower and batch.
        state = state["lower"] if state is not None else None
        policy = self.policies[0](batch)
        policy.act = to_numpy(policy.act)
        policy.action_chain = np.expand_dims(policy.act, axis=1)  # batch.act is [batch_size, 1, environment action dim]
        policy.sampled_action_chain = policy.action_chain
        return policy

    def logging(self, writer: SummaryWriter, step: int) -> bool:
        return False

    def update_lower_stats(self, lower_metrics, desired_goal):
        pass    # only used by wide lower for now, but probably could be used by single graph lower

    def update_schedules(self):
        pass    # updates the adaptive reward schedules, if necessary (see wide lower for usage)

    def update_state_counts(self, data):
        pass
