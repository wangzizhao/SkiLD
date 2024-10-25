from typing import Any, Dict, Optional, Union

import numpy as np
from omegaconf import DictConfig

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.policy import DQNPolicy, DDPGPolicy, PPOPolicy
from tianshou.data import Batch, to_numpy

from Causal import Dynamics
from Option.Terminate import RewardTermTruncManager
from Option.utils import get_new_indices
from State.buffer import HierarchicalReplayBuffer, VectorHierarchicalReplayBufferManager


class UpperPolicy(nn.Module):
    """
    Wraps around a base policy (which handles the RL components) \
    to return the appropriate action space for the upper level \
    policy
    """

    def __init__(
        self,
        policy: Union[DQNPolicy, DDPGPolicy, None],
        dynamics: Dynamics,
        rewtermdone: RewardTermTruncManager,
        config: DictConfig,
    ):
        """
        policy: a tianshou policy based on the desired algorithm
        rewtermdone: reward terminate and done signals, usually based on the true signal for the upper policy
        lower_check_termination: termination for the lower determines if the upper needs a new sample
        config: algorithm specific arguments
        """
        super().__init__()
        self.policy = policy
        self.rewtermdone = rewtermdone
        self.dynamics = dynamics
        self.config = config

        self.n_step = config.policy.upper.n_step        # TD error steps, for off-policy algorithms
        if isinstance(self.policy, PPOPolicy):
            self.n_step = 1

        self.use_prio = config.data.prio.upper_prio
        self.rew_aggregation = config.option.upper.rew_aggregation
        self.lower_timeout = config.option.timeout

        self.her_update_achieved_goal_fn = None
        if config.data.her.upper.use_her:
            self.her_update_achieved_goal_fn = lambda batch: batch
        self.upper_buffer_last_index = 0                # to keep track of indices for on-policy updates

    def reset_training(self, buffer):
        self.upper_buffer_last_index = buffer._index

    def sample_act(self, batch: Batch):
        bs = len(batch)
        if self.policy is None:
            act = np.zeros((bs, 1))
        else:
            act = np.array([self.policy.action_space.sample() for _ in range(bs)])
        policy = Batch(act=act)
        return policy

    def sample_action(self, policy: Batch, random: bool = False):
        raise NotImplementedError

    def process_fn(self, batch, buffer, lower_buffer, indices):
        if self.policy is None:
            return batch

        upper_rew_config = self.config.option.upper

        assert indices.ndim == 1
        num_indices = len(indices)
        upper_indices = np.empty((self.n_step, num_indices), dtype=int)
        upper_indices[0] = indices
        for i in range(1, self.n_step):
            indices = buffer.next(indices)
            upper_indices[i] = indices

        upper_indices_flat = upper_indices.flatten()
        start, end = buffer.lower_buffer_start[upper_indices_flat], buffer.lower_buffer_end[upper_indices_flat]
        # (n_step * num_indices,), (n_step * num_indices, lower_timeout)
        lower_indices, indices_valid_mask = lower_buffer.get_lower_buffer_indices(start, end)
        indices_valid_mask = indices_valid_mask.reshape((self.n_step, num_indices, self.lower_timeout))

        # compute each lower step's reward
        lower_batch = lower_buffer[lower_indices]
        # lower_batch.graph = self.dynamics(lower_batch)
        lower_rew_flat = self.rewtermdone.check_rew(lower_batch)
        lower_rew = np.zeros((self.n_step, num_indices, self.lower_timeout))
        lower_rew[indices_valid_mask] = lower_rew_flat

        # aggregate the lower step's reward
        if self.rew_aggregation == "sum":
            upper_rew = np.sum(lower_rew, axis=-1)
        elif self.rew_aggregation == "max":
            assert np.all(lower_rew_flat >= 0), "empty reward is set to 0 and can be larger than actual reward"
            upper_rew = np.max(lower_rew, axis=-1)
        else:
            raise NotImplementedError

        upper_rew = upper_rew_config.graph_novelty_scale * upper_rew
        if upper_rew_config.use_graph_reachability:
            upper_rew *= buffer.lower_reached_graph[upper_indices]

        # overwrite batch and buffer rew
        buffer.rew[upper_indices] = upper_rew
        batch.rew = upper_rew[0]

        return self.policy.process_fn(batch, buffer, indices)

    def post_process_fn(self, batch, buffer, indices):
        if self.policy is not None:
            self.policy.post_process_fn(batch, buffer, indices)

    def update_history(self, lower_buffer: VectorHierarchicalReplayBufferManager):
        pass

    def update(self, sample_size: int,
               buffer: HierarchicalReplayBuffer,
               lower_buffer: VectorHierarchicalReplayBufferManager,
               **kwargs) -> Dict[str, Any]:
        if buffer is None or len(buffer) == 0 or self.policy is None:
            return {}

        if isinstance(self.policy, PPOPolicy):
            self.upper_buffer_last_index, indices = get_new_indices(self.upper_buffer_last_index, buffer)
            if len(indices) == 0:
                return {}
            batch = buffer[indices]
            repeat = self.config.policy.upper.ppo.repeat_per_collect
            kwargs.update({"batch_size": sample_size, "repeat": repeat})
        else:
            # separate sample for all functions to support HER
            batch, indices = buffer.sample(sample_size,
                                           policy_prio=self.use_prio,
                                           dynamics_prio=False,
                                           her_update_achieved_goal=self.her_update_achieved_goal_fn)

        self.policy.updating = True
        self.rewtermdone.set_updating(True)

        if kwargs.get("mask", None) is not None:
            batch.obs = Batch(observation=batch.obs, mask=torch.tile(kwargs["mask"], (len(batch.obs), 1)))

        batch = self.process_fn(batch, buffer, lower_buffer, indices)
        result = self.policy.learn(batch, **kwargs)         # implemented in contained class: ex: tianshou policy

        if kwargs.get("mask", None) is not None:
            batch.obs = batch.obs.observation

        self.post_process_fn(batch, buffer, indices)        # should require no further change

        if self.policy.lr_scheduler is not None:
            self.policy.lr_scheduler.step()

        self.policy.updating = False
        self.rewtermdone.set_updating(False)
        return result

    def update_lower_stats(self, lower_metrics, desired_goal):
        pass

    @staticmethod
    def needs_upper(batch):
        # returns true if we need a new sample of the upper policy

        # "needs_resample": flag for the beginning of the data collection, initialized in collector.reset_env()
        if "needs_resample" in batch and batch.needs_resample.any():
            batch.needs_resample = np.zeros_like(batch.needs_resample, dtype=bool)
            return np.ones(len(batch), dtype=bool)

        # returns true when the highest lower terminates or the env is done
        # batch.term_chain is computed at the previous timestep, i.e., using obs_prev and obs
        return batch.term_chain.lower | batch.trunc_chain.lower | batch.done

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        _obs, batch.obs = batch.obs, batch.upper_obs    # use the upper obs instead of the obs
        if self.policy is None:
            # fake some dummy upper act
            policy = Batch(act=np.zeros((len(batch), 1)), state=None)
        else:
            if kwargs.get("mask", None) is not None:
                batch.obs = Batch(observation=batch.obs, mask=kwargs["mask"])

            policy = self.policy.forward(batch, state)

        batch.obs = _obs
        policy = Batch(act=to_numpy(policy.act))
        # sample action is a sample of act
        policy = self.sample_action(policy)
        return policy

    def logging(self, writer: SummaryWriter, step: int) -> bool:
        return False

    def update_schedules(self):
        pass

    def update_state_counts(self, data):
        pass