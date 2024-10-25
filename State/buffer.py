from typing import Any, List, Tuple, Union, Optional, Callable

import scipy
import torch
import itertools
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from numba import njit
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import (Batch, SegmentTree, to_numpy,
                           ReplayBuffer, PrioritizedReplayBuffer, HERReplayBuffer,
                           ReplayBufferManager, PrioritizedReplayBufferManager)

from State.extractor import Extractor
from State.utils import Running

UPPER_EXTRA_SAMPLE_KEYS = ()
UPPER_EXTRA_RESERVED_KEYS = ("sample_ready", "lower_buffer_start", "lower_buffer_end", "lower_reached_graph")

# rew is rewritten as reward_chain.lower
# LOWER_EXTRA_KEYS = ("time_upper", "time_lower", "option_choice", "option_resample",
#                     "action_chain", "sampled_action_chain",
#                     "reward_chain", "term_chain", "trunc_chain",
#                     "target", "next_target",
#                     "target_diff", "proximity",
#                     "graph", "true_graph")
LOWER_EXTRA_SAMPLE_KEYS = ("true_graph", "true_graph_count_idx", "graph", "graph_count_idx", "env_rew")
LOWER_EXTRA_RESERVED_KEYS = ("option_choice", "upper_buffer_index")


class HierarchicalReplayBuffer(PrioritizedReplayBuffer, HERReplayBuffer):
    """
    Uses the Tianshou implementation of PrioritizedReplayBuffer with additional keys. Also allows for no-prio
    sampling, weighted sampling, and HER sampling, copying the code from Tianshou HER

    obs, obs_next contain the flattened full state from the environment, act is the primitive action
    time is the current step prior to termination for upper and lower
    option choice is the factor to control (as a binary vector),
    option_resample indicates when a goal is resampled for the lower level policie(s)
    action chain is the actions for all levels of the hierarchy, in order, separated upper and lower
    term and reward chain correspond to terminate and reward signals in order, separated upper and lower
    target, next_target, target_diff are factorized dictionaries
    """

    def __init__(
            self,
            is_upper: bool,
            size: int,
            use_her: bool = False,
            horizon: int = 1,
            future_k: float = 8.0,
            alpha: float = 0.6,
            beta: float = 0.4,
            decay_window: int = 5,
            decay_rate: float = 0.4,
            max_prev_decay: float = 0.7,
            weight_norm: bool = True,
            **kwargs: Any,
    ) -> None:
        self.use_her = use_her  # if flag is not raised, don't use HER
        PrioritizedReplayBuffer.__init__(self, size, alpha, beta, **kwargs)

        if is_upper:
            self._extra_sample_keys = UPPER_EXTRA_SAMPLE_KEYS
            extra_keys = UPPER_EXTRA_SAMPLE_KEYS + UPPER_EXTRA_RESERVED_KEYS
        else:
            self._extra_sample_keys = LOWER_EXTRA_SAMPLE_KEYS
            extra_keys = LOWER_EXTRA_SAMPLE_KEYS + LOWER_EXTRA_RESERVED_KEYS
        self._input_keys = self._input_keys + extra_keys
        self._reserved_keys = self._reserved_keys + extra_keys

        # HER init
        self.horizon = horizon
        self.future_p = 1 - 1 / future_k
        self.weight_norm = weight_norm
        self._original_meta = Batch()
        self._altered_indices = np.array([])

        # PSER init
        self.decay_window = decay_window
        self.decay_rate = decay_rate
        self.max_prev_decay = max_prev_decay

        self.policy_use_pser = self.decay_window > 0

        self.init_weight_tree(["dynamics"])     # policy PER tree is initialized in reset_policy_PER()
        self.reset_policy_PER()

    def reset_policy_PER(self) -> None:
        if self.policy_use_pser:
            self.upper_pser_stats = np.zeros(self.maxsize, dtype=np.float32)
        self.init_weight_tree(["policy"])

    def init_weight_tree(self, tree_names: List[str]) -> None:
        # PER for policy and dynamics update
        if not hasattr(self, "trees"):
            self.trees = {}

        for name in tree_names:
            self.trees[name] = [SegmentTree(self.maxsize), 1.0, 1.0]    # [tree, max_prio, min_prio]

        if not hasattr(self, "prio_cache"):
            self.prio_cache = Batch(used_per=False, tree_name=None)

    def init_weight(self, index: Union[int, np.ndarray]) -> None:
        if self.decay_window:
            if hasattr(self, "upper_pser_stats"):
                self.upper_pser_stats[index] = 0

        for name in self.trees:
            # trees[name][0] is the tree
            # trees[name][1] is the max_prio
            self.trees[name][0][index] = self.trees[name][1] ** self._alpha

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Batch:
        """Return a data batch: self[index].

        If stack_num is larger than 1, return the stacked obs and obs_next with shape
        (batch, len, ...).
        """
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indices = self.sample_indices(0) if index == slice(None) \
                else self._indices[:len(self)][index]
        else:
            indices = index  # type: ignore
        batch = ReplayBuffer.__getitem__(self, indices)
        if hasattr(self, "weight"):
            # PrioritizedReplayBufferManager will delete the weight of each buffer
            weight = self.get_weight(indices)
            # ref: https://github.com/Kaixhin/Rainbow/blob/master/memory.py L154
            batch.weight = weight / np.max(weight) if self._weight_norm else weight
        for k in self._extra_sample_keys:
            batch.__dict__[k] = self._meta[k][indices]
        return batch

    def sample(
            self,
            batch_size: int,
            policy_prio: bool = False,
            dynamics_prio: bool = False,
            her_update_achieved_goal: Callable = None,
    ) -> Tuple[Batch, np.ndarray]:
        """
        Replace Tianshou Sample to add no-prio, weights her reward parameters
        """
        assert not (policy_prio and dynamics_prio)
        assert not self.prio_cache.used_per, "must call update_weight() after each sample() call to clear the cache"

        self.prio_cache = Batch(used_per=policy_prio or dynamics_prio)
        if policy_prio or dynamics_prio:
            tree_name = "policy" if policy_prio else "dynamics"
            self.weight, self._max_prio, self._min_prio = self.trees[tree_name]
            self.prio_cache.tree_name = tree_name

        indices = self.sample_indices(batch_size,
                                      use_prio=policy_prio or dynamics_prio,
                                      her_update_achieved_goal=her_update_achieved_goal,)
        return self[indices], indices

    def update_weight(
        self, index: np.ndarray, new_weight: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Update priority weight by index in this buffer.

        :param np.ndarray index: index you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        if self.prio_cache.used_per:
            if self.prio_cache.tree_name == "dynamics":
                # update weight for dynamics training
                pass
            else:
                # update weight for policy training
                if self.policy_use_pser:
                    n_steps = self.decay_window + 1
                    n_step_indices = np.empty(n_steps * len(index), dtype=index.dtype)
                    n_step_indices[0::n_steps] = index
                    for i in range(1, n_steps):
                        index = self.prev(index)
                        n_step_indices[i::n_steps] = index

                    # new td_error_t ← max{|new_weight|, self.max_prev_decay * old_td_error_t}
                    # for i in range(1, n_steps):
                    #   new td_error_{t - i} ← max{|new_weight| * self.decay_rate ** i, old_td_error_{t - i}}
                    n_step_old_td_error = self.upper_pser_stats[n_step_indices]
                    n_step_old_td_error[0::n_steps] = self.max_prev_decay * n_step_old_td_error[0::n_steps]

                    td_error = np.abs(to_numpy(new_weight))
                    n_step_new_td_error = -np.ones_like(n_step_old_td_error)
                    n_step_new_td_error[0::n_steps] = td_error
                    for i in range(1, n_steps):
                        n_step_new_td_error[i::n_steps] = td_error * self.decay_rate ** i

                    n_step_new_td_error = np.maximum(n_step_new_td_error, n_step_old_td_error)

                    # filter out repeated indices in n_step_indices
                    # TODO: do not consider repeated indices in index
                    unique_mask = np.concatenate([[True], n_step_indices[1:] != n_step_indices[:-1]])
                    index = n_step_indices[unique_mask]
                    new_weight = self.upper_pser_stats[index] = n_step_new_td_error[unique_mask]

            super().update_weight(index, new_weight)
            tree_name = self.prio_cache.tree_name
            self.trees[tree_name] = [self.weight, self._max_prio, self._min_prio]

        self.prio_cache = Batch(used_per=False, tree_name=None)

    def sample_indices(
            self,
            batch_size: int,
            use_prio: bool = False,
            her_update_achieved_goal: Callable = None,
    ) -> np.ndarray:
        """Get a random sample of index with size = batch_size, with @param weights weighting the selection
        see Tianshou.data.base, since most of the code is copied from there

        Return all available indices in the buffer if batch_size is 0; return an empty
        numpy array if batch_size < 0 or no available index can be sampled.

        """
        self._restore_cache()
        if use_prio:
            indices = PrioritizedReplayBuffer.sample_indices(self, batch_size)    # sample with priority
        else:
            if "sample_ready" in self._meta and batch_size > 0:
                indices = np.random.choice(np.where(self.sample_ready)[0], batch_size)
            else:
                indices = ReplayBuffer.sample_indices(self, batch_size)
        if self.use_her and her_update_achieved_goal is not None:
            self.rewrite_transitions(indices.copy(), her_update_achieved_goal)

        return indices

    # Below is copied exactly from Tianshou HER

    def rewrite_transitions(
            self,
            indices: np.ndarray,
            her_update_achieved_goal: Callable,
    ) -> None:
        """
        Re-write the goal of some sampled transitions' episodes according to HER.

        Currently applies only HER's 'future' strategy. The new goals will be written \
        directly to the internal batch data temporarily and will be restored right \
        before the next sampling or when using some of the buffer's method (e.g. \
        `add`, `save_hdf5`, etc.). This is to make sure that n-step returns \
        calculation etc., performs correctly without additional alteration.

        policy_idx & graph_count are used to select graph that are less visited
        """
        if indices.size == 0:
            return

        # Sort indices keeping chronological order
        indices[indices < self._index] += self.maxsize
        indices = np.sort(indices)
        indices[indices >= self.maxsize] -= self.maxsize

        # Construct episode trajectories
        current = indices
        indices = np.empty((self.horizon, indices.size), dtype=indices.dtype)
        indices[0] = current
        for i in range(1, self.horizon):
            indices[i] = self.next(indices[i - 1])

        # Calculate future timestep to use
        terminal = indices[-1]
        episodes_len = (terminal - current + self.maxsize) % self.maxsize
        future_offset = np.random.uniform(size=len(indices[0])) * episodes_len
        future_offset = np.round(future_offset).astype(int)
        future_t = (current + future_offset) % self.maxsize

        # Compute indices
        #   open indices are used to find longest, unique trajectories among
        #   presented episodes
        unique_ep_open_indices = np.sort(np.unique(terminal, return_index=True)[1])
        unique_ep_indices = indices[:, unique_ep_open_indices]
        #   close indices are used to find max future_t among presented episodes
        unique_ep_close_indices = np.hstack(
            [(unique_ep_open_indices - 1)[1:],
             len(terminal) - 1]
        )

        # --------------------------------------- modification starts --------------------------------------- #
        if not self._save_obs_next:
            raise NotImplementedError

        # episode indices that will be altered
        her_ep_mask = np.random.rand(len(unique_ep_open_indices)) < self.future_p

        # update the achieved goal of future_obs
        her_ep_indices = unique_ep_close_indices[her_ep_mask]   # (num_her_episodes, )
        her_future_t = future_t[her_ep_indices]
        if her_future_t.size == 0:
            return
        future_batch = her_update_achieved_goal(self[her_future_t])
        future_achieved_goal = future_batch.obs_next.achieved_goal

        # TODO: if termination is also state dependent, then it should get reassigned here
        #   episode indices that will be altered
        # TODO: we don't re-assign rew here, assume it will be handled by upper_policy/lower_policy.process_fn()
        # Cache original meta
        self._altered_indices = unique_ep_indices[:, her_ep_mask]
        self._original_meta = self._meta.obs.desired_goal[self._altered_indices]
        # Re-assign goals via broadcast assignment
        self._meta.obs.desired_goal[self._altered_indices] = future_achieved_goal[None]
        if self._save_obs_next:
            self._meta.obs_next.desired_goal[self._altered_indices] = future_achieved_goal[None]

    def deactivate(self, indices: np.ndarray) -> None:
        if len(self) == 0:
            return

        indices = indices[self.sample_ready[indices]]
        if indices.size == 0:
            return

        self.sample_ready[indices] = False
        new_weight = np.zeros_like(indices)
        for name in self.trees:
            self.trees[name][0][indices] = new_weight

    def _restore_cache(self) -> None:
        """Write cached original meta back to `self._meta`.

        It's called everytime before 'writing', 'sampling' or 'saving' the buffer.
        """
        if not hasattr(self, '_altered_indices'):
            return

        if self._altered_indices.size == 0:
            return

        self._meta.obs.desired_goal[self._altered_indices] = self._original_meta
        self._meta.obs_next.desired_goal[self._altered_indices] = self._original_meta
        # Clean
        self._original_meta = Batch()
        self._altered_indices = np.array([])

    def reset(self, keep_statistics: bool = False) -> None:
        self._restore_cache()
        return super().reset(keep_statistics)

    def save_hdf5(self, path: str, compression: Optional[str] = None) -> None:
        self._restore_cache()
        return super().save_hdf5(path, compression)

    def set_batch(self, batch: Batch) -> None:
        self._restore_cache()
        return super().set_batch(batch)

    def update(self, buffer: ReplayBuffer) -> np.ndarray:
        self._restore_cache()
        return super().update(buffer)

    def add(
            self,
            batch: Batch,
            buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._restore_cache()
        return super().add(batch, buffer_ids)


class VectorHierarchicalReplayBufferManager(PrioritizedReplayBufferManager):
    """VectorHierarchicalReplayBuffer contains n HierarchicalReplayBuffer with the same size.

    It is used for storing transition from different environments yet keeping the order
    of time.

    :param int total_size: the total size of VectorReplayBuffer.
    :param int buffer_num: the number of ReplayBuffer it uses, which are under the same
        configuration.

    Other input arguments (stack_num/ignore_obs_next/save_only_last_obs/sample_avail)
    are the same as :class:`~tianshou.data.ReplayBuffer`.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """
    def __init__(
            self,
            env,
            total_size: int,
            buffer_num: int,
            extractor: Extractor = None,
            num_lower_policies: int = 1,
            lower_timeout: int = 1,
            count_threshold_for_valid_graph: int = 0,
            # HER parameters
            use_her: bool = False,
            her_use_count_select_goal: bool = False,
            horizon: int = 1,
            future_k: float = 8.0,
            lower_reached_graph_threshold: int = 1,
            # PER parameters
            policy_per_alpha: float = 0.6,
            dynamics_per_alpha: float = 0.6,
            beta: float = 0.4,
            weight_norm: bool = True,
            dynamics_per_pred_error_scale: float = 1.0,
            dynamics_per_change_count_scale: float = 0.0,
            policy_per_td_error_scale: float = 1.0,
            policy_per_graph_count_scale: float = 1.0,
            policy_per_graph_count_power: float = -0.5,
            # PSER parameters
            decay_window: int = 5,
            decay_rate: float = 0.4,
            max_prev_decay: float = 0.7,
            **kwargs: Any,
    ) -> None:
        assert buffer_num > 0
        self.buffer_num = buffer_num
        self.extractor = extractor
        self.total_size = total_size
        self.buf_size = int(np.ceil(total_size / buffer_num))
        self.horizon = horizon
        self.future_p = 1 - 1 / future_k
        self.lower_reached_graph_threshold = lower_reached_graph_threshold
        buffer_list = [HierarchicalReplayBuffer(is_upper=False,
                                                size=self.buf_size,
                                                use_her=use_her,
                                                horizon=horizon,
                                                future_k=future_k,
                                                alpha=0.5,                  # alpha is not used, so use a random value here
                                                beta=beta,
                                                weight_norm=weight_norm,
                                                **kwargs,)
                       for _ in range(buffer_num)]

        self.lower_timeout = lower_timeout

        super().__init__(buffer_list)
        self.__eps = 1e-30

        self.use_her = use_her
        self.her_use_count_select_goal = her_use_count_select_goal
        self.num_lower_policies = num_lower_policies

        self._extra_sample_keys = LOWER_EXTRA_SAMPLE_KEYS
        extra_keys = LOWER_EXTRA_SAMPLE_KEYS + LOWER_EXTRA_RESERVED_KEYS
        self._input_keys = self._input_keys + extra_keys
        self._reserved_keys = self._reserved_keys + extra_keys

        self.policy_per_alpha = policy_per_alpha
        self.policy_per_td_error_scale = policy_per_td_error_scale
        self.policy_per_graph_count_scale = policy_per_graph_count_scale
        self.policy_per_graph_count_power = policy_per_graph_count_power

        self.decay_window = decay_window
        self.decay_rate = decay_rate
        self.max_prev_decay = max_prev_decay

        self.dynamics_per_alpha = dynamics_per_alpha
        self.dynamics_per_pred_error_scale = dynamics_per_pred_error_scale
        self.dynamics_per_change_count_scale = dynamics_per_change_count_scale

        self.count_threshold_for_valid_graph = count_threshold_for_valid_graph

        self.dynamics_per_use_change_count = self.dynamics_per_change_count_scale > 0

        self.policy_use_pser = self.decay_window > 0
        self.policy_per_use_graph_count = self.policy_per_graph_count_scale > 0

        # if her_use_count_select_goal, use this to keep track of graph count
        self.num_factors = env.num_factors
        self.count_idx_to_graph = np.flip(list(itertools.product([0, 1], repeat=self.num_factors + 1)), axis=-1).astype(bool)
        self.reset_graph_count()
        self.reset_change_count(env)
        self.reset_dynamics_PER()
        self.reset_policy_PER()
        self.init_HER_utils()
        self.init_logging_stats()

    def reset_dynamics_PER(self) -> None:
        HierarchicalReplayBuffer.init_weight_tree(self, ["dynamics"])

        if self.dynamics_per_use_change_count:
            maxlen = 20000
            self.pred_error_stats = Running(maxlen)

    def reset_policy_PER(self) -> None:
        if self.policy_per_use_graph_count:
            maxlen = 20000
            self.td_error_stats = [Running(maxlen) for _ in range(self.num_lower_policies)]

            assert np.all(self.valid_graph_count[:, 0] == 0), "no parent graph should not exist"

        # policy PSER weight
        if self.policy_use_pser:
            self.pser_stats = [np.zeros(self.total_size, dtype=np.float32)
                               for _ in range(self.num_lower_policies)]

        # reset PER weight for each policy weight tree
        HierarchicalReplayBuffer.init_weight_tree(self, [f"policy_{i}" for i in range(self.num_lower_policies)])

        # if exist data in the buffer, reset the weight with graph count weight
        index = self.sample_indices(0)      # all activate indices
        if len(index) == 0:
            return

        for i in range(self.num_lower_policies):
            tree_name = f"policy_{i}"
            self.weight, self._max_prio, self._min_prio = self.trees[tree_name]
            if i < self.num_factors:
                weight = self.compute_graph_count_weight(index, i) + np.finfo(np.float32).tiny
                weight *= self._max_prio / weight.max()         # scale weight so that least visited graph has max prio
            else:
                weight = np.ones_like(index) * self._max_prio
            super().update_weight(index, weight)
            self.trees[tree_name] = [self.weight, self._max_prio, self._min_prio]

    def reset_graph_count(self) -> None:
        num_possible_graphs = 2 ** (self.num_factors + 1)
        self.graph_count = np.zeros((self.num_factors, num_possible_graphs), dtype=int)
        self.valid_graph_count = np.zeros((self.num_factors, num_possible_graphs), dtype=int)
        self.valid_graph_indices = [set() for _ in range(self.num_factors)]
        self.graph_to_count_idx = np.power(2, np.arange(self.num_factors + 1)).astype(int)
        if not hasattr(self, "true_graph_count"):
            self.true_graph_count = np.zeros((self.num_factors, num_possible_graphs), dtype=int)

    def reset_change_count(self, env) -> None:
        if self.dynamics_per_use_change_count:
            self.num_variables = 0
            self.variable_longest = 0
            self.variable_idx_to_factor_idx = []
            self.num_variables_in_factor = np.zeros(self.num_factors)
            obs_spaces = list(env.dict_obs_space.spaces.values())
            for i, space in enumerate(obs_spaces):
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
                self.num_variables_in_factor[i] = num_variables_in_factor

            self.factor_to_var_mask = np.zeros((self.num_variables, self.variable_longest), dtype=bool)
            var_idx = 0
            for space in obs_spaces:
                if isinstance(space, gym.spaces.Discrete):
                    self.factor_to_var_mask[var_idx, :space.n] = True
                    var_idx += 1
                elif isinstance(space, gym.spaces.MultiDiscrete):
                    for n in space.nvec:
                        self.factor_to_var_mask[var_idx, :n] = True
                        var_idx += 1
                elif isinstance(space, gym.spaces.Box):
                    assert len(space.shape) == 1
                    n = space.shape[0]
                    self.factor_to_var_mask[var_idx:var_idx + n, 0] = True
                    var_idx += n
                else:
                    raise NotImplementedError

            self.variable_change_count = np.zeros(self.num_variables, dtype=np.int64)
            self.variable_no_change_count = np.zeros(self.num_variables, dtype=np.int64)

    def init_HER_utils(self) -> None:
        if not self.use_her:
            return

        self.her_use_episode_tracker = True
        if self.her_use_episode_tracker:
            # to get episode start / end indices faster than iteratively call self.prev() / self.next()
            # data structure:
            #   episode_ptrs:
            #       (total_size, ) int, the index of the episode that the transition belongs to
            #       for easier indexing, we use the index of the first ADDED-to-buffer transition of the episode
            #   episode_start_index: (total_size, ) int, the index of the first transition of the episode, inclusive
            #   episode_end_index: (total_size, ) int, the index of the last transition of the episode, exclusive
            self.episode_ptrs = np.full(self.total_size, -1, dtype=int)
            self.episode_start_index = np.full(self.total_size, -1, dtype=int)
            self.episode_end_index = np.full(self.total_size, -1, dtype=int)

            # keep track of if to-add indices is the first transition of the episode
            self.if_prev_episode_ends = np.ones(self.buffer_num, dtype=bool)
            self.current_episode_ptrs = np.copy(self._offset)

    def init_logging_stats(self) -> None:
        # for logging HER resampling stats
        self.her_stats = [{"her_selected_count_idx": [],
                           "sampled_count_idx": []}
                          for _ in range(self.num_lower_policies)]
        self.dynamics_per_stats = {"graph_accumulated_pred_error_weight": np.zeros_like(self.true_graph_count, dtype=float),
                                   "graph_accumulated_pred_error": np.zeros_like(self.true_graph_count, dtype=float),
                                   "graph_accumulated_change_count_weight": np.zeros_like(self.true_graph_count, dtype=float),
                                   "graph_sampled_count": np.zeros_like(self.true_graph_count, dtype=int)}
        self.dynamics_pred_error_weight_history = np.zeros((self.total_size, self.num_variables), dtype=np.float32)
        self.dynamics_change_count_weight_history = np.zeros((self.total_size, self.num_variables), dtype=np.float32)

    def _compile(self) -> None:
        super()._compile()
        last = index = np.array([0])
        offset = np.array([0, 1])
        _sort_index(index, offset, last, self.buf_size)
        start = np.array([0])
        end = np.array([1])
        _get_lower_buffer_indices(start, end, self.buf_size, self.lower_timeout)

    def init_weight(self, index: Union[int, np.ndarray]) -> None:
        if self.decay_window:
            for pser_stats_i in self.pser_stats:
                pser_stats_i[index] = 0
        HierarchicalReplayBuffer.init_weight(self, index)

    def update_graph_count(self, graph_count_idx1: np.ndarray, addition: bool = True) -> None:
        # graph_count_idx1: (bs, num_factors)
        bs = graph_count_idx1.shape[0]
        graph_count_idx0 = np.repeat(np.arange(self.num_factors)[None], bs, axis=0)     # (bs, num_factors)
        np.add.at(self.graph_count, (graph_count_idx0, graph_count_idx1), 1 if addition else -1)

        # no parent graph should not exist, and it's used by dynamics model to suggest low confidence in graph prediction
        self.graph_count[:, 0] = 0

        self.valid_graph_count = (self.graph_count - self.count_threshold_for_valid_graph).clip(min=0)
        for i in range(self.num_factors):
            self.valid_graph_indices[i].update(graph_count_idx1[:, i])

    def update_gt_graph_count(self, graph_count_idx1: np.ndarray) -> None:
        # graph_count_idx1: (bs, num_factors)
        bs = graph_count_idx1.shape[0]
        graph_count_idx0 = np.repeat(np.arange(self.num_factors)[None], bs, axis=0)     # (bs, num_factors)
        np.add.at(self.true_graph_count, (graph_count_idx0, graph_count_idx1), 1)

    def update_graph(self, index: np.ndarray, new_graph: np.ndarray) -> None:
        self.update_graph_count(self.graph_count_idx[index], addition=False)
        self.graph[index] = new_graph
        self.graph_count_idx[index] = new_graph_count_idx = np.dot(new_graph, self.graph_to_count_idx)
        self.update_graph_count(new_graph_count_idx)

    def compute_change_count_weight(self, index: np.ndarray) -> np.ndarray:
        variable_changed = self.get_if_variable_changes(self[index])
        count = np.where(variable_changed, self.variable_change_count, self.variable_no_change_count)   # (bs, num_variables)
        change_count_weight = 1 / count

        # # normalize to (0, 1), where avg_change_count / min_count is in (0, inf)
        # avg_change_count = np.mean(self.variable_change_count)                                          # scalar
        # change_count_weight = 1 - np.exp(-avg_change_count / count)                                     # (bs, num_variables)
        return change_count_weight

    def aggregate_variable_value(self, value_in_variable: np.ndarray) -> np.ndarray:
        # weight: (bs, num_variables)
        value_in_factor = np.zeros((*value_in_variable.shape[:-1], self.num_factors), dtype=value_in_variable.dtype)
        for var_idx, factor_idx in enumerate(self.variable_idx_to_factor_idx):
            value_in_factor[..., factor_idx] += value_in_variable[..., var_idx]
        return value_in_factor / self.num_variables_in_factor

    def compute_graph_count_weight(self, index: np.ndarray, policy_idx: int) -> np.ndarray:
        graph_count_idx = self.graph_count_idx[index]                                                   # (bs, num_factors)
        factor_graph_count_idx = graph_count_idx[..., policy_idx]                                       # (bs, )
        count = self.valid_graph_count[policy_idx, factor_graph_count_idx]                              # (bs, )
        count = count.astype(float)
        # graph count may be 0 when the graph count is reset after dynamics warmup
        count[count == 0] = np.inf
        count_weight = np.power(count, self.policy_per_graph_count_power)
        return count_weight

    def update_weight(
        self, index: np.ndarray, new_weight: Union[np.ndarray, torch.Tensor]
    ) -> None:
        assert index.ndim == 1, "index, i.e., batch size must be 1D"

        if self.prio_cache.used_per:
            if self.prio_cache.tree_name == "dynamics":
                # update weight for dynamics training
                alpha = self.dynamics_per_alpha

                pred_error_in_variable = np.abs(to_numpy(new_weight))                                       # (bs, num_variables)
                new_weight = pred_error_in_variable.mean(axis=-1)                                           # (bs, )
                if self.dynamics_per_use_change_count:
                    # record previous pred_error_weight and change_count_weight for logging
                    # as it's only for logging, we use ground truth here
                    true_graph_count_idx = self.true_graph_count_idx[index]                                 # (bs, num_factors)
                    prev_pred_error_weight_in_variable = self.dynamics_pred_error_weight_history[index]     # (bs, num_variables)
                    prev_change_count_weight_in_variable = self.dynamics_change_count_weight_history[index]

                    prev_pred_error_weight_in_factor = self.aggregate_variable_value(prev_pred_error_weight_in_variable)
                    prev_change_count_weight_in_factor = self.aggregate_variable_value(prev_change_count_weight_in_variable)
                    pred_error_in_factor = self.aggregate_variable_value(pred_error_in_variable)

                    # accumulate previous pred_error and change_count weight to each sample's graph
                    bs = len(index)
                    graph_count_idx0 = np.repeat(np.arange(self.num_factors)[None], bs, axis=0)     # (bs, num_factors)
                    np.add.at(self.dynamics_per_stats["graph_accumulated_pred_error_weight"],
                              (graph_count_idx0, true_graph_count_idx),
                              prev_pred_error_weight_in_factor)
                    np.add.at(self.dynamics_per_stats["graph_accumulated_change_count_weight"],
                              (graph_count_idx0, true_graph_count_idx),
                              prev_change_count_weight_in_factor)
                    np.add.at(self.dynamics_per_stats["graph_accumulated_pred_error"],
                              (graph_count_idx0, true_graph_count_idx),
                              pred_error_in_factor)
                    np.add.at(self.dynamics_per_stats["graph_sampled_count"],
                              (graph_count_idx0, true_graph_count_idx),
                              1)

                    # compute td_error moving average to scale count_weight
                    self.pred_error_stats.add(pred_error_in_variable.mean(axis=0))

                    # normalize pred error to weight
                    dist = scipy.stats.norm(loc=self.pred_error_stats.mean, scale=self.pred_error_stats.stddev)
                    cdf = dist.cdf(pred_error_in_variable).clip(max=1 - 10 / self.total_size)
                    pred_error_weight = 1 / (1 - cdf)

                    # update history for logging
                    self.dynamics_pred_error_weight_history[index] = pred_error_weight

                    new_weight = pred_error_weight.sum(axis=-1)
            else:
                alpha = self.policy_per_alpha
                policy_idx = int(self.prio_cache.tree_name.split("_")[-1])
                td_error = np.abs(to_numpy(new_weight))

                if self.policy_use_pser:
                    # update weight for policy training
                    n_steps = self.decay_window + 1
                    n_step_indices = np.empty(n_steps * len(index), dtype=index.dtype)
                    n_step_indices[0::n_steps] = index
                    for i in range(1, n_steps):
                        index = self.prev(index)
                        n_step_indices[i::n_steps] = index

                    # new td_error_t ← max{|new_weight|, self.max_prev_decay * old_td_error_t}
                    # for i in range(1, n_steps):
                    #   new td_error_{t - i} ← max{|new_weight| * self.decay_rate ** i, old_td_error_{t - i}}
                    td_error_buf = self.pser_stats[policy_idx]

                    # to avoid too fast decay:
                    # new td_error_t ← max{|new_weight|, self.max_prev_decay * old_td_error_t}
                    # if the data has never been sampled before old_td_error_t = 0
                    n_step_old_td_error = td_error_buf[n_step_indices]
                    n_step_old_td_error[0::n_steps] = self.max_prev_decay * n_step_old_td_error[0::n_steps]

                    n_step_new_td_error = np.empty_like(n_step_old_td_error)
                    n_step_new_td_error[0::n_steps] = td_error
                    for i in range(1, n_steps):
                        n_step_new_td_error[i::n_steps] = td_error * self.decay_rate ** i

                    n_step_new_td_error = np.maximum(n_step_new_td_error, n_step_old_td_error)
                    td_error = n_step_new_td_error[0::n_steps]

                    # filter out repeated indices in n_step_indices
                    # TODO: do not consider repeated indices in index
                    unique_mask = np.concatenate([[True], n_step_indices[1:] != n_step_indices[:-1]])
                    index = n_step_indices[unique_mask]
                    new_weight = td_error_buf[index] = n_step_new_td_error[unique_mask]

                # if policy_idx == self.num_factors, it's state coverage lower policy which doesn't need graph count
                if self.policy_per_use_graph_count and policy_idx < self.num_factors:
                    if self.num_lower_policies == 1:
                        raise NotImplementedError("graph count is not implemented for single lower policy")

                    # compute td_error moving average to scale count_weight
                    td_error_stats_i = self.td_error_stats[policy_idx]
                    td_error_stats_i.add(td_error.mean())

                    count_weight = self.compute_graph_count_weight(index, policy_idx)
                    new_weight = new_weight * count_weight / td_error_stats_i.mean

                    new_weight = self.policy_per_td_error_scale * new_weight + self.policy_per_graph_count_scale * count_weight

            # somehow can't overwrite self.__eps in PrioritizedReplayBuffer, have to copy the code here
            weight = np.abs(to_numpy(new_weight)) + self.__eps
            self.weight[index] = weight ** alpha
            self._max_prio = max(self._max_prio, weight.max())
            self._min_prio = min(self._min_prio, weight.min())

            tree_name = self.prio_cache.tree_name
            self.trees[tree_name] = [self.weight, self._max_prio, self._min_prio]

        self.prio_cache = Batch(used_per=False, tree_name=None)

    def sample(
            self,
            batch_size: int,
            policy_prio: bool = False,
            dynamics_prio: bool = False,
            her_update_achieved_goal: Callable = None,
            policy_idx: int = None,
    ) -> Tuple[Batch, np.ndarray]:
        """
        Replace Tianshou Sample to add no-prio, weights her reward parameters
        """
        assert not (policy_prio and dynamics_prio)
        assert not self.prio_cache.used_per, "must call update_weight() after each sample() call to clear the cache"

        self.prio_cache = Batch(used_per=policy_prio or dynamics_prio)
        if policy_prio or dynamics_prio:
            if dynamics_prio:
                tree_name = "dynamics"
            else:
                if policy_idx is None:
                    assert self.num_lower_policies == 1
                tree_name = f"policy_{0 if policy_idx is None else policy_idx}"
            self.weight, self._max_prio, self._min_prio = self.trees[tree_name]
            self.prio_cache.tree_name = tree_name

        indices = self.sample_indices(batch_size,
                                      use_prio=policy_prio or dynamics_prio,
                                      her_update_achieved_goal=her_update_achieved_goal,
                                      policy_idx=policy_idx)
        return self[indices], indices

    def sample_indices(
            self,
            batch_size: int,
            use_prio: bool = False,
            her_update_achieved_goal: Callable = None,
            policy_idx: int = None,
    ) -> np.ndarray:
        self._restore_cache()
        if use_prio:
            indices = PrioritizedReplayBuffer.sample_indices(self, batch_size)    # sample with priority
        else:
            indices = ReplayBufferManager.sample_indices(self, batch_size)

        if self.use_her and her_update_achieved_goal is not None:
            self.rewrite_transitions(indices, her_update_achieved_goal, policy_idx)
        return indices

    def update_episode_start_end_indices(
        self,
        batch: Batch,
        buffer_ids: Optional[Union[np.ndarray, List[int]]] = None,
    ) -> None:
        if buffer_ids is None:
            buffer_ids = np.arange(self.buffer_num)

        # get the indices of the transitions that will be added to the buffer
        last_added_indices = self.last_index[buffer_ids]
        buffer_offsets = self._offset[buffer_ids]
        buffer_lengths = self._lengths[buffer_ids]
        relative_last_added_indices = last_added_indices - buffer_offsets
        relative_last_added_indices[buffer_lengths == 0] -= 1
        indices_to_add = (relative_last_added_indices + 1) % self.buf_size + buffer_offsets
        new_epi_end_indices = (relative_last_added_indices + 2) % self.buf_size + buffer_offsets

        # when replacing old transitions, update their episode start indices
        are_buffers_full = buffer_lengths == self.buf_size                              # (num_transitions_to_add, ), bool
        if np.any(are_buffers_full):
            # get indices of old transition to replace
            # for their corresponding episodes, the new episode start after replacement is the same as
            # the new episode end indices of to-add transitions
            indices_to_replace = indices_to_add[are_buffers_full]                       # (num_transitions_to_replace, )
            new_epi_start_indices = new_epi_end_indices[are_buffers_full]               # (num_transitions_to_replace, )

            # update episode start indices
            episode_ptrs = self.episode_ptrs[indices_to_replace]                        # (num_transitions_to_replace, )
            self.episode_start_index[episode_ptrs] = new_epi_start_indices

            # when new_epi_start_indices == episode_end_indices, it means the whole episode is replaced
            # we reset the episode_ptrs and episode_end_indices to -1 representing unintialized
            episode_end_indices = self.episode_end_index[episode_ptrs]
            episode_fully_replaced = new_epi_start_indices == episode_end_indices       # (num_transitions_to_replace, ), bool
            fully_replaced_episode_ptrs = episode_ptrs[episode_fully_replaced]          # (num_fully_replaced_episodes, )
            self.episode_start_index[fully_replaced_episode_ptrs] = -1
            self.episode_end_index[fully_replaced_episode_ptrs] = -1

        """
        for each of to-add transitions
            update their episode ptr
                if the transition is the first transition of the episode, initialize its episode ptrs and start indices
                else use current episode ptrs
            update episode end indices
        """
        # update the tracker of current episode ptrs
        # current_episode_ptrs is a shallow copy, so should update self.current_episode_ptrs
        if_start_new_episodes = self.if_prev_episode_ends[buffer_ids]                   # (num_transitions_to_add, ), bool
        new_episode_start_indices = indices_to_add[if_start_new_episodes]               # (num_new_episodes, )

        buffer_ids_w_new_episode = buffer_ids[if_start_new_episodes]                    # (num_new_episodes, )
        self.current_episode_ptrs[buffer_ids_w_new_episode] = new_episode_start_indices
        current_episode_ptrs = self.current_episode_ptrs[buffer_ids]                    # (num_transitions_to_add, )
        self.episode_ptrs[indices_to_add] = current_episode_ptrs

        # initialize ptrs and start indices for new episodes, which share the same value
        new_episode_ptrs = new_episode_start_indices                                    # (num_new_episodes, )
        assert np.all(self.episode_start_index[new_episode_ptrs] == -1), "episode start index should not have been initialized"
        assert np.all(self.episode_end_index[new_episode_ptrs] == -1), "episode end index should not have been initialized"
        self.episode_start_index[new_episode_ptrs] = new_episode_ptrs

        # update the end indices for all to-add transitions
        self.episode_end_index[current_episode_ptrs] = new_epi_end_indices

        # update the tracker of whether current episode ends
        self.if_prev_episode_ends[buffer_ids] = done_mask = np.logical_or(batch.terminated, batch.truncated)

        # when current episode ends, set the episode ptrs from the episode start index to the episode last index, and
        # update episode ptrs for the whole episode
        done_episode_current_ptrs = current_episode_ptrs[done_mask]                     # (num_done_episodes, )
        done_episode_new_ptrs = indices_to_add[done_mask]                               # (num_done_episodes, )
        starts = self.episode_start_index[done_episode_new_ptrs] = self.episode_start_index[done_episode_current_ptrs]
        ends = self.episode_end_index[done_episode_new_ptrs] = self.episode_end_index[done_episode_current_ptrs]
        self.episode_start_index[done_episode_current_ptrs] = -1
        self.episode_end_index[done_episode_current_ptrs] = -1
        for start, end, new_epi_ptr in zip(starts, ends, done_episode_new_ptrs):
            if start < end:
                self.episode_ptrs[start:end] = new_epi_ptr
            else:
                buffer_start = (start // self.buf_size) * self.buf_size
                buffer_end = buffer_start + self.buf_size
                self.episode_ptrs[start:buffer_end] = new_epi_ptr
                self.episode_ptrs[buffer_start:end] = new_epi_ptr

    def get_episode_current_to_terminal_indices(self, current: np.ndarray) -> np.ndarray:
        if self.her_use_episode_tracker:
            episode_ptrs = self.episode_ptrs[current]                                       # (num_episodes, )
            assert not np.any(episode_ptrs == -1), "episode ptrs should have been initialized"
            terminal = self.episode_end_index[episode_ptrs]                                 # (num_episodes, )
            terminal[terminal < current] += self.buf_size

            indices = np.arange(self.horizon)[:, None] + current                            # (horizon, num_episodes)
            indices = indices.clip(max=terminal - 1)

            buffer_end = (current // self.buf_size + 1) * self.buf_size
            indices[indices >= buffer_end] -= self.buf_size
        else:
            indices = np.empty((self.horizon, current.size), dtype=current.dtype)
            indices[0] = current
            for i in range(1, self.horizon):
                indices[i] = self.next(indices[i - 1])
        return indices

    def get_episode_current_to_start_indices(self, current: np.ndarray) -> np.ndarray:
        if self.her_use_episode_tracker:
            episode_ptrs = self.episode_ptrs[current]                                       # (num_episodes, )
            assert not np.any(episode_ptrs == -1), "episode ptrs should have been initialized"
            start = self.episode_start_index[episode_ptrs]                                  # (num_episodes, )

            current[current < start] += self.buf_size

            indices = current - np.arange(self.horizon)[:, None]                            # (horizon, num_episodes)
            indices = indices.clip(min=start)

            buffer_end = (start // self.buf_size + 1) * self.buf_size
            indices[indices >= buffer_end] -= self.buf_size
        else:
            indices = np.empty((self.horizon, current.size), dtype=current.dtype)
            indices[0] = current
            for i in range(1, self.horizon):
                indices[i] = self.prev(indices[i - 1])
        return indices

    def rewrite_transitions(
            self,
            indices: np.ndarray,
            her_update_achieved_goal: Callable,
            policy_idx: int = None,
    ) -> None:
        """
        Re-write the goal of some sampled transitions' episodes according to HER.

        Currently applies only HER's 'future' strategy. The new goals will be written \
        directly to the internal batch data temporarily and will be restored right \
        before the next sampling or when using some of the buffer's method (e.g. \
        `add`, `save_hdf5`, etc.). This is to make sure that n-step returns \
        calculation etc., performs correctly without additional alteration.

        policy_idx & graph_count are used to select graph that are less visited
        """
        if indices.size == 0:
            return

        # we need to apply her to each buffer
        current = _sort_index(indices, self._extend_offset, self.last_index, self.buf_size)

        # Construct episode trajectories
        indices = self.get_episode_current_to_terminal_indices(current)

        # Calculate future timestep to use
        terminal = indices[-1]
        episodes_len = (terminal - current + self.buf_size) % self.buf_size
        future_offset = np.random.uniform(size=len(indices[0])) * episodes_len
        future_offset = np.round(future_offset).astype(int)
        future_t = indices[future_offset, np.arange(len(current))]

        # Compute indices
        #   open indices are used to find longest, unique trajectories among
        #   presented episodes
        unique_ep_open_indices = np.sort(np.unique(terminal, return_index=True)[1])
        unique_ep_indices = indices[:, unique_ep_open_indices]
        #   close indices are used to find max future_t among presented episodes
        unique_ep_close_indices = np.hstack(
            [(unique_ep_open_indices - 1)[1:],
             len(terminal) - 1]
        )

        # --------------------------------------- modification starts --------------------------------------- #
        if not self._save_obs_next:
            raise NotImplementedError

        # episode indices that will be altered
        her_ep_mask = np.random.rand(len(unique_ep_open_indices)) < self.future_p
        if policy_idx is not None:

            # add episode indices to her if the desired factor is different from the policy factor
            unique_current = current[unique_ep_open_indices]
            desired_factor = self.option_choice[unique_current]                                 # (num_unique_episodes, 1)
            her_ep_mask[desired_factor != policy_idx] = True

            # select future_t based on graph count
            her_ep_indices = unique_ep_close_indices[her_ep_mask]
            her_indices = indices[:, her_ep_indices]                                            # (horizon, num_her_episodes)

            count_weight = self.compute_graph_count_weight(her_indices, policy_idx)             # (horizon, num_her_episodes)
            count_weight = count_weight.clip(min=self.__eps)
            unique_mask = (her_indices[1:] != her_indices[:-1])                                 # (horizon - 1, num_her_episodes)
            count_weight[1:][~unique_mask] = 0                                                  # (horizon, num_her_episodes)
            count_weight = count_weight.T                                                       # (num_her_episodes, horizon)

            prob = count_weight / count_weight.sum(axis=-1, keepdims=True)                      # (num_her_episodes, horizon)

            # sample each episode's future_t
            num_her_episodes = len(her_ep_indices)
            cum_prob = prob.cumsum(axis=-1)                                                     # (num_her_episodes, horizon)
            u = np.random.rand(num_her_episodes, 1)                                             # (num_her_episodes, 1)
            choice = (u < cum_prob).argmax(axis=-1)                                             # (num_her_episodes, )
            future_t[her_ep_indices] = her_goal_indices = her_indices[choice, np.arange(num_her_episodes)]

            sampled_factor_graph_count_idx = self.graph_count_idx[current, policy_idx]              # (num_total_episodes, )
            selected_factor_graph_count_idx = self.graph_count_idx[her_goal_indices, policy_idx]    # (num_her_episodes, )
            self.her_stats[policy_idx]["her_selected_count_idx"].extend(selected_factor_graph_count_idx)
            self.her_stats[policy_idx]["sampled_count_idx"].extend(sampled_factor_graph_count_idx)

            self.relabel_weight_with_her_goal(her_goal_indices, policy_idx)

        # update the achieved goal of future_obs
        her_ep_indices = unique_ep_close_indices[her_ep_mask]                                   # (num_her_episodes, )
        her_future_t = future_t[her_ep_indices]
        if her_future_t.size == 0:
            return

        future_batch = her_update_achieved_goal(self[her_future_t])
        future_achieved_goal = future_batch.obs_next.achieved_goal

        # TODO: if termination is also state dependent, then it should get reassigned here
        #   episode indices that will be altered
        # TODO: we don't re-assign rew here, assume it will be handled by upper_policy/lower_policy.process_fn()
        # Cache original meta
        self._altered_indices = unique_ep_indices[:, her_ep_mask]
        self._original_meta = self._meta.obs.desired_goal[self._altered_indices]
        # Re-assign goals via broadcast assignment
        self._meta.obs.desired_goal[self._altered_indices] = future_achieved_goal[None]
        if self._save_obs_next:
            self._meta.obs_next.desired_goal[self._altered_indices] = future_achieved_goal[None]

    def relabel_weight_with_her_goal(
        self, her_goal_index: np.ndarray, policy_idx: int,
    ) -> None:
        if not self.prio_cache.used_per or self.prio_cache.tree_name == "dynamics":
            return

        if not self.policy_per_use_graph_count or policy_idx >= self.num_factors:
            return

        # compute td_error moving average to scale count_weight
        td_error_moving_average_i = self.td_error_stats[policy_idx].mean
        if td_error_moving_average_i == 0:
            return

        assert self.prio_cache.tree_name == f"policy_{policy_idx}"

        # if policy_idx == self.num_factors, it's state coverage lower policy which doesn't need graph count
        if self.num_lower_policies == 1:
            raise NotImplementedError("graph count is not implemented for single lower policy")

        # fetch all indices of the trajectory before the goal
        last = her_goal_index
        indices = self.get_episode_current_to_start_indices(last)
        indices = indices.T                                                         # (num_her_episodes, horizon)

        count_weight = self.compute_graph_count_weight(indices, policy_idx)         # (num_her_episodes, horizon)
        her_goal_count_weight = count_weight[:, 0:1]                                # (num_her_episodes, 1)

        # cache original graph count idx and reached graph counter
        self._her_altered_indices = indices
        self._her_altered_policy_idx = policy_idx
        self._original_graph_count_idx = original_graph_count_idx = self.graph_count_idx[indices, policy_idx]
        self._original_reached_graph_counter = self.obs_next.reached_graph_counter[indices]

        # if her relabeled graph is less visited, use it to overwrite the weight of all previous indices in the trajectory
        # overwrite graph_count_idx
        her_goal_graph_count_idx = original_graph_count_idx[:, 0:1]                 # (num_her_episodes, 1)
        new_graph_count_idx = np.where(count_weight < her_goal_count_weight,
                                       her_goal_graph_count_idx,
                                       original_graph_count_idx)
        self.graph_count_idx[indices, policy_idx] = new_graph_count_idx

        # update weight with overwritten graph_count_idx
        count_weight = np.maximum(count_weight, her_goal_count_weight)              # (num_her_episodes, horizon)

        unique_mask = (indices[:, 1:] != indices[:, :-1])                           # (num_her_episodes, horizon - 1)
        unique_indices, count_weight = indices[:, 1:][unique_mask], count_weight[:, 1:][unique_mask]

        # scale td_error_weight by td_error moving average and count_weight
        td_error_buf = self.pser_stats[policy_idx]
        td_error_weight = td_error_buf[unique_indices]
        td_error_weight = td_error_weight * count_weight / td_error_moving_average_i

        new_weight = self.policy_per_td_error_scale * td_error_weight + self.policy_per_graph_count_scale * count_weight

        # somehow can't overwrite self.__eps in PrioritizedReplayBuffer, have to copy the code here
        weight = np.abs(to_numpy(new_weight)) + self.__eps
        self.weight[unique_indices] = weight ** self.policy_per_alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

        tree_name = self.prio_cache.tree_name
        self.trees[tree_name] = [self.weight, self._max_prio, self._min_prio]

        # overwrite obs.reached_graph_counter
        achieved_graph = self.graph[indices, policy_idx]                            # (num_her_episodes, horizon, num_factors + 1)
        desired_graph = achieved_graph[:, 0:1, :]                                   # (num_her_episodes, 1, num_factors + 1)
        reached_graph = np.all(achieved_graph == desired_graph, axis=-1)            # (num_her_episodes, horizon)
        reached_graph[:, 1:][~unique_mask] = False
        reached_graph = np.flip(reached_graph, axis=-1)                             # flip because indices are in reverse order
        reached_graph_counter = np.cumsum(reached_graph, axis=-1)                   # (num_her_episodes, horizon)
        # normalize counter the same way as collector
        reached_graph_counter = np.clip(reached_graph_counter / self.lower_reached_graph_threshold, 0, 1)
        reached_graph_counter = np.flip(reached_graph_counter, axis=-1)             # (num_her_episodes, horizon)

        reached_graph_counter = reached_graph_counter[:, 1:][unique_mask]
        self.obs_next.reached_graph_counter[unique_indices] = reached_graph_counter

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Batch:
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indices = self.sample_indices(0) if index == slice(None) \
                else self._indices[:len(self)][index]
        else:
            indices = index  # type: ignore
        if self.prio_cache.used_per:
            batch = super().__getitem__(indices)
        else:
            batch = ReplayBufferManager.__getitem__(self, indices)
        for k in self._extra_sample_keys:
            batch.__dict__[k] = self._meta[k][indices]
        return batch

    def _restore_cache(self) -> None:
        if hasattr(self, '_her_altered_indices') and self._her_altered_indices.size > 0:
            assert hasattr(self, '_her_altered_policy_idx')
            self._meta.graph_count_idx[self._her_altered_indices, self._her_altered_policy_idx] = self._original_graph_count_idx
            self._meta.obs_next.reached_graph_counter[self._her_altered_indices] = self._original_reached_graph_counter
            # Clean
            self._original_graph_count_idx = Batch()
            self._her_altered_indices = np.array([])

        HierarchicalReplayBuffer._restore_cache(self)

    def get_if_variable_changes(self, batch):
        state = batch.obs.observation if type(batch.obs) == Batch else batch.obs
        current_variable = np.zeros((state.shape[0], self.num_variables, self.variable_longest),
                                    dtype=state.dtype)
        current_variable[:, self.factor_to_var_mask] = state            # (batch_size, num_variables, variable_longest)

        next_state = batch.obs_next.observation if type(batch.obs_next) == Batch else batch.obs_next
        next_variable = np.zeros((next_state.shape[0], self.num_variables, self.variable_longest),
                                 dtype=next_state.dtype)
        next_variable[:, self.factor_to_var_mask] = next_state          # (batch_size, num_variables, variable_longest)

        variable_changed = np.any(current_variable != next_variable, axis=-1)
        return variable_changed                                         # (batch_size, num_variables)

    def add(
        self,
        batch: Batch,
        buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._restore_cache()

        self.update_graph_count(batch.graph_count_idx)
        self.update_gt_graph_count(batch.true_graph_count_idx)

        if self.dynamics_per_use_change_count:
            variable_changed = self.get_if_variable_changes(batch)
            self.variable_change_count += variable_changed.sum(axis=0)
            self.variable_no_change_count += (~variable_changed).sum(axis=0)

        if self.use_her and self.her_use_episode_tracker:
            self.update_episode_start_end_indices(batch, buffer_ids)

        ptrs, ep_rews, ep_lens, ep_idxs = super().add(batch, buffer_ids)

        if self.dynamics_per_use_change_count:
            # erase the recorded weight of the old data
            self.dynamics_pred_error_weight_history[ptrs] = 0
            self.dynamics_change_count_weight_history[ptrs] = 0

        return ptrs, ep_rews, ep_lens, ep_idxs

    def get_upper_indices(
        self,
        buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> np.ndarray:
        ptrs = []
        for buffer_id in buffer_ids:
            buffer = self.buffers[buffer_id]
            if len(buffer) == buffer.maxsize:
                ptrs.append(buffer._index + self._offset[buffer_id])
        ptrs = np.array(ptrs, dtype=int)
        if "upper_buffer_index" in self._meta:
            upper_indices = self.upper_buffer_index[ptrs]
        else:
            # lower buffer data not initialized yet
            upper_indices = np.array([], dtype=int)
        return upper_indices

    def get_lower_buffer_indices(self, start, end):
        is_scalar_input = np.isscalar(start)
        if is_scalar_input:
            start = np.array([start])
            end = np.array([end])

        # flat_indices, (num_indices, lower_timeout)
        indices, mask = _get_lower_buffer_indices(start, end, self.buf_size, self.lower_timeout)
        indices = indices[mask]

        if is_scalar_input:
            return indices, mask[0]
        else:
            return indices, mask

    def update_upper_buffer_idx(self, start, end, upper_buffer_idx):
        idx, _ = self.get_lower_buffer_indices(start, end)
        self.upper_buffer_index[idx] = upper_buffer_idx

    # below copied from HERReplayBufferManager
    def save_hdf5(self, path: str, compression: Optional[str] = None) -> None:
        self._restore_cache()
        return super().save_hdf5(path, compression)

    def set_batch(self, batch: Batch) -> None:
        self._restore_cache()
        return super().set_batch(batch)

    def update(self, buffer) -> np.ndarray:
        self._restore_cache()
        return super().update(buffer)

    def restore_cache(self):
        self._restore_cache()

    def logging(self, writer: SummaryWriter, step: int) -> bool:
        logged_her = self.logging_her_stats(writer, step)
        logged_dynamics_per = self.logging_dynamics_per_stats(writer, step)
        return logged_her or logged_dynamics_per

    def logging_her_stats(self, writer: SummaryWriter, step: int) -> bool:
        logged = False

        for factor, stats in enumerate(self.her_stats):
            her_selected_count_idx = stats["her_selected_count_idx"]
            sampled_count_idx = stats["sampled_count_idx"]
            if not her_selected_count_idx or not sampled_count_idx:
                continue

            logged = True

            # convert samples to (uqnique, count)
            sampled_count_idxes, sampled_count = np.unique(sampled_count_idx, return_counts=True)
            her_selected_count_idxes, her_selected_counts = np.unique(her_selected_count_idx, return_counts=True)
            sample_stats = dict(zip(sampled_count_idxes, sampled_count))
            her_stats = dict(zip(her_selected_count_idxes, her_selected_counts))

            # add graph that has never been sampled
            for j, count in enumerate(self.valid_graph_count[factor]):
                if count > 0:
                    if j not in her_selected_count_idxes:
                        her_stats[j] = 0
                    if j not in sampled_count_idxes:
                        sample_stats[j] = 0

            total_her_count = np.sum(list(her_stats.values()))
            total_sampled_count = np.sum(list(sample_stats.values()))
            total_graph_count = np.sum(self.valid_graph_count[factor])

            graph_names = []
            her_selected_count_percents = []
            sampled_count_percents = []
            graph_count_percents = []

            factor_names = self.extractor.factor_names + ["act"]
            for count_idx, her_selected_count in sorted(her_stats.items(), key=lambda item: item[1]):
                her_selected_count_percents.append(100 * her_selected_count / total_her_count)
                sampled_count_percents.append(100 * sample_stats.get(count_idx, 0) / total_sampled_count)

                parents = self.count_idx_to_graph[count_idx].astype(bool)
                graph_name = ", ".join([factor_names[j] for j, p in enumerate(parents)
                                        if p])
                graph_name = graph_name + " -> " + factor_names[factor]
                graph_names.append(graph_name)

                graph_count = self.valid_graph_count[factor, count_idx]
                graph_count_percents.append(100 * graph_count / total_graph_count)

            num_graphs = len(graph_names)
            fig = plt.figure(figsize=(10, max(num_graphs * 1, 3)))

            # plot HER resampling graph frequency
            ax = plt.gca()
            y = np.arange(num_graphs)

            height = 0.2
            rects = ax.barh(y + height, her_selected_count_percents,  height=height, label="HER relabeling frequency")
            ax.bar_label(rects, label_type='edge', fmt="%.3f", padding=3)
            rects = ax.barh(y, sampled_count_percents, height=height, label="PER sampling frequency")
            ax.bar_label(rects, label_type='edge', fmt="%.3f", padding=3)

            # plot total graph count info
            rects = ax.barh(y - height, graph_count_percents, height=height, label="percentage in buffer")
            ax.bar_label(rects, label_type='edge', fmt="%.3f", padding=3)

            plt.xlim([0, 1.1 * max(np.max(her_selected_count_percents),
                                   np.max(sampled_count_percents),
                                   np.max(graph_count_percents))])

            ax.set_yticks(y)
            ax.set_yticklabels(graph_names)

            plt.legend(loc="lower right")
            fig.tight_layout()
            writer.add_figure(f"her_stats_{factor}", fig, step)
            plt.close("all")

        self.her_stats = [{"her_selected_count_idx": [],
                           "sampled_count_idx": []}
                          for _ in range(self.num_lower_policies)]

        return logged

    def logging_dynamics_per_stats(self, writer: SummaryWriter, step: int) -> bool:
        graph_sampled_count = self.dynamics_per_stats["graph_sampled_count"]
        if np.all(graph_sampled_count == 0):
            return False

        # average the accumulated stats
        graph_pred_error = self.dynamics_per_stats["graph_accumulated_pred_error"]
        graph_pred_error_weight = self.dynamics_per_stats["graph_accumulated_pred_error_weight"]
        graph_change_count_weight = self.dynamics_per_stats["graph_accumulated_change_count_weight"]

        total_sampled_count = np.sum(graph_sampled_count[0])
        total_graph_count = np.sum(self.true_graph_count[0])

        graph_names = []
        sampled_count_percents = []
        avg_pred_error = []
        avg_pred_error_weight = []
        avg_change_count_weight = []
        graph_count_percents = []

        factor_names = self.extractor.factor_names + ["act"]

        sorted_idx = np.unravel_index(np.argsort(graph_sampled_count, axis=None), graph_sampled_count.shape)
        for factor, graph_count_idx in zip(*sorted_idx):
            graph_sampled_count_i = graph_sampled_count[factor, graph_count_idx]
            true_graph_count_i = self.true_graph_count[factor, graph_count_idx]

            if true_graph_count_i == 0:
                assert graph_sampled_count_i == 0
                continue

            parents = self.count_idx_to_graph[graph_count_idx].astype(bool)
            graph_name = ", ".join([factor_names[j] for j, p in enumerate(parents)
                                    if p])
            graph_name = graph_name + " -> " + factor_names[factor]
            graph_names.append(graph_name)

            sampled_count_percents.append(100 * graph_sampled_count_i / total_sampled_count)

            graph_pred_error_i = graph_pred_error[factor, graph_count_idx]
            graph_pred_error_weight_i = graph_pred_error_weight[factor, graph_count_idx]
            graph_change_count_weight_i = graph_change_count_weight[factor, graph_count_idx]

            if graph_sampled_count_i == 0:
                assert graph_pred_error_i == 0
                assert graph_pred_error_weight_i == 0
                assert graph_change_count_weight_i == 0
                avg_pred_error.append(0)
                avg_pred_error_weight.append(0)
                avg_change_count_weight.append(0)
            else:
                avg_pred_error.append(graph_pred_error_i / graph_sampled_count_i)
                avg_pred_error_weight.append(graph_pred_error_weight_i / graph_sampled_count_i)
                avg_change_count_weight.append(100 * graph_change_count_weight_i / graph_sampled_count_i)

            graph_count = self.true_graph_count[factor, graph_count_idx]
            graph_count_percents.append(100 * graph_count / total_graph_count)


        # plot HER resampling graph frequency
        height = 0.4
        colors = [p['color'] for p in plt.rcParams['axes.prop_cycle']]

        contents = [
            sampled_count_percents, graph_count_percents,
            avg_pred_error,
            avg_pred_error_weight, avg_change_count_weight
        ]
        labels = [
            "PER sampling frequency", "frequency in buffer",
            "average pred error",
            "average pred error weight",
        ]
        axes = [0, 0, 1, 2]

        num_bars = len(contents)
        num_graphs = len(graph_names)

        fig = plt.figure(figsize=(10, max(num_graphs * 1, 3)))
        y = np.arange(num_graphs) * (num_bars + 1) * height

        ax = plt.gca()
        j = 0
        axes_dict = {0: ax}
        legs = []
        for axis_id_i in np.unique(axes):
            try:
                ax_i = axes_dict[axis_id_i]
            except KeyError:
                axes_dict[axis_id_i] = ax_i = ax.twiny()

            content_maxs = []
            for content, label, axis_id_j in zip(contents, labels, axes):
                if axis_id_j != axis_id_i:
                    continue
                offset = num_bars // 2 + 0.5
                rects = ax_i.barh(y + height * (offset - j), content, height=height, color=colors[j], label=label)
                ax_i.bar_label(rects, label_type='edge', fmt="%.3f", padding=3)
                content_maxs.append(np.max(content))
                j += 1

            ax_i.set_xlim([0, 1.1 * max(content_maxs)])
            ax_i.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labeltop=False
            )
            leg = plt.legend()
            legs.append(leg)

        ax.set_yticks(y)
        ax.set_yticklabels(graph_names)

        patches = legs[0].get_patches()
        texts = legs[0].get_texts()
        for i, leg in enumerate(legs):
            if i != 0:
                patches = patches + leg.get_patches()
                texts = texts + leg.get_texts()
            if i != len(legs) - 1:
                leg.remove()

        plt.legend(patches, [text.get_text() for text in texts], loc="lower right")
        fig.tight_layout()
        writer.add_figure(f"dynamics_per_stats", fig, step)
        plt.close("all")

        self.dynamics_per_stats = {"graph_accumulated_pred_error_weight": np.zeros_like(self.true_graph_count, dtype=float),
                                   "graph_accumulated_change_count_weight": np.zeros_like(self.true_graph_count, dtype=float),
                                   "graph_accumulated_pred_error": np.zeros_like(self.true_graph_count, dtype=float),
                                   "graph_sampled_count": np.zeros_like(self.true_graph_count, dtype=int)}

        return True

@njit
def _sort_index(
    index: np.ndarray,
    offset: np.ndarray,
    last_index: np.ndarray,
    buf_size: int,
) -> np.ndarray:
    index = index % offset[-1]
    index = np.sort(index)
    sorted_index = np.zeros_like(index)
    for start, end, last in zip(offset[:-1], offset[1:], last_index):
        mask = (start <= index) & (index < end)
        if np.any(mask):
            subind = index[mask]

            subind[subind <= last] += buf_size
            subind = np.sort(subind)
            subind[subind >= end] -= buf_size

            sorted_index[mask] = subind
    return sorted_index

@njit
def _get_lower_buffer_indices(
    start: np.ndarray,
    end: np.ndarray,
    buf_size: int,
    timeout: int,
):
    # return a flat concatenated lower indices, and a mask to map the indices to (num_indices, timeout) format
    num_indices = len(start)
    indices = np.zeros((num_indices, timeout), dtype=np.int64)
    mask = np.zeros((num_indices, timeout), dtype=np.bool_)

    for i, (start_i, end_i) in enumerate(zip(start, end)):
        if end_i >= start_i:
            num_indices_i = end_i - start_i + 1
            for idx in range(start_i, end_i + 1):
                indices[i, idx - start_i] = idx
        else:
            buffer_start = (start_i // buf_size) * buf_size
            buffer_end = buffer_start + buf_size
            assert buffer_start <= start_i and buffer_end > end_i
            num_indices_i = (buffer_end - start_i) + (end_i + 1 - buffer_start)
            for idx in range(start_i, buffer_end):
                indices[i, idx - start_i] = idx
            offset = buffer_end - start_i
            for idx in range(buffer_start, end_i + 1):
                indices[i, idx - buffer_start + offset] = idx

        mask[i, :num_indices_i] = True
    return indices, mask
