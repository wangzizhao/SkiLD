import time, copy
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
from array2gif import write_gif

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.data import Collector
from tianshou.env import BaseVectorEnv

from Causal import Dynamics
from Option.hierarchical_model import HierarchicalModel
from State.buffer import HierarchicalReplayBuffer, VectorHierarchicalReplayBufferManager
from State.utils import compute_proximity, reset_upper_memory, update_upper_memory_trajectory, upper_buffer_add


def construct_her_obs(achieved_goal, desired_goal, env_obs, reached_graph_counter):
    return Batch(achieved_goal=achieved_goal,
                 desired_goal=desired_goal,
                 # if env is goal-based, only upper policy needs to consider the goal, and lower policy doesn't need to
                 observation=env_obs.observation if isinstance(env_obs, Batch) else env_obs,
                 reached_graph_counter=reached_graph_counter)


class HRLCollector(Collector):
    """
    Modified collector for Hierarchical RL policies. In particular, this method \
    records additional information into the replay buffer and calls \
    HierarchicalModel.update() at each time step. Changed lines identified with ###EDIT###

    Collector enables the policy to interact with different types of envs with \
    exact number of steps or episodes.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        If set to None, it will not store the data. Default to None.
    :param function preprocess_fn: a function called before the data has been added to
        the buffer, see issue #42 and :ref:`preprocess_fn`. Default to None.
    :param bool exploration_noise: determine whether the action needs to be modified
        with corresponding policy's exploration noise. If so, "policy.
        exploration_noise(act, batch)" will be called automatically to add the
        exploration noise into action. Default to False.

    The "preprocess_fn" is a function called before the data has been added to the
    buffer with batch format. It will receive only "obs" and "env_id" when the
    collector resets the environment, and will receive the keys "obs_next", "rew",
    "terminated", "truncated, "info", "policy" and "env_id" in a normal env step.
    Alternatively, it may also accept the keys "obs_next", "rew", "done", "info",
    "policy" and "env_id".
    It returns either a dict or a :class:`~tianshou.data.Batch` with the modified
    keys and values. Examples are in "test/base/test_collector.py".

    .. note::

        Please make sure the given environment has a time limitation if using n_episode
        collect option.

    .. note::

        In past versions of Tianshou, the replay buffer that was passed to `__init__`
        was automatically reset. This is not done in the current implementation.
    """

    def __init__(
        self,
        policy: HierarchicalModel,
        dynamics: Dynamics,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[VectorHierarchicalReplayBufferManager] = None,
        upper_buffer: Optional[HierarchicalReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
        name: str = "",
        root_dir: str = "",
        num_factors: int = 30,
        upper_rew_aggregation: str = "sum",
        reached_graph_threshold: int = 1,
        lower_training_ignore_lower_done: bool = False,
        save_gif_num: bool = False,
        scripted_lower: bool = False,
    ) -> None:
        self.name = name
        self.dynamics = dynamics  # this is needed for reset
        self.upper_buffer = upper_buffer
        self.reached_graph_threshold = float(reached_graph_threshold)
        self.exp_path = root_dir
        self.num_factors = num_factors
        self.upper_rew_aggregation = upper_rew_aggregation
        self.lower_training_ignore_lower_done = lower_training_ignore_lower_done
        self.save_gif_num = save_gif_num
        self.scripted_lower = scripted_lower

        self.epoch_num = 0
        self.graph_to_count_idx = np.power(2, np.arange(num_factors + 1)).astype(int)

        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)

        self.random_all = self.random_upper_only = False

        # extra logging metrics
        self.epi_logging_metrics = Batch(rew=np.zeros(self.env_num),
                                         success=np.zeros(self.env_num, dtype=bool),
                                         lower_rew=np.zeros(self.env_num),
                                         lower_time=np.zeros(self.env_num),
                                         upper_rew=np.zeros(self.env_num))
        self.lower_logging_metrics = Batch(reached_graph=np.zeros(self.env_num, dtype=bool),
                                           reached_goal=np.zeros(self.env_num, dtype=bool),
                                           updated=np.zeros(self.env_num, dtype=bool))

    def _assign_buffer(self, buffer: Optional[ReplayBuffer]) -> None:
        """Check if the buffer matches the constraint."""
        if buffer is None:
            buffer = VectorReplayBuffer(self.env_num, self.env_num)
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
            if isinstance(buffer, CachedReplayBuffer):
                assert buffer.cached_buffer_num >= self.env_num
        else:  # ReplayBuffer or PrioritizedReplayBuffer
            assert buffer.maxsize > 0
            if self.env_num > 1:
                if type(buffer) == ReplayBuffer:
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                else:
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"Cannot use {buffer_type}(size={buffer.maxsize}, ...) to collect "
                    f"{self.env_num} envs,\n\tplease use {vector_type}(total_size="
                    f"{buffer.maxsize}, buffer_num={self.env_num}, ...) instead."
                )
        self.buffer = buffer

    def _reset_env_with_ids(
        self,
        local_ids: Union[List[int], np.ndarray],
        global_ids: Union[List[int], np.ndarray],
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        obs_reset, info = self.env.reset(global_ids, **gym_reset_kwargs)
        if self.preprocess_fn:
            processed_data = self.preprocess_fn(
                obs=obs_reset, info=info, env_id=global_ids
            )
            obs_reset = processed_data.get("obs", obs_reset)
            info = processed_data.get("info", info)
        self.data.info[local_ids] = info

        # adapted below this line
        self.data.time_upper[local_ids] = 0
        self.data.time_lower[local_ids] = 0

        self.data.upper_obs_next[local_ids] = obs_reset

        # just a placeholder, achieved and desired target goals should be reassigned after calling upper policy
        self.data.obs_next[local_ids] = construct_her_obs(achieved_goal=self.data.obs.achieved_goal[local_ids],
                                                          desired_goal=self.data.obs.desired_goal[local_ids],
                                                          env_obs=obs_reset,
                                                          reached_graph_counter=np.zeros(len(local_ids)),)

    def reset_env(self, gym_reset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().reset_env(gym_reset_kwargs=gym_reset_kwargs)

        obs = self.data.obs

        self.data.update(
            time_upper=np.zeros(len(self.env)),
            time_lower=np.zeros(len(self.env)),
            needs_resample=np.ones(len(self.env), dtype=bool),
            upper_obs=obs,
            target=self.policy.lower.get_target(self.data),         # used by history policy
            upper_buffer_index=np.zeros(len(self.env), dtype=int),  # placeholders, the values will not be used by lower buffer
        )

        # desired_goal should be assigned after calling upper policy
        self.data.obs = Batch(observation=obs.observation if isinstance(obs, Batch) else obs,
                              reached_graph_counter=np.zeros(len(self.env)),)

        self.upper_buffer_memory = None
        self.upper_trajectories = [list() for _ in range(self.env_num)]

    def record_reset_lower_logging_metrics(self, lower_metrics, local_ids, global_ids):
        num_reset = len(global_ids)
        self.policy.update_lower_stats(self.lower_logging_metrics[global_ids], self.data.obs.desired_goal[local_ids])
        for k in lower_metrics.keys():
            lower_metrics[k].extend(self.lower_logging_metrics.get(k)[global_ids])
        self.lower_logging_metrics[global_ids] = Batch(reached_graph=np.zeros(num_reset, dtype=bool),
                                                       reached_goal=np.zeros(num_reset, dtype=bool),
                                                       updated=np.zeros(num_reset, dtype=bool))
        return lower_metrics

    def record_reset_all_logging_metrics(self, epi_metrics, lower_metrics, local_ids, global_ids):
        num_reset = len(global_ids)
        for k in epi_metrics.keys():
            epi_metrics[k].extend(self.epi_logging_metrics.get(k)[global_ids])
        self.epi_logging_metrics[global_ids] = Batch(rew=np.zeros(num_reset),
                                                     success=np.zeros(num_reset, dtype=bool),
                                                     lower_time=np.zeros(num_reset),
                                                     lower_rew=np.zeros(num_reset),
                                                     upper_rew=np.zeros(num_reset))
        lower_metrics = self.record_reset_lower_logging_metrics(lower_metrics, local_ids, global_ids)
        return epi_metrics, lower_metrics

    def update_logging_metrics(self, data, global_ids):
        epi_metric_stats = Batch(rew=data.env_rew,
                                 lower_rew=data.reward_chain.lower,
                                 lower_time=data.time_lower,
                                 upper_rew=data.reward_chain.upper,
                                 success=data.info.get("success", np.zeros_like(data.env_rew, dtype=bool)))
        lower_metric_states = data.lower_reached
        for logging, stats in zip([self.epi_logging_metrics, self.lower_logging_metrics],
                                  [epi_metric_stats, lower_metric_states]):
            for k, v in logging.items():
                if k not in stats:
                    continue
                assert isinstance(v, np.ndarray)
                if v.dtype == bool:
                    # success type: use np.logical_or to upate
                    v[global_ids] = v[global_ids] | stats[k]
                elif isinstance(v[0], np.floating):  # check if v.dtype is float16/32/64
                    # reward type: use np.add to update
                    v[global_ids] = v[global_ids] + stats[k]
                else:
                    raise NotImplementedError

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode.

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param int n_step: how many steps you want to collect.
        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default
            to False.
        :param float render: the sleep time between rendering consecutive frames.
            Default to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rews`` array of episode reward over collected episodes.
            * ``lens`` array of episode length over collected episodes.
            * ``idxs`` array of episode start index in buffer over collected episodes.
            * ``rew`` mean of episodic rewards.
            * ``len`` mean of episodic lengths.
            * ``rew_std`` standard error of episodic rewards.
            * ``len_std`` standard error of episodic lengths.
        """
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        # extra logging metrics
        epi_metrics = {k: [] for k in self.epi_logging_metrics.keys()}
        lower_metrics = {k: [] for k in self.lower_logging_metrics.keys()}

        random_all = random | self.random_all
        random = random_all | self.random_upper_only
        assert not (random_all and self.random_upper_only), \
            "can't randomly sample both upper and lower actions and randomly sample upper actions only at the same time"


        ###GIF EDIT###
        # First, test whether we can save gif from mini-behavior
        save_gif = self.save_gif_num and not self.policy.training
        num_of_gifs = min(self.save_gif_num, len(ready_env_ids))
        pause = 0.15

        if save_gif:
            frames = [[] for _ in range(num_of_gifs)]

        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)
            # get the next action
            if random:
                act_sample = self.policy.random_sample(random_upper_only=self.random_upper_only, batch=self.data)
                self.data.update(act=act_sample.act,
                                 option_resample=act_sample.resampled, option_choice=act_sample.option_choice,
                                 action_chain=act_sample.action_chain, sampled_action_chain=act_sample.sampled_action_chain)
                self.data.obs.desired_goal = act_sample.sampled_action_chain.upper
            else:

                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)

                # option_choice: a representation of the lower level option selection
                #                (1 hot if factored, dag representation), returned by forward in policy.option_choice
                # action_chain: policy.action_chain should fill this in
                self.data.update(policy=policy, act=act,
                                 option_resample=result.resampled, option_choice=result.option_choice,
                                 action_chain=result.action_chain, sampled_action_chain=result.sampled_action_chain)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)

            # uncomment the following to use a manual lower
            if self.scripted_lower:
                for i, id in enumerate(ready_env_ids):
                    self.env.set_env_attr("desired_goal", self.data.obs.desired_goal[i], id=id)

            # step in env
            obs_next, rew, terminated, truncated, info = self.env.step(
                action_remap,  # type: ignore
                ready_env_ids
            )
            done = np.logical_or(terminated, truncated)

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info
            )
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                        act=self.data.act,
                    )
                )

            # ---------------------------------- major modification starts ---------------------------------- #

            # self.data.rew will be replaced by lower policy reward, so use self.data.env_rew to keep a copy
            # obs_next for the upper policy
            self.data.update(env_rew=rew,
                             upper_obs_next=obs_next)

            # Initialize and update primitive values first
            # desired target goal: the goal given by the upper action
            # time_lower: updates the time since last sample for the upper policy,
            #             used by rrt to calculate lower policy truncation (timeout)
            #             (length n vector where n is the number in the lower hierarchy)
            time_lower = self.data.time_lower * (1 - self.data.option_resample.astype(int)) + 1  # TODO: might be last_new_action
            # time_upper: updates the time since last sample for the upper policy, not used for now
            time_upper = self.data.time_upper + self.data.option_resample.astype(int)
            self.data.obs.reached_graph_counter = self.data.obs.reached_graph_counter * (1 - self.data.option_resample)

            # target, next_target, target_diff: gets the factored state for achieved goals for all factors
            # used by rrt to calculated reward & lower policy termination, and lower policy to extract achieved goal
            target, next_target = self.policy.lower.get_target(self.data), self.policy.lower.get_target(self.data, next=True)
            target_diff = next_target - target

            # proximity: how close objects are to each other
            proximity = compute_proximity(target)

            # true_graph: the true graph connectivity
            # graph: the connectivity of the graph
            # graph may require the target/next_target/target_diff
            # used by rrt to calculated reward & lower policy termination, and lower policy to extract achieved goal
            self.data.true_graph = true_graph = self.data.info.factor_graph
            true_graph_count_idx = np.dot(true_graph, self.graph_to_count_idx)
            graph = self.dynamics(self.data)
            graph_count_idx = np.dot(graph, self.graph_to_count_idx)

            # hacky way to collect visited graphs for skill evaluation
            if hasattr(self, "collected_graphs"):
                num_env = len(true_graph)
                self.collected_graphs[self.num_collected_episodes:self.num_collected_episodes + num_env, self.num_steps] = graph
                self.num_steps += 1
                if np.any(done):
                    assert np.all(done)
                    self.num_collected_episodes += num_env
                    self.num_steps = 0

            self.data.update(
                time_upper=time_upper,
                time_lower=time_lower,
                target=target,
                next_target=next_target,
                target_diff=target_diff,
                proximity=proximity,
                true_graph_count_idx=true_graph_count_idx,
                graph=graph,
                graph_count_idx=graph_count_idx,
            )

            # achieved target goal: the actual goal reached at the current time step
            # achieved target goal computation uses (obs, graph, obs_next)
            # achieved target goal must be computed before the terminate and reward chains, but after graph
            self.data.obs.achieved_goal = achieved_goal = self.policy.lower.get_achieved_goal(self.data, use_next_obs=True)
            if "all_desired_goals" in self.data.obs:
                self.data.obs.all_achieved_goals = self.data.obs.achieved_goal
                self.data.obs.achieved_goal = achieved_goal = \
                    self.data.obs.achieved_goal[np.arange(len(self.data.obs.achieved_goal)), self.data.option_choice]
            

            # self.data.obs.desired_goal is computed in self.policy.forward(), with data.upper_obs as inputs
            # notice that desired_goal and achieved_goal for obs_next is not generated by the upper policy yet, and we use
            # desired_goal and achieved_goal for obs as a placeholder
            # TODO: these placeholders rather than true desired_goal and desired_goal will be stored into replay buffer,
            #  does it matter?
            # when computing target for value update,
            #   if desired_goal for obs_next is the same (upper policy doesn't sample a new high-level action), should be fine
            #   if desired_goal is different and lower policy is truncated, may have issues
            # when computing her reward, should be fine
            # TODO: implement all_achieved_goal/desired goal implementation in HER
            self.data.obs_next = construct_her_obs(achieved_goal=achieved_goal,
                                                   desired_goal=self.data.obs.desired_goal,
                                                   env_obs=obs_next,
                                                   reached_graph_counter=self.data.obs.reached_graph_counter,)
            if "all_desired_goals" in self.data.obs: 
                self.data.obs_next.all_desired_goals = self.data.obs.all_desired_goals

            # reward_chain: rewards for upper and lower policies
            # term_chain, trunc_chain: termination and truncation for upper and lower policies
            #   they already take env termination and truncation into consideration
            #   for now, upper policy never terminates nor truncates
            #   lower policy terminates when obs.achieved_goal reaches obs.desired_goal, and truncates when timeout
            reward_chain, term_chain, trunc_chain = self.policy.check_rew_term_trunc(self.data)

            self.data.update(
                reward_chain=reward_chain,
                term_chain=term_chain,
                trunc_chain=trunc_chain,
                rew=reward_chain.lower,
            )
            if not self.lower_training_ignore_lower_done:
                # replace terminated and truncated from env with those for lower policy
                self.data.update(
                    # TODO: in wider rew/terminated situations, this probably does not work and requires a "single_term_rew"
                    terminated=term_chain.lower,
                    truncated=trunc_chain.lower,
                )

            reached_graph, reached_goal = self.policy.lower.rewtermdone.check_lower_reached(self.data)
            self.data.lower_reached = Batch(reached_graph=reached_graph,
                                            reached_goal=reached_goal,
                                            updated=np.ones_like(reached_graph, dtype=bool))
            current_reached_graph_counter = self.data.obs.reached_graph_counter
            next_reached_graph_counter = current_reached_graph_counter + reached_graph / self.reached_graph_threshold
            self.data.obs_next.reached_graph_counter = np.clip(next_reached_graph_counter, 0, 1)

            if render and not save_gif:
                upper = self.policy.upper
                num_factors = upper.num_factors
                factor_names = upper.extractor.factor_names + ["act"]
                desired_goal = self.data.obs.desired_goal
                factor = desired_goal[0, :num_factors].argmax(axis=-1)
                parent = desired_goal[0, num_factors:upper.graph_size]
                parent = parent.reshape(num_factors + 1, upper.num_edge_classes).argmax(axis=-1)
                graph_name = ", ".join([factor_names[i] for i, p in enumerate(parent) if p]) + " -> " + factor_names[factor]
                print("graph_name", graph_name)
                print("reached_graph", reached_graph)
                print("reward", reward_chain.upper[0])
                img = self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # GIF EDIT
            if save_gif:
                env_render_cur_frames = self.env.render()
                for i in range(min(num_of_gifs, len(ready_env_ids))):
                    frames[i].append(np.moveaxis(env_render_cur_frames[i], 2, 0))
                    if done[i]:
                        # TODO: we can also stop when low_done
                        # TODO: only for printer; this is assuming that graph doesn't change
                        print("Saving gif...")
                        factor = self.data.obs.desired_goal[i][:15].reshape(5, 3).argmax(-1)
                        z = self.data.obs.desired_goal[i][15:].argmax(-1)
                        gif_name = f"ep:{self.epoch_num}_factor:{factor}_z:{z}_count:{episode_count}_{i}.gif"
                        write_gif(np.array(frames[i]),
                                  str(self.exp_path) + "/gifs/" + gif_name,
                                  fps=1 / pause)
                        frames[i] = []

            # add data into the buffer
            if self.policy.training:
                # deactivate upper samples whose lower values are overwritten
                upper_buffer_idx = self.buffer.get_upper_indices(buffer_ids=ready_env_ids)
                self.upper_buffer.deactivate(upper_buffer_idx)

            # add data into the lower buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids
            )
            if self.policy.training:
                # if some/all envs are reset, reset corresponding upper_buffer_memory
                self.upper_buffer_memory = reset_upper_memory(self.upper_buffer_memory, self.data, ptr)
                self.upper_buffer_memory = update_upper_memory_trajectory(self.upper_buffer_memory,
                                                                          self.upper_trajectories,
                                                                          self.data,
                                                                          ptr,
                                                                          ready_env_ids,
                                                                          self.upper_rew_aggregation)
                self.policy.update_state_counts(self.data)

            # logging
            self.update_logging_metrics(self.data, ready_env_ids)
            lower_done = term_chain.lower | trunc_chain.lower
            if np.any(lower_done):
                env_ind_local = np.where(lower_done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                lower_metrics = self.record_reset_lower_logging_metrics(lower_metrics, env_ind_local, env_ind_global)

            # ---------------------------------- major modification ends ---------------------------------- #

            # collect statistics
            step_count += len(ready_env_ids)
            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])

                # use and update reset_invalid_ids properly
                upper_buffer_add(self.upper_trajectories, self.upper_buffer, self.buffer, env_ind_global)
                # logging
                epi_metrics, lower_metrics = self.record_reset_all_logging_metrics(epi_metrics, lower_metrics,
                                                                                   env_ind_local, env_ind_global)

                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                # reset_env_with_ids also sets invalid env ids properly
                self._reset_env_with_ids(
                    env_ind_local, env_ind_global, gym_reset_kwargs
                )
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

                        # the first (surplus_env_num) environments are removed,
                        # so no need to keep an eye on whether they are properly reset
                        if self.upper_buffer_memory is not None:
                            self.upper_buffer_memory = self.upper_buffer_memory[mask]

            self.data.obs = self.data.obs_next
            self.data.upper_obs = self.data.upper_obs_next

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        if self.policy.training:
            self.policy.upper.update_history(self.buffer)

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={}
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = list(
                map(
                    np.concatenate,
                    [episode_rews, episode_lens, episode_start_indices]
                )
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        result = {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
        }

        # extra logging, default value np.nan will not be recorded by the logger
        result.update({f"episode/{k}": np.mean(v) if len(v) else np.nan for k, v in epi_metrics.items()})
        result.update({f"lower/{k}": np.mean(v) if len(v) else np.nan for k, v in lower_metrics.items()})

        return result
