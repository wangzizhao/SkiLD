import numpy as np

from tianshou.data import Batch, to_numpy
from Option.Terminate import RewardTerminateTruncate
from collections import deque

DONT_CARE_EDGE_TYPE = 2


class RTTLowerGraph(RewardTerminateTruncate):
    """
    returns -1 reward unless a graph is reached, then returns term=True and 0 reward
    timeouts according to lower policy time
    """

    def __init__(self, **kwargs):
        config = kwargs["config"]

        super().__init__(timeout=config.option.timeout)

        self.extractor = kwargs["extractor"]
        self.diayn_discriminator = kwargs["diayn_discriminator"]
        if self.diayn_discriminator is not None:
            self.diayn_discriminator.reached_graph = self.reached_graph
            self.diayn_discriminator.get_achieved_goal = self.get_achieved_goal

        self.device = config.device
        self.num_factors = num_factors = config.num_factors

        self.goal_epsilon = config.option.target_goal_epsilon
        self.goal_scale = config.option.lower.goal_scale

        lower_rew_config = config.option.lower

        # graph reaching reward
        self.use_reached_graph_counter = lower_rew_config.use_reached_graph_counter
        self.reached_graph_threshold = lower_rew_config.reached_graph_threshold
        self.graph_reward_scale = lower_rew_config.graph_reward_scale

        # diayn reward
        self.diayn_graph_conditioned = lower_rew_config.diayn_graph_conditioned
        self.diayn_scale = lower_rew_config.diayn_scale

        # count-based reward
        self.use_count_reward = lower_rew_config.use_count_reward
        self.count_reward_scale = lower_rew_config.count_reward_scale
        self.state_count = {}

        self.goal_type = config.policy.upper.goal_type
        self.graph_type = config.policy.upper.graph_type
        self.add_count_based_lower = config.policy.upper.add_count_based_lower
        self.num_edge_classes = num_edge_classes = config.graph_encoding.num_edge_classes

        if self.graph_type == "graph":
            self.graph_info_size = num_factors * (num_factors + 1) * num_edge_classes
        elif self.graph_type == "factor":
            self.graph_info_size = num_factors + (num_factors + 1) * num_edge_classes
        elif self.graph_type == "none":
            self.graph_info_size = 0
            assert self.graph_reward_scale == 0
            assert not self.diayn_graph_conditioned
        else:
            raise NotImplementedError(f"unknown graph type: {self.graph_type}")

        self.cache = None

        self.init_rew_schedule_and_adaption(lower_rew_config)

    def init_rew_schedule_and_adaption(self, lower_rew_config):
        # scheduled reward parameters
        self.reached_graph_negative_constant = lower_rew_config.reached_graph_negative_constant
        self.schedule_rew_scale_min = lower_rew_config.schedule.schedule_rew_scale_min
        assert self.schedule_rew_scale_min >= 0

        self.reached_graph_schedule = lower_rew_config.schedule.reached_graph_schedule
        self.state_count_schedule = lower_rew_config.schedule.state_count_schedule
        self.update_count_schedule = lower_rew_config.schedule.update_count_schedule
        assert self.reached_graph_schedule >= 0 and self.state_count_schedule >= 0 and self.update_count_schedule >= 0
        assert (self.reached_graph_schedule > 0) + (self.state_count_schedule > 0) + (self.update_count_schedule > 0) <= 1

        self.needs_state_counts = self.use_count_reward or (self.state_count_schedule > 0) or self.add_count_based_lower

        # adaptive reward parameters
        # diayn_adaptive_rew_scale = max(graph_reaching_percentage - adaptive_diayn_coef, 0) / (1 - adaptive_diayn_coef)
        self.adaptive_diayn_coef = lower_rew_config.adaptive.adaptive_diayn_coef
        assert 0 <= self.adaptive_diayn_coef < 1 or self.adaptive_diayn_coef == -1

        # max length of deque to keep track of graph reaching stats
        self.history_stats_size = lower_rew_config.adaptive.history_stats_size

        # stats trackers
        self.upper_unique_graph_from_id = None
        self.upper_choose_from_history_action_mask = None
        self.reset_rew_schedule_and_adaption()

    def reset_rew_schedule_and_adaption(self):
        self.reached_graph_stats = {}                       # {graph: deque of reached_graph}
        self.diayn_accuracy_stats = {}                      # {graph: deque of diayn_accuracy}

        # init rew scales
        self.adaptive_rew_scale = {}                        # {graph: float}
        self.schedule_rew_scale = 1                         # TODO: change to graph-specific schedule if needed, {graph: float}

        # Caleb's ones
        self.total_sample_graph_counter = 0
        self.total_reached_graph_counter = 0
        self.state_count_total = 0

    def update_cache(self, batch, reached_graph, reached_goal):
        self.cache = Batch(desired_goal=batch.obs.desired_goal,
                           achieved_goal=batch.obs.achieved_goal,
                           reached_graph=reached_graph,
                           reached_goal=reached_goal)

    def matches_cache(self, batch):
        return (np.all(self.cache.desired_goal == batch.obs.desired_goal) and
                np.all(self.cache.achieved_goal == batch.obs.achieved_goal))

    def get_graph(self, graph_goal_array):
        graph = graph_goal_array[..., :self.graph_info_size]

        bs = graph.shape[:-1]
        num_factors, num_edge_classes = self.num_factors, self.num_edge_classes

        if self.graph_type == "graph":
            # desired_graph: (bs, num_factors, num_factors + 1)
            graph = graph.reshape(*bs, num_factors, num_factors + 1, num_edge_classes).argmax(axis=-1)
        elif self.graph_type == "factor":
            factor, parents = graph[..., :num_factors], graph[..., num_factors:]
            parents = parents.reshape(*bs, num_factors + 1, num_edge_classes).argmax(axis=-1)
            graph = (factor, parents)
        elif self.graph_type == "none":
            graph = None
        else:
            raise NotImplementedError(f"unknown graph type: {self.graph_type}")

        return graph

    def reached_graph(self, batch, return_true_for_state_coverage_policy=True):
        desired_graph = self.get_graph(batch.obs.desired_goal)
        achieved_graph = self.get_graph(batch.obs.achieved_goal)

        if self.graph_type == "graph":
            # desired_graph: (bs, num_factors, num_factors + 1), achieved_graph: (bs, num_factors, num_factors + 1)
            edge_same = desired_graph == achieved_graph
            edge_dont_care = desired_graph == DONT_CARE_EDGE_TYPE
            reached_graph = (edge_same | edge_dont_care).all(axis=(-2, -1))
        elif self.graph_type == "factor":
            desired_factor, desired_parents = desired_graph
            achieved_factor, achieved_parents = achieved_graph
            assert np.all(desired_factor == achieved_factor)

            # desired_parents, achieved_parents: (bs, num_factors + 1)
            parents_same = desired_parents == achieved_parents
            parents_dont_care = desired_parents == DONT_CARE_EDGE_TYPE
            reached_graph = (parents_same | parents_dont_care).all(axis=-1)         # (bs,)
            if return_true_for_state_coverage_policy:
                is_state_coverage_policy = np.all(desired_parents == 0, axis=-1)
                reached_graph[is_state_coverage_policy] = True
        elif self.graph_type == "none":
            reached_graph = np.ones(len(batch.obs), dtype=bool)
        else:
            raise NotImplementedError(f"unknown graph type: {self.graph_type}")

        return reached_graph, desired_graph

    def get_achieved_goal(self, batch, desired_graph, use_next_obs):
        # get achieved goal
        if self.graph_type == "factor":
            factor, _ = desired_graph
            factored_state = batch.get("next_target" if use_next_obs else "target",
                                       self.extractor.slice_targets(batch, next=use_next_obs))  # (bs, num_factors, longest)
            achieved_goal = factored_state[np.nonzero(factor)]                                  # (bs, longest)
        elif self.graph_type == "graph" or self.graph_type == "none":
            achieved_goal = batch.obs_next.observation if use_next_obs else batch.obs.observation
        else:
            raise NotImplementedError(f"unknown graph type: {self.graph_type}")
        return achieved_goal

    def update_state_count(self, state):
        assert state.ndim == 2
        states, counts = np.unique(state, axis=0, return_counts=True)
        for s, c in zip(states, counts):
            s = s.tobytes()               # make np.ndarray hashable, so it's a valid key for dict
            self.state_count[s] = self.state_count.get(s, 0) + c

    def compute_state_count_rew(self, state):
        assert state.ndim == 2
        count = []
        for s in state:
            s = s.tobytes()
            count.append(self.state_count.get(s, 1.))
        return 1 / np.sqrt(count)

    def _goal_rew(self, batch, desired_graph, reached_graph):
        """
        batch.obs: Batch consisting of
            observation (bs, obs_size)
            desired_goal: concat of [graph_info, goal]
                graph_info: (bs, graph_info_size)
                goal: (bs, factor_longest / diayn.num_classes)
        desired_graph:
            if self.graph_type == "factor":
                desired_graph = (factor, parents)
                factor: (bs, )
                parents: (bs, num_factors + 1)
            elif self.graph_type == "graph":
                (bs, num_factors, num_factors + 1)
        reached_graph: bool, (bs, )
        """
        desired_goal = batch.obs.desired_goal[..., self.graph_info_size:]
        desired_graph_onehot = batch.obs.desired_goal[..., :self.graph_info_size]

        if self.goal_type == "value":
            assert self.graph_type == "factor"
            achieved_next_state = batch.obs.achieved_goal[..., self.graph_info_size:]
            reached_goal = np.linalg.norm(achieved_next_state - desired_goal, axis=-1) < self.goal_epsilon
            rew = reached_goal * self.goal_scale
        elif self.goal_type == "diayn":
            achieved_state = self.get_achieved_goal(batch, desired_graph, use_next_obs=False)
            achieved_next_state = self.get_achieved_goal(batch, desired_graph, use_next_obs=True)
            rew = self.diayn_discriminator.get_intrinsic_reward(desired_graph_onehot,
                                                                achieved_state,
                                                                achieved_next_state,
                                                                desired_goal)
            # scale reach goal by rates
            adaptive_rew_scale = self.get_rew_adaptive_scale(batch.obs.desired_goal)
            rew = rew * self.diayn_scale * self.schedule_rew_scale * adaptive_rew_scale

            # diayn always reach goal
            reached_goal = np.ones_like(reached_graph, dtype=bool)
        else:
            raise NotImplementedError(f"unknown goal type: {self.goal_type}")

        if self.use_count_reward:
            # factor + parent + factor value as key
            graph_cond_next_state = np.concatenate([desired_graph_onehot, achieved_next_state], axis=-1)
            count_rew = self.compute_state_count_rew(graph_cond_next_state)

            # augment diayn reward with state coverage reward
            rew = rew + count_rew * self.count_reward_scale

            # update state counts
            if self.training and not self.updating:
                self.update_state_count(graph_cond_next_state)

        if self.add_count_based_lower:
            assert self.graph_type == "factor"

            # factor + factor value as key
            desired_factor = desired_graph_onehot[..., :self.num_factors]
            factor_cond_next_state = np.concatenate([desired_factor, achieved_next_state], axis=-1)
            count_rew = self.compute_state_count_rew(factor_cond_next_state)

            # for state coverage lower, only use count reward
            desired_parents = desired_graph[1]
            is_state_coverage = np.all(desired_parents == 0, axis=-1)
            rew[is_state_coverage] = count_rew[is_state_coverage]

            # update state counts
            if self.training and not self.updating:
                self.update_state_count(factor_cond_next_state)

        return reached_graph & reached_goal, rew

    def rew(self, batch):
        reached_graph, desired_graph = self.reached_graph(batch)
        if self.use_reached_graph_counter:
            reached_graph_count = batch.obs_next.reached_graph_counter * self.timeout
            give_reach_graph_rew = reached_graph_count >= self.reached_graph_threshold | reached_graph
        else:
            give_reach_graph_rew = reached_graph
        reach_graph_rew = give_reach_graph_rew + self.reached_graph_negative_constant           # (bs,)

        reached_goal, reach_goal_rew = self._goal_rew(batch, desired_graph, reached_graph)      # (bs,)
        if self.goal_type == "diayn":
            if self.diayn_graph_conditioned:
                reach_goal_rew = give_reach_graph_rew * reach_goal_rew
        else:
            reach_goal_rew = give_reach_graph_rew * reach_goal_rew

        self.update_cache(batch, reached_graph, reached_goal)
        return reach_graph_rew * self.graph_reward_scale + reach_goal_rew

    def term(self, batch):
        if self.goal_type == "value":
            _, reached_goal = self.lower_reached(batch)
            return reached_goal
        else:
            return np.zeros_like(batch.time_lower, dtype=bool)

    def lower_reached(self, batch):
        if self.matches_cache(batch):
            return self.cache.reached_graph, self.cache.reached_goal

        reached_graph, desired_graph = self.reached_graph(batch)
        reached_goal, _ = self._goal_rew(batch, desired_graph, reached_graph)                   # (bs,)
        return reached_graph, reached_goal

    def trunc(self, batch):
        return batch.time_lower >= self.timeout

    def update_state_counts(self, batch):
        self.state_count_total += len(batch.obs)

    def update_lower_stats(self, lower_metrics, desired_goal):
        for goal, reach_goal, reach_graph, lower_updated in zip(desired_goal,
                                                                lower_metrics.reached_goal,
                                                                lower_metrics.reached_graph,
                                                                lower_metrics.updated):
            if not lower_updated:
                # lower metrics has not been updated, reach_graph and reach_goal are not valid
                continue

            graph = goal[:self.graph_info_size]
            graph_key = graph.tobytes()
            if graph_key not in self.reached_graph_stats:
                self.reached_graph_stats[graph_key] = [graph, deque(maxlen=self.history_stats_size)]
            self.reached_graph_stats[graph_key][1].append(reach_graph)

    def get_rew_adaptive_scale(self, desired_goal):
        if self.adaptive_diayn_coef >= 0:
            adaptive_rew_scale = np.zeros(len(desired_goal))
            for i, goal in enumerate(desired_goal):
                graph = goal[:self.graph_info_size].tobytes()
                if graph in self.adaptive_rew_scale:
                    adaptive_rew_scale[i] = self.adaptive_rew_scale[graph]
            return adaptive_rew_scale
        else:
            return 1

    def update_schedules(self):
        '''
        updates the reward scales for diayn and graph reaching, this should be called somewhat
        infrequently, since changing the rewards essentially changes the learning target, and
        for adaptive methods, this can have unpredictable consequences
        '''
        super().update_schedules()
        # reward scaling strategies:
        # adaptive rewards (adjust DIAYN reward based on):
        #             graph success rate / the graph counter
        #             target success rate / diayn reward value

        # self.adaptive_rew_scale = 1.0
        # if self.adaptive_scale_factor > 0:
        #     self.total_reached_graph_sample = np.sum(self.total_reached_graph_deque)
        #     self.total_diayn_accuracy = np.sum(self.total_diayn_accuracy_deque)
        #     total_sample_graph = min(self.history_stats_size, self.total_sample_graph_counter)
        #     if self.adaptive_reaching:
        #         self.adaptive_rew_scale = 1 - np.exp(-self.total_reached_graph_sample / total_sample_graph
        #                                               * self.adaptive_scale_factor)
        #     elif self.adaptive_diayn:
        #         total_diayn_prediction = min(self.state_count_total, self.history_stats_size)
        #         self.adaptive_rew_scale = 1 - np.exp((self.total_diayn_accuracy / total_diayn_prediction -
        #                                                self.total_reached_graph_counter / total_sample_graph)
        #                                               * self.adaptive_scale_factor)
        if self.adaptive_diayn_coef >= 0:
            upper_unique_graph_from_id = self.upper_unique_graph_from_id
            upper_choose_from_history_action_mask = to_numpy(self.upper_choose_from_history_action_mask).astype(bool)
            valid_graphs = upper_unique_graph_from_id[upper_choose_from_history_action_mask]
            for graph_key, (graph, reached_graph_deque) in self.reached_graph_stats.items():
                if np.any(np.all(valid_graphs == graph, axis=-1)):
                    reached_graph_percentage = np.mean(reached_graph_deque)
                    self.adaptive_rew_scale[graph_key] = \
                        max(reached_graph_percentage - self.adaptive_diayn_coef, 0) / (1 - self.adaptive_diayn_coef)

        self.schedule_rew_scale = 1.0
        if self.reached_graph_schedule > 0:
            self.schedule_rew_scale = 1 - np.exp(-self.total_reached_graph_counter / self.reached_graph_schedule)
        elif self.state_count_schedule > 0:
            self.schedule_rew_scale = 1 - np.exp(-self.state_count_total / self.state_count_schedule)
        elif self.update_count_schedule > 0:
            self.schedule_rew_scale = 1 - np.exp(-self.num_updates / self.update_count_schedule)

        self.schedule_rew_scale = max(self.schedule_rew_scale, self.schedule_rew_scale_min)
