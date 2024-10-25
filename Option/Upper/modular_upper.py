from typing import Any, Dict, Union, Optional

import itertools

import torch
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from tianshou.data import Batch, to_numpy
from tianshou.policy import BasePolicy

from Causal import Dynamics
from Networks import GraphEncoding
from State.extractor import Extractor
from State.buffer import HierarchicalReplayBuffer, VectorHierarchicalReplayBufferManager
from Option.Upper import UpperPolicy
from Option.Terminate import RewardTermTruncManager
from Option.Terminate import RTTUpperGraphCount

EPS = 1e-2
SOFTMAX_TEMP = 0.2


class ModularUpper(UpperPolicy):
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
        """
        policy: a tianshou policy based on the desired algorithm
        rewtermdone: reward terminate and done signals, usually based on the true signal for the upper policy
        lower_check_termination: termination for the lower determines if the upper needs a new sample
        config: algorithm specific arguments
        """
        super().__init__(policy, dynamics, rewtermdone, config)
        self.extractor = extractor
        self.random_eps = config.policy.upper.random_eps

        # env info
        self.num_factors = config.num_factors
        self.num_edge_classes = config.graph_encoding.num_edge_classes
        self.factor_longest = extractor.longest
        self.breakpoints = extractor.breakpoints
        self.factor_spaces = config.factor_spaces

        self.graph_type = config.policy.upper.graph_type
        self.goal_type = config.policy.upper.goal_type
        self.lower_type = config.policy.lower.type

        # diayn goal
        self.num_diayn_classes = config.policy.upper.diayn.num_classes

        # helper to convert index to onehot
        self.factor_onehot_helper = np.eye(self.num_factors, dtype=np.float32)
        self.edge_onehot_helper = np.eye(self.num_edge_classes, dtype=np.float32)
        self.diyan_onehot_helper = np.eye(self.num_diayn_classes, dtype=np.float32)
        self.count_idx_to_graph = np.flip(list(itertools.product([0, 1], repeat=self.num_factors + 1)), axis=-1).astype(bool)

        if self.graph_type == "graph":
            self.graph_size = self.num_factors * (self.num_factors + 1) * self.num_edge_classes
        elif self.graph_type == "factor":
            self.graph_size = self.num_factors + (self.num_factors + 1) * self.num_edge_classes
        else:
            raise NotImplementedError

        if self.goal_type == "value":
            if self.graph_type == "graph":
                self.goal_size = config.obs_size
            elif self.graph_type == "factor":
                self.goal_size = self.factor_longest
            else:
                raise NotImplementedError
        elif self.goal_type == "diayn":
            self.goal_size = config.policy.upper.diayn.num_classes
        else:
            raise NotImplementedError

        self.graph_action_space = config.policy.upper.graph_action_space
        self.add_count_based_lower = config.policy.upper.add_count_based_lower
        self.sample_action_space_n = config.policy.upper.sample_action_space_n
        self.is_task_learning = config.train.mode == "task_learning_lower_frozen"
        self.goal_learned = config.policy.upper.goal_learned  # or self.is_task_learning

        # learned graph
        if self.graph_action_space == "sample_from_history":
            self.graph_action_size = 0
        elif self.graph_action_space == "choose_from_history":
            self.graph_action_size = 1
        else:
            raise NotImplementedError(f"unknown graph action space type: {self.graph_action_space}")

        # check and overwrite some config given the graph and goal type
        if self.graph_action_space == "choose_from_history":
            # avoid randomly sampling graphs out of support from history
            self.random_eps = 0

        if self.is_task_learning:
            assert self.graph_action_space == "choose_from_history"
            assert self.goal_type == "diayn"

        # historical graph
        self.history_update_ready = False

        # for choose_from_history action space
        self.unique_graph_from_hash = {}
        self.unique_graph_from_id = np.zeros((self.sample_action_space_n, self.graph_size), dtype=np.float32)
        self.unique_graph_to_id = {}                # when using randomly sampled graph, need to map graph to id
        self.unique_graph_index = 0
        self.unique_factor_graph_from_hash = None   # if not none, will save a dictionary of unique graphs for every factor
        self.choose_from_history_action_mask = torch.zeros(self.sample_action_space_n, dtype=torch.float32, device=config.device)
        self.choose_from_history_action_mask[0] = 1

        if self.graph_action_space == "choose_from_history" and self.add_count_based_lower:
            if self.graph_type == "graph":
                raise NotImplementedError
            # pre-fill the history graph with graphs reserved for state coverage lower policies
            # the rtt function will only use state count reward for those lower policies
            # 0th policy maximizes 0th factor value count, 1st policy maximizes 1st factor value count, etc.
            for i in range(self.num_factors + 1):
                parents = np.zeros(self.num_factors + 1, dtype=int)
                parents_onehot = self.edge_onehot_helper[parents].flatten()
                self.unique_graph_from_id[i, i] = 1
                self.unique_graph_from_id[i, self.num_factors:] = parents_onehot

            self.unique_graph_index = self.num_factors
            self.choose_from_history_action_mask[:self.num_factors] = not self.is_task_learning

        # keep at least one element in the mask to avoid NaN error
        if not torch.any(self.choose_from_history_action_mask):
            self.choose_from_history_action_mask[0] = 1

        # for adjust historical graph sample frequency
        self.lower_success_tracker = {}

        # action stats, graph hash -> (graph, selection_count, reach_graph_count, reach_goal_count)
        self.action_stats = {}

        # history goal
        self.achieved_goal_tracker = {}
        self.history_goal = list()

    def update_history(self, lower_buffer: VectorHierarchicalReplayBufferManager):
        if not self.history_update_ready or len(lower_buffer) == 0 or self.is_task_learning:
            return

        if self.graph_type != "factor":
            raise NotImplementedError

        self.choose_from_history_action_mask[:] = 0
        graph_filter = np.eye(self.num_factors, self.num_factors + 1)
        for factor in range(self.num_factors):
            graph_filter_i = graph_filter[factor]
            factor_onehot = self.factor_onehot_helper[factor]           # (num_factors,)

            for factor_count_idx in lower_buffer.valid_graph_indices[factor]:
                graph = self.count_idx_to_graph[factor_count_idx]       # (num_factors + 1,)

                # filter out trivial graphs (no parents or the object itself as the only parent)
                if not np.any(graph > graph_filter_i):
                    continue

                graph = graph.astype(int)
                parent = self.edge_onehot_helper[graph].flatten()       # ((num_factors + 1) * num_edge_classes,)
                graph = np.concatenate([factor_onehot, parent])         # (num_factors + (num_factors + 1) * num_edge_classes,)

                graph_key = graph.astype(int).tobytes()
                graph = graph.astype(np.float32)

                # add new graph to history
                if graph_key not in self.unique_graph_from_hash:
                    if self.unique_factor_graph_from_hash is not None:
                        self.unique_factor_graph_from_hash[factor][graph_key] = graph
                    self.unique_graph_from_hash[graph_key] = graph

                    if self.graph_action_space == "choose_from_history" and self.unique_graph_index >= self.sample_action_space_n:
                        raise ValueError("sample_action_space_n is too small")

                    self.unique_graph_from_id[self.unique_graph_index] = graph
                    self.choose_from_history_action_mask[self.unique_graph_index] = 1
                    self.unique_graph_to_id[graph_key] = self.unique_graph_index
                    self.unique_graph_index = self.unique_graph_index + 1

                    if graph_key not in self.lower_success_tracker:
                        self.lower_success_tracker[graph_key] = 0
                else:
                    graph_id = self.unique_graph_to_id[graph_key]
                    self.choose_from_history_action_mask[graph_id] = 1

        if self.graph_action_space == "choose_from_history" and self.add_count_based_lower:
            # always keep state count maximization policy available
            self.choose_from_history_action_mask[:self.num_factors] = not self.is_task_learning

        # keep at least one element in the mask to avoid NaN error
        if not torch.any(self.choose_from_history_action_mask):
            self.choose_from_history_action_mask[0] = 1

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any,
    ) -> Batch:
        if self.graph_action_space == "choose_from_history":
            mask = self.choose_from_history_action_mask
            # if self.is_task_learning and self.goal_type == "diayn":
            #     mask = mask.repeat(self.num_diayn_classes)
            kwargs["mask"] = mask
        return super().forward(batch, state, **kwargs)

    def update(self, sample_size: int,
               buffer: HierarchicalReplayBuffer,
               lower_buffer: VectorHierarchicalReplayBufferManager,
               **kwargs) -> Dict[str, Any]:
        if self.graph_action_space == "choose_from_history":
            mask = self.choose_from_history_action_mask
            # if self.is_task_learning and self.goal_type == "diayn":
            #     mask = mask.repeat(self.num_diayn_classes)
            kwargs["mask"] = mask
        return super().update(sample_size, buffer, lower_buffer, **kwargs)

    def update_lower_stats(self, lower_metrics, desired_goal):
        for goal, reach_goal, reach_graph, lower_updated in zip(desired_goal,
                                                                lower_metrics.reached_goal,
                                                                lower_metrics.reached_graph,
                                                                lower_metrics.updated):
            if not lower_updated:
                # lower metrics has not been updated, reach_graph and reach_goal are not valid
                continue

            graph = goal[:self.graph_size].astype(int)
            graph_hash = graph.tobytes()
            if graph_hash not in self.lower_success_tracker:
                self.lower_success_tracker[graph_hash] = 0
            self.lower_success_tracker[graph_hash] += int(reach_goal)

            if graph_hash not in self.action_stats:
                self.action_stats[graph_hash] = [graph, 0., 0., 0.]
            self.action_stats[graph_hash][1] += 1.
            self.action_stats[graph_hash][2] += float(reach_goal)
            self.action_stats[graph_hash][3] += float(reach_graph)

    def sample_action(self, policy: Batch, random: bool = False):
        act = policy.act
        bs = act.shape if self.graph_action_space == "choose_from_history" else act.shape[:-1]
        assert len(bs) == 1, "only support 1-d batch size for now"

        if self.policy is not None:
            act = self.policy.map_action(act)

        if self.graph_action_space == "choose_from_history" and self.unique_graph_index == 0:
            # no history graph, sample a random graph instead
            random = True

        # get desired graph
        if self.graph_action_space == "sample_from_history":
            graph = self.sample_history_graph(bs)  # TODO: modify when getting back a wide upper sampled graph
        else:
            if random:
                graph = self.sample_history_graph(bs)
                if self.graph_action_space == "choose_from_history":
                    act = np.array([self.unique_graph_to_id.get(g.tobytes(), 0)
                                    for g in graph])
                else:
                    raise NotImplementedError(f"unknown graph action space: {self.graph_action_space}")
            elif self.graph_action_space == "choose_from_history":
                # during task learning, the action space is selected_diayn_id * sample_action_space_n + graph_id
                graph_id = act % self.sample_action_space_n
                graph = self.unique_graph_from_id[graph_id]
            else:
                raise NotImplementedError(f"unknown graph action space: {self.graph_action_space}")

        # get desired goal
        if random or not self.goal_learned:
            if self.goal_type == "value":
                goal = self.sample_history_goal(graph)
            elif self.goal_type == "diayn":
                goal = self.sample_diayn_z(bs)
                # if self.is_task_learning and self.graph_action_space == "choose_from_history":
                #     act += np.argmax(goal, axis=-1) * self.sample_action_space_n
            else:
                raise NotImplementedError(f"unknown goal type: {self.goal_type}")
        else:
            if self.goal_type == "value":
                goal_act = act[..., self.graph_action_size:]
                goal = self.sample_learned_goal(goal_act, graph)
            elif self.goal_type == "diayn":
                goal = act // self.sample_action_space_n
                goal = self.diyan_onehot_helper[goal]
            else:
                raise NotImplementedError(f"unknown goal type: {self.goal_type}")

        if self.policy is not None and random:
            if isinstance(self.policy.action_space, Box):
                act = np.clip(act, self.policy.action_space.low, self.policy.action_space.high)
                policy.act = self.policy.map_action_inverse(act)
            elif isinstance(self.policy.action_space, Discrete):
                policy.act = act
        policy.sampled_act = np.concatenate([graph, goal], axis=-1)

        # get option_choice
        if self.graph_type == "graph":
            policy.option_choice = np.zeros(*bs, dtype=int)
        elif self.graph_type == "factor":
            if self.lower_type == "single_graph":
                policy.option_choice = np.zeros(*bs, dtype=int)
            elif self.lower_type == "wide":
                factor = graph[..., :self.num_factors].argmax(axis=-1)
                policy.option_choice = factor
                if self.add_count_based_lower:
                    # using all zero factor_onehot to represent using the state count maximization policy,
                    # which is the last policy in the wide lower policy
                    parents_onehot = graph[..., self.num_factors:].reshape(*bs, self.num_factors + 1, self.num_edge_classes)
                    parents = parents_onehot.argmax(axis=-1)
                    no_parents = np.all(parents == 0, axis=-1)
                    policy.option_choice[no_parents] = self.num_factors
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(f"unknown graph encoding type: {self.graph_type}")

        return policy

    def sample_single_history_graph(self, bs, graph_from_hash, force_factor=-1):
        if len(graph_from_hash) and np.random.rand() >= self.random_eps:
            graphs = list(graph_from_hash)
            prob_unnormal = np.array([1 / np.sqrt(self.lower_success_tracker[g] + 1) for g in graphs])
            prob = prob_unnormal / np.sum(prob_unnormal)
            graph_count = np.random.multinomial(np.prod(bs), prob)

            sampled_graphs = np.zeros((*bs, self.graph_size), dtype=np.float32)
            for g, c, cum in zip(graphs, graph_count, np.cumsum(graph_count)):
                if c > 0:
                    sampled_graphs[cum - c:cum] = graph_from_hash[g]
            np.random.shuffle(sampled_graphs)
        else: # generates a random sample for that factor
            if self.graph_type == "graph": # keeping the graph sample type for now, in case there are alternative uses for wide upper
                sampled_graphs = np.random.randint(2, size=(*bs, self.num_factors, self.num_factors + 1))
                sampled_graphs = self.edge_onehot_helper[sampled_graphs].reshape(*bs, -1)
            elif self.graph_type == "factor":
                factor = np.random.randint(self.num_factors, size=bs) if force_factor == -1 else (np.ones((bs)) * force_factor).astype(int)
                factor_onehot = self.factor_onehot_helper[factor]
                parents = np.random.randint(2, size=(*bs, self.num_factors + 1))
                parents_onehot = self.edge_onehot_helper[parents].reshape(*bs, -1)
                sampled_graphs = np.concatenate([factor_onehot, parents_onehot], axis=-1)
            else:
                raise NotImplementedError
        return sampled_graphs

    def sample_history_graph(self, bs):
        # bs: tuple
        assert len(bs) == 1
        return self.sample_single_history_graph(bs, self.unique_graph_from_hash)

    def sample_learned_goal(self, act, graph):
        if self.graph_type == "factor":
            goal_logits = act.reshape(-1, self.factor_longest)
            goal_logits = torch.as_tensor(goal_logits, dtype=torch.get_default_dtype())
            factor = graph[..., :self.num_factors].argmax(axis=-1).flatten()

            goal = torch.zeros_like(goal_logits)
            for i, f in enumerate(factor):
                factor_space = self.factor_spaces[f]
                factor_length = self.breakpoints[f + 1] - self.breakpoints[f]

                if isinstance(factor_space, MultiDiscrete):
                    split_size = list(factor_space.nvec) + [self.factor_longest - factor_length]
                    variable_goal_logits = goal_logits[i].split(split_size)[:-1]  # -1 is the padding tensor
                    variable_goals = []
                    for logits in variable_goal_logits:
                        if self.training:
                            var_goal = OneHotCategorical(logits=logits / SOFTMAX_TEMP).sample()
                        else:
                            var_goal = F.one_hot(logits.argmax(), len(logits)).float()
                        variable_goals.append(var_goal)
                    variable_goals = torch.cat(variable_goals)
                    goal[i, :factor_length] = variable_goals
                elif isinstance(factor_space, Box):
                    goal[i, :factor_length] = goal_logits[i, :factor_length]
                else:
                    raise NotImplementedError

            goal = goal.reshape(*graph.shape[:-1], self.factor_longest)
            goal = to_numpy(goal)
        else:
            raise NotImplementedError
        return goal

    def recover_goal_act(self, graph, goal):
        if self.graph_type == "factor":
            goal = goal.reshape(-1, self.factor_longest)
            goal = torch.as_tensor(goal, dtype=torch.get_default_dtype())
            factor = graph[..., :self.num_factors].argmax(axis=-1).flatten()

            goal_act = torch.zeros_like(goal)
            for i, f in enumerate(factor):
                factor_space = self.factor_spaces[f]
                factor_length = self.breakpoints[f + 1] - self.breakpoints[f]

                if isinstance(factor_space, MultiDiscrete):
                    split_size = list(factor_space.nvec) + [self.factor_longest - factor_length]
                    variable_goals = goal[i].split(split_size)[:-1]  # -1 is the padding tensor
                    variable_goal_act = []
                    for vg in variable_goals:
                        vg = vg + EPS
                        vg_act = torch.log(vg / vg.sum()) * SOFTMAX_TEMP
                        # inverse of softmax is not unique, minus mean to make it as between -1 and 1 as possible
                        vg_act = vg_act - vg_act.mean()
                        variable_goal_act.append(vg_act)
                    variable_goal_act = torch.cat(variable_goal_act)
                    goal_act[i, :factor_length] = variable_goal_act
                elif isinstance(factor_space, Box):
                    goal_act[i, :factor_length] = goal[i, :factor_length]
                else:
                    raise NotImplementedError
            goal_act = goal_act.reshape(*graph.shape[:-1], self.factor_longest)
            goal_act = to_numpy(goal_act)
        else:
            raise NotImplementedError

        return goal_act

    def sample_history_goal(self, graph):
        # for single graph modules, uses the achieved goal tracker
        return self.sample_history_goal_single(graph, self.achieved_goal_tracker)

    def sample_history_goal_single(self, graph, achieved_goal_tracker):
        assert graph.ndim == 2
        goal = np.zeros((len(graph), self.goal_size), dtype=np.float32)
        for i, g in enumerate(graph):
            k = g.tobytes()
            if k in achieved_goal_tracker and np.random.rand() >= self.random_eps:
                idx = np.random.choice(achieved_goal_tracker[k])
                goal[i] = self.history_goal[idx]
            else:
                if self.graph_type == "factor":
                    factor = g[:self.num_factors].argmax(axis=-1)
                    factor_space = self.factor_spaces[factor]
                    factor_goal = factor_space.sample()
                    if isinstance(factor_space, Box):
                        goal[i, :len(factor_goal)] = factor_goal
                    elif isinstance(factor_space, MultiDiscrete):
                        for v_goal, v_len, v_end in zip(factor_goal, factor_space.nvec, np.cumsum(factor_space.nvec)):
                            # v_goal: int, index of the discrete variable goal
                            v_start = v_end - v_len
                            goal[i, v_start + v_goal] = 1
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

        return goal

    def sample_diayn_z(self, bs):
        diayn_z = np.random.randint(low=0, high=self.num_diayn_classes, size=bs)
        diayn_z_onehot = self.diyan_onehot_helper[diayn_z]
        return diayn_z_onehot

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(self, *args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict.update({prefix + k: v for k, v in
                           {"unique_graph_from_hash": self.unique_graph_from_hash,
                            "unique_graph_from_id": self.unique_graph_from_id,
                            "unique_graph_to_id": self.unique_graph_to_id,
                            "unique_graph_index": self.unique_graph_index,
                            "choose_from_history_action_mask": self.choose_from_history_action_mask,
                            "lower_success_tracker": self.lower_success_tracker,
                            "achieved_goal_tracker": self.achieved_goal_tracker,
                            "history_goal": self.history_goal}.items()
                           })
        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for attr_name in ["unique_graph_from_hash",
                          "unique_graph_from_id",
                          "unique_graph_to_id",
                          "unique_graph_index",
                          "choose_from_history_action_mask",
                          "lower_success_tracker",
                          "achieved_goal_tracker",
                          "history_goal",]:
            if prefix + attr_name in state_dict:
                setattr(self, attr_name, state_dict.pop(prefix + attr_name))

            if self.graph_action_space == "choose_from_history" and self.add_count_based_lower:
                if self.graph_type == "graph":
                    raise NotImplementedError
                self.choose_from_history_action_mask[:self.num_factors] = not self.is_task_learning
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def logging(self, writer: SummaryWriter, step: int) -> bool:
        if self.graph_type != "factor":
            return False

        if len(self.action_stats) > 100:
            self.action_stats = {}
            return False

        # add graph that has never been sampled
        for k, graph in self.unique_graph_from_hash.items():
            graph_id = self.unique_graph_to_id[k]
            graph_valid = self.choose_from_history_action_mask[graph_id]
            if k not in self.action_stats and graph_valid:
                self.action_stats[k] = [graph, 0., 0., 0.]

        num_graphs = len(self.action_stats)
        total_count = np.sum([total_count for _, total_count, _, _ in self.action_stats.values()])

        graph_names = []
        count_percents = []
        reach_graph_percents = []
        count_rews = []
        rtt = self.rewtermdone.rtt_functions[0]

        factor_names = self.extractor.factor_names + ["act"]
        for graph, count, reach_goal_count, reach_graph_count in sorted(self.action_stats.values(),
                                                                        key=lambda x: x[1]):
            factor = graph[:self.num_factors].argmax(axis=-1)
            parent = graph[self.num_factors:].reshape(self.num_factors + 1, self.num_edge_classes).argmax(axis=-1).astype(bool)
            graph_name = ", ".join([factor_names[i] for i, p in enumerate(parent)
                                    if p])
            graph_name = graph_name + " -> " + factor_names[factor]
            graph_names.append(graph_name)
            count_percents.append(100 * count / total_count)
            reach_graph_percents.append(100 * reach_graph_count / total_count)

            if isinstance(rtt, RTTUpperGraphCount) and rtt.use_factor_subgraph:
                count_rews.append(rtt.compute_reward(parent, is_graph=True, factor=factor))

        fig = plt.figure(figsize=(10, num_graphs * 0.4))

        # plot upper action frequency
        ax = plt.gca()
        y = np.arange(num_graphs)

        align = 'edge' if count_rews else 'center'
        rects = ax.barh(y, count_percents, align=align, height=0.4, label="action percentage")
        ax.bar_label(rects, label_type='edge', fmt="%.1f", padding=3)
        rects = ax.barh(y, reach_graph_percents, align=align, height=0.4, label="achieve percentage")
        ax.bar_label(rects, labels=[f"{v:.1f}" if v > 1 else "" for v in rects.datavalues], label_type='center')

        plt.xlim([0, np.max(count_percents) * 1.1])

        # plot upper action reward
        if count_rews:
            rects = ax.barh(y, count_rews, align='edge', height=-0.4, label="action reward")
            ax.bar_label(rects, label_type='edge', fmt="%.5f", padding=3)

        ax.set_yticks(y)
        ax.set_yticklabels(graph_names)

        plt.legend(loc="lower right")
        fig.tight_layout()
        writer.add_figure("action_stats", fig, step)
        plt.close("all")
        self.action_stats = {}

        return True
