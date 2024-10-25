import numpy as np

from Option.Terminate import RewardTerminateTruncate


class RTTUpperGraphCount(RewardTerminateTruncate):
    """
    Count the number of each unique graph and return reward as 1 / sqrt(count)
    """

    def __init__(self, **kwargs):
        # initialize hyperparameters
        super().__init__(**kwargs)
        self.power = kwargs["power"]
        self.use_factor_subgraph = kwargs["use_factor_subgraph"]
        if not self.use_factor_subgraph:
            raise NotImplementedError
        self.buffer = None
        self.graph_count = None

    def compute_reward(self, graph_count_idx, is_graph=False, factor=None):
        """
        if self.use_factor_subgraph:
            if factor is None: sum the reward of all factor sub-graphs
            else: return the reward of the {factor}-th sub-graph
        """
        if is_graph:
            graph_count_idx = np.dot(graph_count_idx, self.graph_to_count_idx)      # (bs, num_factors) or (bs,)

        # upper: reward for lower achievement
        assert graph_count_idx.ndim <= 2

        if self.use_factor_subgraph:
            if factor is None:
                # graph_count_idx: (bs, num_factors)
                graph_count_idx = graph_count_idx[..., None]
                graph_count = self.graph_count
                if graph_count_idx.ndim == 3:
                    graph_count = graph_count[None]
                count = np.take_along_axis(graph_count, graph_count_idx, axis=-1).squeeze(-1)
            else:
                count = self.graph_count[factor, graph_count_idx]
        else:
            raise NotImplementedError

        # clip to 1 so that unseen graph will have a count of 1
        count = count.clip(min=1)
        rew = np.power(count, self.power)
        if self.use_factor_subgraph and factor is None:
            rew = np.sum(rew, axis=-1)

        return rew

    def update_graph_count(self):
        assert self.buffer is not None

        self.graph_to_count_idx = self.buffer.graph_to_count_idx
        self.graph_count = self.buffer.valid_graph_count

        # all-zero sub-graph is generated when dynamics model makes incorrect prediction,
        # thus the graph may not be accurate and, as a result, not counted
        # those graph correspond to idx 0 in axis 1
        assert np.all(self.graph_count[:, 0] == 0)

    def initialize_graph_count(self, graph):
        if self.use_factor_subgraph:
            assert graph.ndim == 3
            num_factors = graph.shape[1]
            self.graph_to_count_idx = np.power(2, np.arange(num_factors + 1)).astype(int)
            self.graph_count = np.ones((num_factors, 2 ** (num_factors + 1)), dtype=int)
        else:
            raise NotImplementedError

    def rew(self, batch):
        if self.training and not self.updating:
            self.update_graph_count()

        if self.graph_count is None:
            self.initialize_graph_count(batch.graph)

        rew = self.compute_reward(batch.graph_count_idx)
        return rew

    def term(self, batch):
        return batch.terminated

    def trunc(self, batch):
        return batch.truncated

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(self, *args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict.update({prefix + k: v for k, v in
                           {"graph_to_count_idx": self.graph_to_count_idx,
                            "graph_count": self.graph_count}.items()
                           })
        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for attr_name in ["graph_to_count_idx",
                          "graph_count",]:
            if prefix + attr_name in state_dict:
                setattr(self, attr_name, state_dict.pop(prefix + attr_name))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
