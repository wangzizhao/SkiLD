import numpy as np

from Option.Terminate import RewardTerminateTruncate


class RTTUpperSpecialGraph(RewardTerminateTruncate):
	"""
	Count the number of each unique graph and return reward as 1 / sqrt(count)
	Special reward for special graph
	"""

	def __init__(self, **kwargs):
		# initialize hyperparameters
		super().__init__(**kwargs)
		self.graph_count = {}

	def update_graph_count(self, batch):
		assert batch.graph.ndim == 3
		graphs, counts = np.unique(batch.graph, axis=0, return_counts=True)
		for g, c in zip(graphs, counts):
			g = g.tobytes()               # make np.ndarray hashable, so it's a valid key for dict
			self.graph_count[g] = self.graph_count.get(g, 0) + c

	def rew(self, batch):
		if self.training and not self.updating:
			self.update_graph_count(batch)
		rew = []
		for graph in batch.graph:
			# 0： agent, 1: frig, 2: sink, 3, 4, 5:obj, 6: action
			cur_r = np.sum(graph[3:, 2])
			rew.append(cur_r)
		return np.array(rew)

	def term(self, batch):
		return batch.terminated

	def trunc(self, batch):
		return batch.truncated
