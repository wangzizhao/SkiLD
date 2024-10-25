import copy
from Causal import Dynamics


class GroundTruthGraph(Dynamics):
    """
    Returns the ground truth causal graph by accessing data.true_graph
    """

    def __init__(self, env, extractor):
        super().__init__(env, extractor)
        # TODO: we probably don't need to initialize anything
        pass

    def __call__(self, data):
        return copy.deepcopy(data.true_graph)
