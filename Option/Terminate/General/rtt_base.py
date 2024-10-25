from torch import nn
from tianshou.data import Batch


class RewardTerminateTruncate(nn.Module):
    """
    contains the functions to compute reward, termination and timeout.
    Since reward and termination are often linked, having them in the same
    class allows shared code
    """
    def __init__(self, **kwargs):
        super().__init__()
        # initialize hyperparameters
        self.timeout = kwargs["timeout"]
        self.updating = False
        self.num_updates = 0

    def rew(self, batch):
        # returns the reward computed using a batch
        # (preferably only using obs.achieved_goal, obs.desired_goal and obs.reward)
        raise NotImplementedError

    def term(self, batch):
        # returns the termination computed using a batch
        raise NotImplementedError

    def trunc(self, batch):
        # returns timeouts from batch
        raise NotImplementedError

    def lower_reached(self, batch):
        # returns if the lower policy reached the graph and the goal from batch
        raise NotImplementedError

    def update_lower_stats(self, lower_metrics, desired_goal):
        pass

    def update_schedules(self):
        self.num_updates += 1       # updates any adaptive rates (count based, etc.)
    
    def update_state_counts(self, batch):
        pass