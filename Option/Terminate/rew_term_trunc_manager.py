from typing import Iterable

import numpy as np
from torch import nn

from Option.Terminate import RewardTerminateTruncate


class RewardTermTruncManager(nn.Module):
    # computes the reward, termination and done signals for a particular policy
    # this should be a function of the observation, goal and time.
    # upper_policy and lower_policies will have this attached to them
    def __init__(self, rtt_functions: Iterable[RewardTerminateTruncate]):
        super().__init__()
        # see Option.Terminate.General.rtt_base for what goes in here
        # these functions manage reward, terminate and trunc
        self.rtt_functions = nn.ModuleList(rtt_functions)

    def check_rew(self, data, idx=None):
        if idx is None:
            reward = np.array([rtt_func.rew(data) for rtt_func in self.rtt_functions])
            return reward.sum(axis=0)
        else:
            rtt_func = self.rtt_functions[idx]
            return rtt_func.rew(data)

    def check_rew_term_trunc(self, data, idx=0):
        rtt_func = self.rtt_functions[idx]

        # TODO: may not work for wide lower (multiple low policies)
        rewards = rtt_func.rew(data)
        terms = rtt_func.term(data)
        timeouts = rtt_func.trunc(data)

        env_terms = data.terminated
        env_truncs = data.truncated

        return rewards, terms | env_terms, timeouts | env_truncs

    def check_lower_reached(self, data, idx=0):
        rtt_func = self.rtt_functions[idx]
        reached_graph, reached_goal = rtt_func.lower_reached(data)
        return reached_graph, reached_goal

    def set_updating(self, updating):
        for rtt in self.rtt_functions:
            rtt.updating = updating

    def update_lower_stats(self, lower_metrics, desired_goal):
        for rtt in self.rtt_functions:
            rtt.update_lower_stats(lower_metrics, desired_goal)

    def update_schedules(self):
        for rtt in self.rtt_functions:
            rtt.update_schedules()

    def update_state_counts(self, batch):
        for rtt in self.rtt_functions:
            rtt.update_state_counts(batch)
