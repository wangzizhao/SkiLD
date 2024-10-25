from typing import Union

import numpy as np
from omegaconf import DictConfig
from tianshou.data import Batch
from tianshou.policy import BasePolicy

from Causal import Dynamics
from Option.Terminate import RewardTermTruncManager
from Option.Upper import UpperPolicy


class DiaynSamplerUpper(UpperPolicy):
    def __init__(
            self,
            policy: Union[BasePolicy, None],
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
        super().__init__(policy, dynamics, rewtermdone, config)

        assert config.policy.upper.goal_type == "diayn"
        self.num_diayn_classes = config.policy.upper.diayn.num_classes
        self.diayn_onehot_helper = np.eye(self.num_diayn_classes, dtype=np.float32)

    def sample_action(self, policy: Batch, random: bool = False):
        act = policy.act
        bs = act.shape[:1]
        assert len(bs) <= 1, "only support 1-d batch size for now"

        if self.policy is None:
            policy.sampled_act = self.sample_diayn_z(bs)
        else:
            policy.sampled_act = self.diayn_onehot_helper[act]
        policy.option_choice = np.zeros(*bs, dtype=int)

        return policy

    def sample_diayn_z(self, bs):
        diayn_z = np.random.randint(low=0, high=self.num_diayn_classes, size=bs)
        diayn_z_onehot = self.diayn_onehot_helper[diayn_z]
        return diayn_z_onehot
