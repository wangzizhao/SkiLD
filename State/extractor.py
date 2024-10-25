import numpy as np
import gymnasium as gym
from tianshou.data.batch import Batch
import torch


class Extractor:
    """
    Handles factorizaton of state
    """

    def __init__(self, env: gym.Env):
        self.dict_obs_space = env.dict_obs_space
        self.breakpoints = env.breakpoints                          # includes 0, (num_factors + 1)
        self.goal_based = env.goal_based
        self.num_factors = env.num_factors

        self.factor_names = list(self.dict_obs_space.keys())

        # get the longest segment
        self.longest = np.max(self.breakpoints[1:] - self.breakpoints[:-1])
        self.obs_mask = np.zeros((self.num_factors, self.longest), dtype=bool)
        for i, (lbp, rbp) in enumerate(zip(self.breakpoints[:-1], self.breakpoints[1:])):
            self.obs_mask[i, :rbp - lbp] = True

    def slice_targets(self, batch, next=False, flatten=False, skip_num=0):
        """
        takes an observation of state of length n, and breaks it into segments at the breakpoints
        padded to the longest segment length
        """
        obs = (batch.obs_next if next else batch.obs) if type(batch) == Batch else batch  # allows array inputs
        observation = obs.observation if type(obs) == Batch else obs
        bs = observation.shape[:-1]
        if flatten: skipped, observation = observation[..., :skip_num], observation[..., skip_num:]
        if type(observation) == torch.Tensor: 
            target = torch.zeros((bs + (self.num_factors, self.longest)), dtype=observation.dtype, device=observation.device)
            # print(observation.shape, target.shape, bs, np.arange(*bs), self.obs_mask.shape, self.num_factors, self.longest, flatten, observation.shape, skipped.shape, skip_num)
            target[:, self.obs_mask] = observation # TODO issues if bs is multidimensional, but [..., self.obs_mask] notation doesn't work for tensors
            if flatten:
                target = target.reshape(-1, self.num_factors * self.longest)
                target = torch.cat([skipped, target], dim=-1)
        else: 
            target = np.zeros((bs + (self.num_factors, self.longest)), dtype=observation.dtype) # assumes numpy array
            target[..., self.obs_mask] = observation
            if flatten:
                target = target.reshape(-1, self.num_factors * self.longest)
                target = np.concatenate([skipped, target], axis=-1)
        return target
