import numpy as np

from Option.Sampler.sampler import Sampler


class DeltaSampler(Sampler):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def __call__(self, logits, random=False):
        if random:
            bs = len(logits)
            return np.array([self.action_space.sample() for _ in range(bs)])
        return logits
