from torch.distributions.bernoulli import Bernoulli
from torch.distributions.one_hot_categorical import OneHotCategorical
from Option.Sampler.sampler import Sampler
import torch
import numpy as np


class FactoredGraphGoalSampler(Sampler):
    def __init__(self, num_factors, factor_dim):
        super().__init__()
        self.num_factors = num_factors
        self.factor_dim = factor_dim

    def __call__(self, logits, random=False):
        # samples are always random, random flag ignores the logits except for size
        # logits: batch_size x [num_factors + num_factors + factor dim]
        if random:
            hot = np.zeros((len(logits), self.num_factors))
            hot[np.arange(len(logits)), np.random.randint(self.num_factors, size = len(logits))] = 1
            bin = np.random.randint(2, size=(len(logits), self.num_factors + 1))  # one more factor for action
            target = np.random.rand(len(logits), self.factor_dim)
            return np.concatenate([hot, bin, target], axis=-1)

        logits = torch.tensor(logits)
        factor = OneHotCategorical(logits=logits[..., :self.num_factors]).sample()
        graph = Bernoulli(logits=logits[..., self.num_factors: 2 * self.num_factors + 1]).sample()
        return torch.cat([factor, graph, logits[..., self.num_factors * 2 + 1:]], dim=-1).cpu().numpy()
