import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation


class FlattenDictObservation(FlattenObservation):
    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        # TODO: check if goal_based
        self.goal_based = False

        self.dict_obs_space = env.observation_space
        self.num_factors = len(env.observation_space.spaces)

        # get state to factor mapping
        self.breakpoints = [0]
        self.factor_spaces = []
        for obs_k, obs_space in env.observation_space.spaces.items():
            if isinstance(obs_space, spaces.Box):
                assert len(obs_space.shape) == 1
                self.breakpoints.append(self.breakpoints[-1] + np.sum(obs_space.shape[0]))
            elif isinstance(obs_space, spaces.MultiDiscrete):
                self.breakpoints.append(self.breakpoints[-1] + np.sum(obs_space.nvec))
            else:
                raise NotImplementedError
            self.factor_spaces.append(obs_space)
        self.breakpoints = np.array(self.breakpoints)

        super().__init__(env)

    def __getattr__(self, name: str):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore.

        Args:
            name: The variable name

        Returns:
            The value of the variable in the wrapper stack

        Warnings:
            This feature is deprecated and removed in v1.0 and replaced with `env.get_attr(name})`
        """
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        if isinstance(self.env, gym.Wrapper):
            return getattr(self.env.unwrapped, name)
        else:
            return getattr(self.env, name)
