from typing import Tuple, Union, Dict, Sequence, Callable
import os

import time
import torch
import random
import tianshou

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import mini_behavior

from State.buffer import HierarchicalReplayBuffer, VectorHierarchicalReplayBufferManager
from tianshou.data import ReplayBuffer, Batch
from Utils.logger import HRLLogger
from Utils.flatten_dict_observation_wrapper import FlattenDictObservation
from Env.minecraft2d import CraftWorld

REPO_PATH = Path(__file__).resolve(). parents[1]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def clean_dict(_dict):
    for k, v in _dict.items():
        if v == "":  # encode empty string as None
            v = None
        if isinstance(v, dict):
            v = clean_dict(v)
        _dict[k] = v
    return AttrDict(_dict)


def init_logistics(config: DictConfig, saving: bool = True, training: bool = True,) \
        -> Tuple[gym.Env, tianshou.env.BaseVectorEnv, tianshou.env.BaseVectorEnv, HRLLogger, AttrDict]:
    init_printing_format()
    torch.set_default_dtype(torch.float32)

    logger = None
    save_gif = config.save.save_gif_num
    config.exp_path = None
    if saving or save_gif:
        init_saving(config)
    if saving:
        logger = init_logger(config)

    train_env = init_env(config, config.env.num_train_envs) if training else None
    test_env = init_env(config, config.env.num_test_envs, render_mode="rgb_array" if save_gif else "human")
    single_env = get_single_env(config)

    if training:
        init_seed(train_env, config.seed)
    init_seed(test_env, config.seed + 10000)

    config = clean_dict(config)
    config.num_factors = single_env.num_factors
    config.goal_based = config.env.goal_based = single_env.goal_based
    config.factor_spaces = single_env.factor_spaces
    obs_space = single_env.observation_space
    assert isinstance(obs_space, Box)
    assert len(obs_space.shape) == 1
    config.obs_size = single_env.observation_space.shape[0]

    config = overwrite_config(config)

    return single_env, train_env, test_env, logger, config


def overwrite_config(config):
    # overwrite conflicting hyperparameters
    if config.train.mode == "task_learning_lower_frozen":
        assert config.load.load_policy != "", "load_policy must be set for task learning"
        # no need for extra upper random exploration, in addition to init_random_step
        config.train.init_upper_random_step = 0
        config.train.dynamics_pretrain_step = 0
        config.train.dynamics_warmup_step = 0

        config.train.diayn_update_per_env_step = 0
        config.train.dynamics_update_per_env_step = 0

        config.train.reset.upper.reset_freq = 0
        config.train.reset.lower.reset_freq = 0
        config.train.reset.lower.warmup_env_step = 0
        # lower is not trained, using which dynamics doesn't matter
        config.dynamics.type = "gt"

        config.policy.upper.reward_type = "task"
        config.option.upper.rew_aggregation = "sum"

    if config.policy.upper.graph_type == "none":
        assert config.policy.upper.goal_type == "diayn"
        assert config.policy.lower.type == "single_graph"

        # lower is not trained to achieve graph, using which dynamics doesn't matter
        config.dynamics.type = "gt"

        # make lower only use diayn reward
        config.option.update_schedule_frequency = 0
        config.option.lower.graph_reward_scale = 0
        config.option.lower.use_reached_graph_counter = False
        config.option.lower.training_ignore_lower_done = False
        config.option.lower.use_count_reward = False
        config.option.lower.diayn_graph_conditioned = False
        config.option.lower.diayn_scale = 1
        config.option.lower.adaptive.adaptive_diayn_coef = -1

        # turn off all priority
        config.data.her.lower.use_her = False
        config.data.prio.lower_prio = False
        config.data.prio.lower_policy.decay_window = 1

        config.policy.upper.add_count_based_lower = False
        assert config.policy.upper.diayn.num_classes > 4, "should give diayn more capacity"

    if config.dynamics.type == "gt":
        config.train.dynamics_update_per_env_step = 0
        config.train.dynamics_warmup_step = 0

    return config


def init_saving(config: DictConfig) -> None:
    save_dir = Path(config.alt_path) if len(config.alt_path) > 0 else REPO_PATH

    info = config.info.replace(" ", "_")
    experiment_dirname = info + "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    exp_path = save_dir / "results" / config.sub_dirname / experiment_dirname

    if config.train.mode == "task_learning_lower_frozen":
        task_name = config.env.task_name
        load_policy = config.load.load_policy
        assert load_policy != "", "load_policy must be set for task learning"
        # assume load_policy is in the format of results/{method_name}/{exp_info}_seed_{seed}_{timestamp}
        # for example, "results/cleaning_car_cmi/cleaning_car_cdf_per_seed_0_2024_03_07_03_28_02"
        method_name = load_policy.split(os.sep)[-3]
        skill_config_path = Path(load_policy).parent / "config.yaml"
        # seed = OmegaConf.load(skill_config_path)["seed"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_path = save_dir / "results" / "task" / task_name / f"{method_name}_seed_{config.seed}_{timestamp}"
        # exp_path = save_dir / "results" / "task" / task_name / experiment_dirname

    exp_path.mkdir(parents=True)

    gif_path = exp_path / "gifs"
    if config.save.save_gif_num:
        gif_path.mkdir(parents=True)

    with open(exp_path / "config.yaml", "w") as fp:
        OmegaConf.save(config=config, f=fp.name)

    config.exp_path = exp_path
    config.replay_buffer_dir = None

    if config.save.save_replay_buffer:
        config.replay_buffer_dir = save_dir / "replay_buffer" / config.sub_dirname / experiment_dirname
        config.replay_buffer_dir.mkdir(parents=True)


def init_logger(config: DictConfig) -> HRLLogger:
    writer = SummaryWriter(config.exp_path)
    logger = HRLLogger(writer, train_interval=1, test_interval=1, update_interval=10, policy_interval=1)
    return logger


def init_loading(
        config: AttrDict,
        dynamics,
        policy,
        lower_replay_buffer,
        upper_replay_buffer,
):
    load_config = config.load
    if load_config.load_dynamics is not None:
        load_dynamics = REPO_PATH / load_config.load_dynamics
        if load_dynamics.is_file() and load_dynamics.exists():
            print("dynamics loaded", load_dynamics)
            dynamics.load_state_dict(torch.load(load_dynamics, map_location=config.device))

    if load_config.load_policy is not None:
        load_policy = REPO_PATH / load_config.load_policy
        if load_policy.is_file() and load_policy.exists():
            print("policy loaded", load_policy)
            state_dict = torch.load(load_policy, map_location=config.device)
            if config.train.mode == "skill_learning":
                strict = True
            elif config.train.mode == "task_learning_lower_frozen":
                strict = False
                state_dict = {k: v for k, v in state_dict.items() if "upper.policy" not in k}
            else:
                raise ValueError("mode not recognized")
            policy.load_state_dict(state_dict, strict=strict)

    if load_config.load_lower_replay_buffer is not None and lower_replay_buffer is not None and load_config.load_rpb:
        load_lower_replay_buffer = REPO_PATH / load_config.load_lower_replay_buffer
        if load_lower_replay_buffer.is_file() and load_lower_replay_buffer.exists():
            lower_replay_buffer = VectorHierarchicalReplayBufferManager.load_hdf5(load_lower_replay_buffer)
            print("lower replay buffer loaded", load_lower_replay_buffer)

    if load_config.load_upper_replay_buffer is not None and upper_replay_buffer is not None and load_config.load_rpb:
        load_upper_replay_buffer = REPO_PATH / load_config.load_upper_replay_buffer
        if load_upper_replay_buffer.is_file() and load_upper_replay_buffer.exists():
            upper_replay_buffer = HierarchicalReplayBuffer.load_hdf5(load_upper_replay_buffer)
            print("higher replay buffer loaded", load_upper_replay_buffer)

    return lower_replay_buffer, upper_replay_buffer


def init_seed(env: tianshou.env.BaseVectorEnv, seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # TODO: check if env has seed()
    env.seed(seed)


def init_printing_format() -> None:
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)


def to_device(dictionary, device):
    """
    place dict of tensors + dict to device recursively
    """
    new_dictionary = {}
    for key, val in dictionary.items():
        if isinstance(val, dict):
            new_dictionary[key] = to_device(val, device)
        elif isinstance(val, torch.Tensor):
            new_dictionary[key] = val.to(device)
        else:
            raise ValueError("Unknown value type {} for key {}".format(type(val), key))
    return new_dictionary


def get_single_env(config: DictConfig, render_mode='human') -> gym.Env:
    env_config = config.env
    env_name = env_config.env_name
    mini_behavior_config = env_config.mini_behavior
    craft_world_config = env_config.craft_world
    igibson_config = env_config.igibson
    if env_name in mini_behavior_config:
        env_specific_config = mini_behavior_config[env_name]
        env_id = "MiniGrid-" + env_name + "-v0"
        kwargs = {"evaluate_graph": env_config.evaluate_graph,
                  "discrete_obs": mini_behavior_config.discrete_obs,
                  "room_size": env_specific_config.room_size,
                  "max_steps": env_specific_config.max_steps,
                  "use_stage_reward": env_specific_config.use_stage_reward,
                  "random_obj_pose": env_specific_config.random_obj_pose,
                  }
        if config.train.mode == "task_learning_lower_frozen":
            task_name = env_config.task_name
            assert task_name != ""
            kwargs["task_name"] = task_name

        env = gym.make(env_id, **kwargs)
        env = FlattenDictObservation(env)
        env.set_render_mode(render_mode)

        config.episode_length = env_specific_config.max_steps
    elif env_name == "craft_world":
        env = CraftWorld(
            use_stage_reward=craft_world_config.use_stage_reward,
            goal=craft_world_config.goal,
            must_craft_at_workspace=craft_world_config.must_craft_at_workspace,
            width=craft_world_config.width,
            height=craft_world_config.height,
            horizon=craft_world_config.horizon,
        )
        config.episode_length = craft_world_config.horizon
        env = FlattenDictObservation(env)
    elif env_name == "test_environment":
        env = TestEnv()
    else:
        # TODO: assume igibson
        # TODO: check if other modes are possible
        config.episode_length = igibson_config.max_step
        if config.train.mode == "task_learning_lower_frozen":
            task_name = env_config.task_name
            assert task_name != ""
            igibson_config.downstream_task = task_name
        else:
            igibson_config.downstream_task = "fruit"

        igibson_config = OmegaConf.to_container(igibson_config, resolve=True)
        if config.env.render:
            mode = "gui_interactive"  # "headless"  #
        else:
            mode = "headless"
        from igibson.envs.igibson_factor_obs_env import iGibsonFactorObsEnv
        env = iGibsonFactorObsEnv(
            config_file=igibson_config,
            mode=mode,
            action_timestep=1 / 10.0,
            physics_timestep=1 / 120.0,
            device_idx=config.cuda_id,
        )
        env = FlattenDictObservation(env)

    return env


def init_env(config: DictConfig, num_envs: int, render_mode='human') -> tianshou.env.BaseVectorEnv:
    if config.env.render:
        assert num_envs == 1

    env_fns = [lambda: get_single_env(config, render_mode) for _ in range(num_envs)]
    if num_envs == 1:
        return tianshou.env.DummyVectorEnv(env_fns)
    else:
        return tianshou.env.SubprocVectorEnv(env_fns)


def get_preprocess(
    state_shape: Dict[str, Union[int, Sequence[int]]], keys: Sequence[str]
) -> Tuple[Callable, int]:
    """Copied from Tianshou.common.preprocess obs, returns the obs-generating preprocess function
    """
    original_shape = state_shape
    flat_state_shapes = []
    for k in keys:
        flat_state_shapes.append(int(np.prod(state_shape[k])))

    def preprocess_obs(
        obs: Union[Batch, dict, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        if isinstance(obs, dict) or (isinstance(obs, Batch) and keys[0] in obs):
            if original_shape[keys[0]] == obs[keys[0]].shape:
                # No batch dim
                new_obs = torch.Tensor([obs[k] for k in keys]).flatten()
            else:
                bsz = obs[keys[0]].shape[0]
                new_obs = np.concatenate([obs[k].reshape(bsz, -1) for k in keys], axis=1)
                new_obs = torch.Tensor(new_obs)
        else:
            new_obs = torch.Tensor(obs)
        return new_obs
    
    return preprocess_obs
