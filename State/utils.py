import copy

import numpy as np
from collections import deque
from tianshou.data import Batch


UPPER_RESET_KEYS = ("upper_obs", "action_chain", "rew", "time_upper", "graph", "true_graph")


class Running:
    def __init__(self, n):
        self.n = n
        self.data = deque([0.0] * n, maxlen=n)
        self.mean = self.variance = self.stddev = 0.0

    def add(self, x):
        n = self.n
        oldmean = self.mean
        goingaway = self.data[0]
        self.mean = newmean = oldmean + (x - goingaway) / n
        self.data.append(x)
        self.variance += (x - goingaway) * ((x - newmean) + (goingaway - oldmean)) / (n - 1)
        self.stddev = np.sqrt(np.maximum(self.variance, 1e-20))


def compute_proximity(targets):
    # returns the object distances between every object, as a batch size x nxn matrix
    dists = list()
    for i in range(targets.shape[1]):
        target = np.expand_dims(targets[:, i], axis=1)
        dists.append(np.linalg.norm(targets - target, axis=-1))
    return np.stack(dists, axis=1)


def print_data_shape(data):
    for key in data.keys():
        if type(data[key]) == np.ndarray:
            print((key, data[key].shape))
        elif type(data[key]) == Batch:
            print(key)
            print_data_shape(data[key])
        else:
            print(key, data[key])


def reset_upper_memory(upper_buffer_memory, data, lower_buffer_ptr):
    # cases to reset upper memory:
    #   1. upper_buffer_memory is not initialized (when the collector is just initialized)
    #   2. when some / all envs are reset
    #   3. when the upper policy generates a new action because the lower policy is done
    #   all these cases will lead to corresponding entries in data.option_resample to be True,
    #   so we use data.option_resample as our reset mask

    reset_mask = data.option_resample
    if upper_buffer_memory is None:
        # 1st case
        assert reset_mask.all()
        upper_buffer_memory = Batch()
        for k in UPPER_RESET_KEYS:
            upper_buffer_memory[k] = copy.deepcopy(data[k])
        upper_buffer_memory.lower_reached_graph = data.lower_reached.reached_graph
        upper_buffer_memory.lower_buffer_start = lower_buffer_ptr
        upper_buffer_memory.lower_buffer_end = lower_buffer_ptr
        upper_buffer_memory.sample_ready = np.ones_like(lower_buffer_ptr, dtype=bool)

    if not reset_mask.any():
        return upper_buffer_memory

    for k in UPPER_RESET_KEYS:
        upper_buffer_memory[k][reset_mask] = copy.deepcopy(data[k][reset_mask])

    upper_buffer_memory.obs = upper_buffer_memory.upper_obs
    upper_buffer_memory.act = upper_buffer_memory.action_chain.upper
    # reward will be updated when the collector calls update_upper_memory_trajectory() which calls update_upper_memory()
    # so initialize as 0 here
    upper_buffer_memory.rew[reset_mask] = 0
    upper_buffer_memory.lower_buffer_start[reset_mask] = lower_buffer_ptr[reset_mask]
    upper_buffer_memory.lower_reached_graph[reset_mask] = data.lower_reached.reached_graph[reset_mask]
    return upper_buffer_memory


def update_upper_memory(upper_buffer_memory, data, lower_buffer_ptr, upper_rew_aggregation="sum"):
    # TODO: only update rew, terminated, truncated, obs_next for now, check if other keys are used
    # updates "next" values to the current value when relevant, and combines rewards
    if upper_rew_aggregation == "sum":
        upper_buffer_memory.rew += data.reward_chain.upper
    elif upper_rew_aggregation == "max":
        upper_buffer_memory.rew = np.maximum(data.reward_chain.upper, upper_buffer_memory.rew)
    else:
        raise NotImplementedError

    upper_buffer_memory.terminated = data.term_chain.upper
    upper_buffer_memory.truncated = data.trunc_chain.upper
    upper_buffer_memory.obs_next = data.upper_obs_next
    upper_buffer_memory.info = data.info
    upper_buffer_memory.lower_buffer_end = lower_buffer_ptr
    upper_buffer_memory.lower_reached_graph = upper_buffer_memory.lower_reached_graph | data.lower_reached.reached_graph
    assert (upper_buffer_memory.time_upper == data.time_upper).all()


def update_upper_memory_trajectory(upper_buffer_memory, upper_trajectories,
                                   data, lower_buffer_ptr, ready_env_ids,
                                   upper_rew_aggregation="sum"):
    update_upper_memory(upper_buffer_memory, data, lower_buffer_ptr, upper_rew_aggregation)

    lower_done = data.term_chain.lower | data.trunc_chain.lower
    env_ind_local = np.where(lower_done)[0]
    for id in env_ind_local:
        global_ind = ready_env_ids[id]
        upper_trajectories[global_ind].append(upper_buffer_memory[id])
    return upper_buffer_memory


def upper_buffer_add(upper_trajectories, upper_buffer, lower_buffer, env_ind_global):
    for id in env_ind_global:
        for upper_data in upper_trajectories[id]:
            if upper_buffer is not None:
                upper_idx, _, _, _ = upper_buffer.add(upper_data)
                lower_buffer.update_upper_buffer_idx(upper_data.lower_buffer_start, upper_data.lower_buffer_end, upper_idx)
        upper_trajectories[id] = list()


class ObjDict(dict):
    def __init__(self, ins_dict=None):
        super().__init__()
        if ins_dict is not None:
            for n in ins_dict.keys(): 
                self[n] = ins_dict[n]

    def insert_dict(self, ins_dict):
        for n in ins_dict.keys(): 
            self[n] = ins_dict[n]

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

