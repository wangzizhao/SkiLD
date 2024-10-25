import copy
import numpy as np
import yaml


class Index:
    def __init__(self):
        self.name2idx = dict()
        self.idx2name = dict()

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            dict_ = self.idx2name
        elif isinstance(item, str):
            dict_ = self.name2idx
        else:
            raise NotImplementedError("Unknow item {} with type = {}".format(item, type(item)))

        return dict_.get(item, None)

    def __len__(self):
        return len(self.name2idx)

    def __contains__(self, item):
        return self[item] is not None

    def index(self, item):
        assert isinstance(item, str)
        if item not in self.name2idx:
            idx = len(self.name2idx)
            self.name2idx[item] = idx
            self.idx2name[idx] = item
        idx = self[item]
        return idx


class Cookbook(object):
    def __init__(self, recipes_path):
        with open(recipes_path) as recipes_f:
            recipes = yaml.safe_load(recipes_f)

        self.primitive_recipes = recipes["primitives"]

        self.index = index = Index()
        self.environment_idxes = [index.index(e) for e in recipes["environment"]]
        self.primitive_idxes = [index.index(p) for p in recipes["primitives"]]
        self.craft_idxes = [index.index(p) for p in recipes["recipes"]]

        self.environments = {}
        for env_obj_name, env_obj_info in recipes["environment"].items():
            if env_obj_name in ["boundary", "workshop", "furnace"]:
                continue
            d = {}
            for key, val in env_obj_info.items():
                if key == "_require":
                    val = index[val]
                d[key] = val
            self.environments[index[env_obj_name]] = d

        self.primitives = {}
        for primitive_name, primitive_info in recipes["primitives"].items():
            d = {}
            for key, val in primitive_info.items():
                # special keys
                if "_" in key:
                    val = index[val]
                d[key] = val
            self.primitives[index[primitive_name]] = d

        self.recipes = {}
        furnace_slots, furnace_max_stage = [], 1
        for craft_name, craft_info in recipes["recipes"].items():
            d, use_furnace = {}, False
            for key, val in craft_info.items():
                # special keys
                if "_" in key:
                    if key == "_at":
                        use_furnace = val == "furnace"
                        val = index[val]
                else:
                    key = index[key]
                d[key] = val

            self.recipes[index[craft_name]] = d
            if use_furnace:
                furnace_slots.append(index[craft_name])
                furnace_max_stage = max(furnace_max_stage, d["_step"])

        self.idx2furnace_slot, self.furnace_slot2idx = {}, {}
        for i, craft_idx in enumerate(furnace_slots):
            self.idx2furnace_slot[craft_idx] = i + 1
            self.furnace_slot2idx[i + 1] = craft_idx
        self.furnace_max_stage = furnace_max_stage

        self.n_kinds = len(index)
