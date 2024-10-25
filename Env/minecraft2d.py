try:
    from Env.cookbook import Cookbook
except:
    from cookbook import Cookbook

import gymnasium as gym
import os
import copy
import numpy as np

from collections import OrderedDict
from gymnasium import spaces

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4


class CraftWorld(gym.Env):
    def __init__(
            self,
            use_stage_reward=False,
            goal="gem",
            width=16,
            height=16,
            horizon=500,
            must_craft_at_workspace=True,
    ):
        self.use_stage_reward = use_stage_reward
        self.goal = goal
        self.width = width
        self.height = height
        self.horizon = horizon
        self.must_craft_at_workspace = must_craft_at_workspace

        recipe_path = os.path.join((os.path.dirname(os.path.realpath(__file__))), "craft_recipes",
                                   self.goal + "_recipe.yaml")
        self.cookbook = cookbook = Cookbook(recipe_path)

        self.environment_idxes = cookbook.environment_idxes
        self.primitive_idxes = cookbook.primitive_idxes
        self.craft_idxes = cookbook.craft_idxes

        self.environments = cookbook.environments
        self.primitives = cookbook.primitives
        self.recipes = cookbook.recipes

        self.has_furnace = "furnace" in self.cookbook.index
        self.goal_idx = cookbook.index[self.goal]
        self.boundary_idx = cookbook.index["boundary"]
        self.workshop_idx = cookbook.index["workshop"]
        self.furnace_idx = cookbook.index["furnace"]

        # for computing staged reward
        assert self.goal_idx in self.primitive_idxes
        goal_info = self.primitives[self.goal_idx]

        self.path_tool = None
        self.pick_tool = goal_info.get("_require", None)

        if "_surround" in goal_info:
            surround_idx = goal_info["_surround"]
            if surround_idx in self.environments:
                surround_info = self.environments[surround_idx]
            elif surround_idx in self.primitives:
                surround_info = self.primitives[surround_idx]
            else:
                raise NotImplementedError
            self.path_tool = surround_info.get("_require", None)

        self.craft_tools = {}
        self.inter_tools = {}
        self.num_craft_tool_stages = 0

        for tool_idx in [self.path_tool, self.pick_tool]:
            if tool_idx is None:
                continue
            recipe, num_stages = self.add_recipe(tool_idx)
            self.craft_tools[tool_idx] = [recipe, num_stages]
            self.num_craft_tool_stages += num_stages

        # Intialize actions: USE; move to treasure; move to workshop; move to walls; move to ingredients; craft tools
        self.action_info_ready = False
        self.slice_dict = None
        self.dynamics_keys = None
        self.reset()

        # Observation and action space
        # agent: (x, y, dir)
        observation_space = [("agent", spaces.MultiDiscrete([self.width, self.height, 4]))]

        # object: (x, y)
        for key in self.state:
            if key == "furnace_slot":
                nvec = [len(self.cookbook.idx2furnace_slot) + 1]
            elif key == "furnace_stage":
                nvec = [self.cookbook.furnace_max_stage + 1]
            else:
                nvec = [self.width, self.height]
            observation_space.append((key, spaces.MultiDiscrete(nvec)))

        # inventory_name: num
        max_primitive_num = max([primitive_info["num"] for primitive_info in self.primitives.values()])
        for idx in list(self.primitives.keys()) + list(self.recipes.keys()):
            name = self.cookbook.index.idx2name[idx]
            observation_space.append(
                (
                    f"inventory_{name}",
                    spaces.MultiDiscrete([max_primitive_num + 1])
                )
            )
        self.observation_space = spaces.Dict(observation_space)
        self.action_space = spaces.Discrete(self.action_dim)

    def add_inter_tool(self, index):
        if index in self.inter_tools:
            return
        sub_recipe, num_sub_stages = self.add_recipe(index)
        self.inter_tools[index] = [sub_recipe, num_sub_stages]
        self.num_craft_tool_stages += num_sub_stages
        if index == self.furnace_idx:
            self.num_craft_tool_stages += 2  # 2 extra stages to set up the furnace

    def add_recipe(self, index):
        recipe_new, num_stages = {}, 0
        for k, v in self.recipes[index].items():
            if k in ["_at", "_yield", "_step"]:
                if k == "_at" and v == self.furnace_idx:
                    self.add_inter_tool(self.furnace_idx)
                continue
            elif k in self.primitive_idxes:
                recipe_new[k] = v
                num_stages += v
                tool_idx = self.primitives[k].get("_require", None)
                if tool_idx:
                    self.add_inter_tool(tool_idx)
            else:
                assert k in self.recipes, "Unknow key {}".format(k)
                sub_recipe, num_sub_stages = self.add_recipe(k)
                recipe_new[k] = [v, sub_recipe, num_sub_stages]
                num_stages += v * num_sub_stages
        return recipe_new, num_stages + 1  # 1 extra stage to craft the tool

    def seed(self, seed=None):
        pass

    def reset(self, seed=None, options=None):
        # deterministic reset for now
        np.random.seed(0)

        self.cur_step = self.stage_completion_tracker = 0

        assert self.goal_idx not in self.environments
        self.sample_scenario()

        return self.get_state(), {}

    def neighbors(self, pos, dir=None):
        x, y = pos
        neighbors = []
        if x > 0 and (dir is None or dir == LEFT):
            neighbors.append((x - 1, y, LEFT))
        if y > 0 and (dir is None or dir == DOWN):
            neighbors.append((x, y - 1, DOWN))
        if x < self.width - 1 and (dir is None or dir == RIGHT):
            neighbors.append((x + 1, y, RIGHT))
        if y < self.height - 1 and (dir is None or dir == UP):
            neighbors.append((x, y + 1, UP))
        return neighbors

    def random_free(self, requires_free_neighbor=False):
        grid = self.grid
        pos = None
        while pos is None:
            x, y = np.random.randint(grid.shape[0]), np.random.randint(grid.shape[1])
            if grid[x, y, :].any():
                continue
            if requires_free_neighbor:
                if any([self.grid[nx, ny].any() for nx, ny, _ in self.neighbors((x, y))]):
                    continue
            pos = (x, y)
        return pos

    def sample_scenario(self):
        cookbook = self.cookbook

        # generate grid
        self.grid = grid = np.zeros((self.width, self.height, self.cookbook.n_kinds))
        i_bd = self.cookbook.index["boundary"]
        grid[0, :, i_bd] = 1
        grid[self.width - 1:, :, i_bd] = 1
        grid[:, 0, i_bd] = 1
        grid[:, self.height - 1:, i_bd] = 1

        self.state = OrderedDict()

        # treasure
        gx, gy = np.random.randint(1, self.width - 1), np.random.randint(1, 3)

        assert not grid[gx, gy].any()
        grid[gx, gy, self.goal_idx] = 1
        self.state[self.goal] = np.array([gx, gy])

        obst_idx = self.primitives[self.goal_idx].get("_surround", None)
        if obst_idx:
            assert obst_idx in self.environments
            obst_name = cookbook.index[obst_idx]
            num_obsts = 0
            for ox, oy, _ in self.neighbors((gx, gy)):
                item_name = obst_name + str(num_obsts)
                if grid[ox, oy, :].any():
                    assert grid[ox, oy, :].argmax() == i_bd
                    self.state[item_name] = np.array([0, 0])
                else:
                    grid[ox, oy, obst_idx] = 1
                    self.state[item_name] = np.array([ox, oy])

                num_obsts += 1

        # ingredients
        for primitive_idx, primitive_info in self.primitives.items():
            if primitive_idx == self.goal_idx:
                continue
            for i in range(primitive_info["num"]):
                x, y = self.random_free(requires_free_neighbor=True)
                grid[x, y, primitive_idx] = 1
                primitive_name = cookbook.index[primitive_idx]
                item_name = primitive_name + str(i)
                self.state[item_name] = np.array([x, y])

        # generate crafting stations
        station_names = ["workshop"]

        has_furname = "furnace" in cookbook.index
        if has_furname:
            station_names.append("furnace")

        for station_name in station_names:
            x, y = self.random_free(requires_free_neighbor=True)
            grid[x, y, cookbook.index[station_name]] = 1
            self.state[station_name] = np.array([x, y])

        if has_furname:
            self.state["furnace_ready"] = False
            self.state["furnace_slot"] = 0
            self.state["furnace_stage"] = 0

        # generate init pos
        self.inventory = np.zeros(self.cookbook.n_kinds, dtype=int)
        self.pos = self.random_free()
        self.dir = np.random.randint(4)

        # set up action information
        if not self.action_info_ready:
            self.action_info_ready = True
            self.craft_action_starts = USE + 1
            self.action_dim = self.craft_action_starts + len(self.craft_idxes)

    def get_state(self):
        state = copy.deepcopy(self.state)
        for k, v in state.items():
            if isinstance(v, (int, bool)):
                state[k] = np.array([v])

        state["agent"] = np.array([self.pos[0], self.pos[1], self.dir])

        for idx, num in zip(
                list(self.primitives.keys()) + list(self.recipes.keys()),
                self.inventory[len(self.environment_idxes):]
        ):
            name = self.cookbook.index.idx2name[idx]
            state[f"inventory_{name}"] = np.array([num])
        return state

    def check_success(self):
        return self.inventory[self.goal_idx] > 0

    def has_craft(self, craft_idx):
        return self.inventory[craft_idx] > 0 or (craft_idx == self.furnace_idx and self.state["furnace_ready"])

    def can_collect_treasure(self):
        has_path = any([not self.grid[x, y].any() for x, y, _ in self.neighbors(self.state[self.goal])])
        if has_path:
            if self.pick_tool:
                return self.inventory[self.pick_tool] > 0
        else:
            if self.pick_tool and not self.inventory[self.pick_tool]:
                return False
            if self.path_tool and not self.inventory[self.path_tool]:
                return False
        return True

    def reward(self):
        num_required_tools = len(self.inter_tools) + len(self.craft_tools)
        if self.check_success():
            reward = 1
            self.stage_completion_tracker = num_required_tools + 2
        elif self.can_collect_treasure():
            reward = 0.75
            self.stage_completion_tracker = num_required_tools + 1
        else:
            self.stage_completion_tracker = 0
            for k in {**self.inter_tools, **self.craft_tools}:
                if self.has_craft(k):
                    self.stage_completion_tracker += 1
            reward = 0.5 * self.stage_completion_tracker / num_required_tools

        if self.use_stage_reward:
            return reward
        else:
            return float(reward == 1)

    def find_key(self, x, y):
        for key, val in self.state.items():
            if "_" in key:
                continue
            if (val == (x, y)).all():
                return key
        return None

    def step(self, action):
        assert action < self.action_dim

        prev_pos = x, y = self.pos
        prev_dir = self.dir
        prev_state = copy.deepcopy(self.state)
        state = self.state
        inventory = self.inventory
        cookbook = self.cookbook

        remove_thing_from_grid = crafted_in_workshop = start_craft_in_furnace = False
        thing = thing_name = None

        # move actions
        if action == DOWN:
            dx, dy = (0, -1)
            n_dir = DOWN
        elif action == UP:
            dx, dy = (0, 1)
            n_dir = UP
        elif action == LEFT:
            dx, dy = (-1, 0)
            n_dir = LEFT
        elif action == RIGHT:
            dx, dy = (1, 0)
            n_dir = RIGHT
        else:
            dx, dy = 0, 0
            n_dir = self.dir
            for nx, ny, _ in self.neighbors(self.pos, self.dir):
                here = self.grid[nx, ny, :]
                if not self.grid[nx, ny, :].any():
                    continue

                assert here.sum() == 1, "impossible world configuration"
                thing = here.argmax()
                if thing == self.boundary_idx:
                    continue

                thing_name = self.find_key(nx, ny)

                if action == USE:
                    if thing in self.primitive_idxes:
                        primitive_info = self.primitives[thing]
                        required_tool_idx = primitive_info.get("_require", None)
                        if required_tool_idx is None or inventory[required_tool_idx] > 0:
                            remove_thing_from_grid = True
                            inventory[thing] += 1
                    elif thing == self.workshop_idx:
                        continue
                    elif thing == self.furnace_idx:
                        furnace_slot, furnace_stage = state["furnace_slot"], state["furnace_stage"]
                        if not state["furnace_ready"]:
                            if inventory[self.furnace_idx]:
                                assert furnace_slot == furnace_stage == 0
                                state["furnace_ready"] = True
                                inventory[self.furnace_idx] -= 1
                        else:
                            if furnace_slot != 0:
                                craft_idx = cookbook.furnace_slot2idx[furnace_slot]
                                craft_recipe = self.recipes[craft_idx]
                                if furnace_stage == craft_recipe["_step"]:
                                    inventory[craft_idx] += craft_recipe.get("_yield", 1)
                                    state["furnace_slot"], state["furnace_stage"] = 0, 0
                    else:
                        env_obj_info = self.environments[thing]
                        required_tool_idx = env_obj_info.get("_require", None)
                        if required_tool_idx is None or inventory[required_tool_idx] > 0:
                            remove_thing_from_grid = True
                            if env_obj_info["_consume"]:
                                inventory[required_tool_idx] -= 1

                    if remove_thing_from_grid:
                        self.grid[nx, ny, thing] = 0
                        state[thing_name] = np.array([0, 0])

                    break

                else:
                    output = self.craft_idxes[action - self.craft_action_starts]
                    recipe = self.recipes[output]

                    yld = recipe.get("_yield", 1)
                    ing = [i for i in recipe if isinstance(i, int)]
                    if any(inventory[i] < recipe[i] for i in ing):
                        continue

                    if not self.must_craft_at_workspace:
                        # overwrite the workspace requirement
                        thing = recipe["_at"]
                        thing_name = self.cookbook.index.idx2name[thing]

                    if thing != recipe["_at"]:
                        continue

                    if thing == self.workshop_idx:
                        for i in ing:
                            inventory[i] -= recipe[i]
                        inventory[output] += yld
                        crafted_in_workshop = True
                    elif thing == self.furnace_idx:
                        if not state["furnace_ready"] or state["furnace_slot"] != 0:
                            continue
                        for i in ing:
                            inventory[i] -= recipe[i]
                        state["furnace_slot"] = cookbook.idx2furnace_slot[output]
                        start_craft_in_furnace = True
                    else:
                        raise NotImplementedError

        n_x = x + dx
        n_y = y + dy
        if not 0 <= n_x < self.width:
            n_x = x
        if not 0 <= n_y < self.height:
            n_y = y
        if self.grid[n_x, n_y, :].any():
            has_obstacle = True
            n_x, n_y = x, y
        self.pos = (n_x, n_y)
        self.dir = n_dir

        if self.has_furnace:
            furnace_slot, furnace_stage = state["furnace_slot"], state["furnace_stage"]
            if furnace_slot != 0 and not start_craft_in_furnace:
                assert state["furnace_ready"]
                craft_idx = cookbook.furnace_slot2idx[furnace_slot]
                if furnace_stage < self.recipes[craft_idx]["_step"]:
                    state["furnace_stage"] += 1

        evaluate_mask = True
        mask = factor_mask = None
        if evaluate_mask:
            if self.slice_dict is None:
                slice_dict, cum = {}, 0
                for k, space in self.observation_space.spaces.items():
                    k_dim = len(space.nvec)
                    slice_dict[k] = slice(cum, cum + k_dim)
                    cum += k_dim
                slice_dict["action"] = slice(cum, cum + 1)
                self.slice_dict, self.feature_dim = slice_dict, cum
            else:
                slice_dict = self.slice_dict
            action_idx = self.feature_dim

            mask = np.eye(self.feature_dim, self.feature_dim + 1, dtype=bool)
            inventory_offset = min(
                slice_k.start
                for k, slice_k in self.slice_dict.items()
                if k.startswith("inventory")
            )
            inventory_offset = inventory_offset - len(self.environment_idxes)

            if self.has_furnace:
                furnace_ready_slice = slice_dict["furnace_ready"]
                furnace_slot_slice = slice_dict["furnace_slot"]
                furnace_stage_slice = slice_dict["furnace_stage"]

            agent_pos_slice = slice(slice_dict["agent"].start, slice_dict["agent"].start + 2)
            agent_dir_slice = slice(slice_dict["agent"].start + 2, slice_dict["agent"].stop)

            if action in [DOWN, UP, LEFT, RIGHT]:
                if prev_dir != self.dir:
                    mask[agent_dir_slice, action_idx] = True
                if (x, y) != self.pos:
                    pos_idx = int(action < 2)
                    mask[pos_idx, action_idx] = True
                # else:
                #     mask[pos_idx, agent_pos_slice] = True
                #     if has_obstacle:
                #         n_x, n_y = x + dx, y + dy
                #         obstacle_key = self.find_key(n_x, n_y)
                #         mask[pos_idx, slice_dict[obstacle_key]] = True
            else:
                if thing_name is not None:
                    thing_pos_slice = slice_dict[thing_name]

                if action == USE:
                    if remove_thing_from_grid:
                        mask[thing_pos_slice, agent_pos_slice] = True
                        mask[thing_pos_slice, agent_dir_slice] = True
                        mask[thing_pos_slice, thing_pos_slice] = True
                        mask[thing_pos_slice, action_idx] = True

                        if thing in self.primitive_idxes:
                            # thing_invent_idx = inventory_offset + thing
                            # mask[thing_invent_idx, agent_pos_slice] = True
                            # mask[thing_invent_idx, agent_dir_slice] = True
                            # mask[thing_invent_idx, thing_pos_slice] = True
                            # mask[thing_invent_idx, action_idx] = True
                            if required_tool_idx:
                                tool_invent_idx = inventory_offset + required_tool_idx
                                mask[thing_pos_slice, tool_invent_idx] = True
                                # mask[thing_invent_idx, tool_invent_idx] = True
                        else:
                            if required_tool_idx:
                                tool_invent_idx = inventory_offset + required_tool_idx
                                mask[thing_pos_slice, tool_invent_idx] = True
                                # if env_obj_info["_consume"]:
                                #     mask[tool_invent_idx, agent_pos_slice] = True
                                #     mask[tool_invent_idx, agent_dir_slice] = True
                                #     mask[tool_invent_idx, thing_pos_slice] = True
                                #     mask[tool_invent_idx, tool_invent_idx] = True
                                #     mask[tool_invent_idx, action_idx] = True

                    if self.has_furnace:
                        if prev_state["furnace_ready"] != state["furnace_ready"]:
                            furnace_invent_idx = inventory_offset + self.furnace_idx
                            mask[furnace_ready_slice, agent_pos_slice] = True
                            mask[furnace_ready_slice, agent_dir_slice] = True
                            mask[furnace_ready_slice, thing_pos_slice] = True
                            mask[furnace_ready_slice, furnace_invent_idx] = True
                            mask[furnace_ready_slice, action_idx] = True

                        if prev_state["furnace_slot"] != state["furnace_slot"]:
                            craft_invent_idx = inventory_offset + craft_idx
                            for slice_ in [furnace_slot_slice, furnace_stage_slice, craft_invent_idx]:
                                mask[slice_, agent_pos_slice] = True
                                mask[slice_, agent_dir_slice] = True
                                mask[slice_, thing_pos_slice] = True
                                mask[slice_, furnace_slot_slice] = True
                                mask[slice_, furnace_stage_slice] = True
                                mask[slice_, action_idx] = True

                else:
                    if crafted_in_workshop or start_craft_in_furnace:
                        ing = [inventory_offset + i for i in ing]
                        if thing == self.furnace_idx:
                            things = ing + [slice_dict["furnace_slot"]]
                            ing = ing + [slice_dict["furnace_ready"], slice_dict["furnace_slot"]]
                        else:
                            # original:
                            # things = ing + [inventory_offset + output]

                            # temporary change: only consider the output in the graph
                            things = [inventory_offset + output]
                        for thing in things:
                            mask[thing, agent_pos_slice] = True
                            mask[thing, agent_dir_slice] = True
                            mask[thing, thing_pos_slice] = True
                            mask[thing, action_idx] = True
                            for i in ing:
                                mask[thing, i] = True

                if self.has_furnace and state["furnace_stage"] != prev_state["furnace_stage"]:
                    mask[furnace_stage_slice, furnace_slot_slice] = True
                    mask[furnace_stage_slice, furnace_stage_slice] = True

            num_factors = len(slice_dict) - 1
            factor_mask = np.zeros((num_factors, num_factors + 1), dtype=bool)
            factor_names = list(self.observation_space.spaces.keys()) + ["action"]
            for i, obs_k_i in enumerate(factor_names[:-1]):
                slice_i = slice_dict[obs_k_i]
                for j, obs_k_j in enumerate(factor_names):
                    slice_j = slice_dict[obs_k_j]
                    factor_mask[i, j] = mask[slice_i, slice_j].any()

        self.cur_step += 1
        terminate = False
        truncate = self.cur_step >= self.horizon

        reward = self.reward()
        info = {"success": self.check_success(),
                "stage_completion": self.stage_completion_tracker}
        if mask is not None:
            info["variable_graph"] = mask
            info["factor_graph"] = factor_mask

        return self.get_state(), reward, terminate, truncate, info

    def next_to(self, i_kind):
        x, y = self.pos
        return self.grid[x - 1:x + 2, y - 1:y + 2, i_kind].any()

    def render(self):
        h, w = self.height, self.width
        cell_w = 3

        # First row
        print(" " * (cell_w + 1), end='')
        for i in range(w):
            print("| {:^{}d} ".format(i, cell_w), end='')
        print("| ")
        print((w * (cell_w + 3) + cell_w + 2) * "-")

        # Other rows
        for j in reversed(range(h)):
            print("{:{}d} ".format(j, cell_w), end='')

            for i in range(w):
                symbol = ""
                if (i, j) == self.pos:
                    if self.dir == LEFT:
                        symbol = u"\u2190"
                    elif self.dir == RIGHT:
                        symbol = u"\u2192"
                    elif self.dir == UP:
                        symbol = u"\u2191"
                    elif self.dir == DOWN:
                        symbol = u"\u2193"
                elif self.grid[i, j].any():
                    thing = self.grid[i, j].argmax()
                    name = self.cookbook.index[thing]
                    state_key = self.find_key(i, j)
                    if thing in self.primitive_idxes and state_key and state_key[-1].isdigit():
                        symbol = name[:cell_w - 1] + state_key[-1]
                    else:
                        symbol = name[:cell_w]

                assert len(symbol) <= cell_w

                print("| {:^{}} ".format(symbol, cell_w), end='')
            print("| ")
            print((w * (cell_w + 3) + cell_w + 2) * "-")

        print("inventory")
        for i in range(len(self.inventory)):
            if i == self.furnace_idx or i in self.primitive_idxes + self.craft_idxes:
                print("| {} ".format(self.cookbook.index[i]), end='')
        print("| ")
        for i, num in enumerate(self.inventory):
            if i == self.furnace_idx or i in self.primitive_idxes + self.craft_idxes:
                print("| {:^{}} ".format(num, len(self.cookbook.index[i])), end='')
        print("| ")

        if self.has_furnace:
            print("furnace")
            slot_msg = step_msg = ""
            slot = self.state["furnace_slot"]
            if slot:
                craft_idx = self.cookbook.furnace_slot2idx[slot]
                slot_msg = self.cookbook.index[craft_idx]
                total_stage = self.recipes[craft_idx]["_step"]
                step_msg = "{}/{}".format(self.state["furnace_stage"], total_stage)

            msgs = [["ready", "Y" if self.state["furnace_ready"] else "N"],
                    ["slot", slot_msg],
                    ["step", step_msg]]
            for i in range(len(msgs[0])):
                for ele in msgs:
                    ele_w = max([len(e) for e in ele])
                    print("| {:^{}} ".format(ele[i], ele_w), end='')
                print("| ")
            print()


if __name__ == "__main__":
    env = CraftWorld(goal="gold", width=10, height=10, must_craft_at_workspace=False)
    actions = ["DOWN", "UP", "LEFT", "RIGHT", "USE"] + [env.cookbook.index.idx2name[i] for i in
                                                        env.craft_idxes]
    for _ in range(100):
        env.reset()
        print("state")
        env.render()
        while True:
            action = np.random.randint(env.action_dim)
            print("action")
            print(actions[action])
            state, reward, _, done, info = env.step(action)
            print("next state")
            env.render()
            factor_mask = info["factor_graph"]
            factor_names = list(env.observation_space.spaces.keys()) + ["action"]
            num_factors = len(factor_names) - 1
            if (factor_mask != np.eye(*factor_mask.shape, dtype=bool)).any():
                for name, row in zip(factor_names[:-1], factor_mask):
                    if row.sum() > 1:
                        parent = ", ".join([factor_names[i] for i in range(num_factors + 1) if row[i]])
                        print(f"{parent} -> {name}")
                # import pdb; pdb.set_trace()
            if done:
                break
