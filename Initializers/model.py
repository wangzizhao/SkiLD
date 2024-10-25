from typing import Tuple, Union, Dict

import sys, copy, os
import torch
from tianshou.utils.net.common import get_dict_state_decorator

import gymnasium as gym

from Causal import Dynamics, GroundTruthGraph, DynamicsGrad, DynamicsAC

from Initializers.init_utils import AttrDict, get_preprocess
from State.extractor import Extractor
from Option.hierarchical_model import HierarchicalModel
from Option.Upper import UpperPolicy, DiaynSamplerUpper, ModularUpper, WideUpper
from Option.Lower import LowerPolicy, SingleGraphLowerPolicy, WideLowerPolicy, FactorLowerPolicy
from Option.Terminate import RewardTermTruncManager
from Option.Terminate import RTTUpperTask, RTTUpperGraphCount, RTTUpperSpecialGraph
from Option.Terminate import RTTLowerGraph

from Networks import GraphEncoding, DiaynDiscriminator
from Initializers.policy import initialize_policy


def initialize_dynamics(config: AttrDict, env: gym.Env, extractor: Extractor) -> Dynamics:
    if config.dynamics.type == "grad":
        # overwrite: policy will only using random upper action until dynamics starts training
        config.train.init_upper_random_step = max(config.train.init_upper_random_step, config.train.dynamics_warmup_step)
        dynamics = DynamicsGrad(env, extractor, config)
    elif config.dynamics.type == "gt":
        config.train.dynamics_warmup_step = 0
        dynamics = GroundTruthGraph(env, extractor)
    elif config.dynamics.type == "ac":
        config.train.dynamics_warmup_step = max(config.train.init_upper_random_step, config.train.dynamics_warmup_step)
        config.train.init_random_step = max(config.train.init_random_step, config.train.dynamics_pretrain_step)
        dynamics = DynamicsAC(env, extractor, config)
    else:
        raise NotImplementedError

    return dynamics


def initialize_diayn_discriminator(
        config: AttrDict,
        extractor,
        device: torch.device):
    diayn_discriminator = None
    upper_config = config.policy.upper

    if upper_config.goal_type != "diayn":
        return None

    # initialize the diayn
    graph_type = upper_config.graph_type
    num_factors = config.num_factors
    num_edge_classes = config.graph_encoding.num_edge_classes
    if graph_type == "graph":
        graph_action_size = num_factors * (num_factors + 1) * num_edge_classes
        goal_size = config.obs_size
    elif graph_type == "factor":
        graph_action_size = num_factors + (num_factors + 1) * num_edge_classes
        goal_size = extractor.longest
    elif graph_type == "none":
        graph_action_size = 0
        goal_size = config.obs_size
    else:
        raise NotImplementedError(f"unknown graph encoding type: {graph_type}")

    if upper_config.goal_type == "diayn":
        diayn_discriminator = DiaynDiscriminator(config, goal_size, graph_action_size, device).to(device)
    return diayn_discriminator


def initialize_upper_policy(
        config: AttrDict,
        env: gym.Env,
        extractor: Extractor,
        dynamics: Dynamics,
        diayn_discriminator: Union[DiaynDiscriminator, None] = None,
        net_args: Dict = None,
) -> Tuple[UpperPolicy, int, RewardTermTruncManager]:
    upper_config = config.policy.upper
    upper_type = upper_config.type

    num_factors = config.num_factors
    if upper_type in ["modular", "wide"]:
        # compute policy action size and upper policy action size
        goal_type = upper_config.goal_type
        graph_type = upper_config.graph_type
        graph_action_space = upper_config.graph_action_space
        goal_learned = upper_config.goal_learned

        if graph_type == "graph":
            graph_action_size = num_factors * (num_factors + 1) * config.graph_encoding.num_edge_classes
        elif graph_type == "factor":
            graph_action_size = num_factors + (num_factors + 1) * config.graph_encoding.num_edge_classes
        elif graph_type == "none":
            graph_action_size = 0
        else:
            raise NotImplementedError(f"unknown graph type: {graph_type}")

        if goal_type == "value":
            assert upper_config.graph_type == "factor"
            assert (env.observation_space.low >= -1).all()
            assert (env.observation_space.high <= 1).all()
            if graph_type == "graph":
                goal_action_size = config.obs_size
            elif graph_type == "factor":
                goal_action_size = extractor.longest
            else:
                raise NotImplementedError(f"unknown graph encoding type: {graph_type}")
        elif goal_type == "diayn":
            goal_action_size = upper_config.diayn.num_classes
        else:
            raise NotImplementedError(f"unknown goal type: {goal_type}")

        upper_action_size = graph_action_size + goal_action_size

        if graph_type == "none" or graph_action_space == "sample_from_history":
            if config.train.mode == "task_learning_lower_frozen":
                upper_policy_action_size = config.policy.upper.diayn.num_classes
            else:
                upper_policy_action_size = 0
        elif graph_action_space == "choose_from_history":
            upper_policy_action_size = upper_config.sample_action_space_n
        else:
            raise NotImplementedError(f"unknown graph action space type: {graph_action_space}")

        if upper_type == "wide":  # set the upper policy action size here, TODO: might not handle all action spaces
            # adds num_factors for the option selection
            # multiplies by num_factors to output desired goals for all factors
            upper_policy_action_size = upper_policy_action_size * num_factors + config.num_factors

        if goal_learned:
            upper_policy_action_size += goal_action_size
            raise NotImplementedError("goal_learned is not supported yet")

        # initialize policy
        if upper_policy_action_size > 0:
            if graph_action_space == "choose_from_history":
                algo = upper_config.discrete_algo
                if config.train.mode == "skill_learning":
                    action_space = gym.spaces.Discrete(upper_policy_action_size)
                elif config.train.mode == "task_learning_lower_frozen":
                    assert upper_config.goal_type == "diayn", "only diayn is supported for task learning for now"
                    action_space = gym.spaces.Discrete(upper_policy_action_size)
                else:
                    raise NotImplementedError(f"unknown train mode: {config.train.mode}")
            else:
                algo = upper_config.continuous_algo
                action_space = gym.spaces.Box(low=-1, high=1, shape=(upper_policy_action_size,))

            if config.goal_based:
                dict_state_dec, flat_state_shape = get_dict_state_decorator(
                    state_shape={"observation": env.observation_space.shape[0], "desired_goal": extractor.longest},
                    keys=["observation", "desired_goal"]  # TODO: might need achieved goal at some point later
                )
            else:
                config.data.her.upper.use_her = False
                dict_state_dec, flat_state_shape = get_dict_state_decorator(
                    state_shape={"observation": env.observation_space.shape[0]},
                    keys=["observation"]
                )

            # TODO: properly create and generate general network arguments (net_args=)
            policy = initialize_policy(algo, flat_state_shape, action_space, upper_config, dict_state_dec, config.device, net_args=net_args)
        else:
            policy = None

        upper_rew_type = config.policy.upper.reward_type
        if upper_rew_type == "task":
            rtt = RTTUpperTask(timeout=0)
        elif upper_rew_type == "graph_count":
            rtt = RTTUpperGraphCount(timeout=0,
                                     power=config.option.upper.graph_count_power,
                                     use_factor_subgraph=config.option.upper.use_factor_subgraph)
        elif upper_rew_type == "special":
            rtt = RTTUpperSpecialGraph(timeout=0)
        else:
            raise NotImplementedError(f"unknown upper reward type: {upper_rew_type}")

        rewtermdone = RewardTermTruncManager(rtt if isinstance(rtt, list) else [rtt])

        if graph_type == "none":
            upper_policy = DiaynSamplerUpper(policy, dynamics, rewtermdone, config)
        elif config.policy.upper.type == "wide":
            upper_policy = WideUpper(policy, dynamics, rewtermdone, extractor, config)
        else:
            upper_policy = ModularUpper(policy, dynamics, rewtermdone, extractor, config)

        lower_rtt = RTTLowerGraph(config=config, extractor=extractor, diayn_discriminator=diayn_discriminator)
        lower_rewtermdone = RewardTermTruncManager([lower_rtt])

        if isinstance(lower_rtt, RTTLowerGraph) and isinstance(upper_policy, ModularUpper):
            if lower_rtt.adaptive_diayn_coef >= 0:
                # has diayn rew adapter, pass the upper action space to lower rtt to keep track of each graph's success rate
                lower_rtt.upper_unique_graph_from_id = upper_policy.unique_graph_from_id
                lower_rtt.upper_choose_from_history_action_mask = upper_policy.choose_from_history_action_mask
    else:
        raise NotImplementedError(f"unknown upper policy type: {upper_type}")

    return upper_policy, upper_action_size, lower_rewtermdone


def initialize_lower_policy(
        config: AttrDict,
        env: gym.Env,
        upper_action_size: int,
        extractor: Extractor,
        rewtermdone: RewardTermTruncManager,
        dynamics: Dynamics,
        diayn_discriminator: Union[DiaynDiscriminator, None] = None,
        net_args: Dict = None,
) -> LowerPolicy:
    assert isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 1

    state_shape = {"observation": env.observation_space.shape[0], "desired_goal": upper_action_size}
    keys = ["observation", "desired_goal"]
    if config.option.lower.use_reached_graph_counter:
        state_shape["reached_graph_counter"] = 1
        keys.append("reached_graph_counter")
    dict_state_dec, flat_state_shape = get_dict_state_decorator(state_shape, keys)

    preprocess = get_preprocess(state_shape, keys)

    device = config.device
    num_factors = config.num_factors
    lower_config = config.policy.lower
    lower_type = lower_config.type
    action_space = env.action_space

    if isinstance(action_space, gym.spaces.Discrete):
        algo = lower_config.discrete_algo
    elif isinstance(action_space, gym.spaces.Box):
        algo = lower_config.continuous_algo
    else:
        raise NotImplementedError(f"unknown action space type: {type(action_space)}")

    net_args = set_goal_net_args(net_args, flat_state_shape, state_shape["observation"])
    if lower_type == "single_graph":
        policy = initialize_policy(algo, flat_state_shape, action_space, lower_config, dict_state_dec, device, net_args=net_args)
        lower_policy = SingleGraphLowerPolicy([policy], dynamics, diayn_discriminator, rewtermdone,
                                              extractor, action_space, config)
        config.num_lower_policies = 1
    elif lower_type == "wide":
        policies = []
        num_lower_policies = (num_factors + 1) if config.policy.upper.add_count_based_lower else num_factors
        for i in range(num_lower_policies):
            rl_policy = initialize_policy(algo, flat_state_shape, action_space, lower_config, dict_state_dec, device, net_args=net_args)
            policy = FactorLowerPolicy([rl_policy], dynamics, diayn_discriminator, rewtermdone,
                                       extractor, action_space, config, i)
            policy.preprocess = preprocess
            policies.append(policy)
        lower_policy = WideLowerPolicy(policies, dynamics, rewtermdone, extractor, action_space, config)
        config.num_lower_policies = num_lower_policies
    else:
        raise NotImplementedError(f"unknown lower policy type: {lower_type}")
    return lower_policy


def set_goal_net_args(net_args, flat_state_shape, obs_shape):
    if net_args is not None:
        net_args = copy.deepcopy(net_args)
        goal_dim = flat_state_shape - obs_shape
        net_args.factor.single_obj_dim = goal_dim
        net_args.factor.first_obj_dim = goal_dim
        net_args.factor_net.append_keys = True # there is no key separation
        net_args.factor.start_dim = -1
        return net_args
    return None


def initialize_gen_net_args(config, extractor):
    if len(config.policy.net_config_path):
        # slightly redundant if we load the ac path in ac_dynamics also
        if os.path.join(sys.path[0],"Causal", "ac_infer") not in sys.path: sys.path.append(os.path.join(sys.path[0],"Causal", "ac_infer"))
        from Causal.ac_infer.Hyperparam.read_config import read_config
        from State.utils import ObjDict
        all_args = read_config(config.policy.net_config_path)
        net_args = all_args[config.policy.net_config_name]
        # we need to assign the appropriate factored components
        net_args.factor = ObjDict()
        net_args.factor.single_obj_dim = extractor.longest
        net_args.factor.first_obj_dim = extractor.num_factors * extractor.longest
        net_args.factor.object_dim = extractor.longest
        net_args.factor.name_idx = -1
        net_args.factor.query_aggregate = True
        net_args.factor_net.append_keys = False # there is no key separation
        net_args.factor.num_keys = extractor.num_factors
        net_args.factor.num_queries = extractor.num_factors
        net_args.factor.start_dim = 0
        net_args.aggregate_final = True # policy space is never factor dependent
        net_args.gpu = config.cuda_id
        net_args.extractor = extractor # needs the extractor to pad the state
        return net_args
    return None


def initialize_models(config: AttrDict, env: gym.Env):
    """
    Initializes the hierarchical model and interaction model based on the environment
    This includes initializing components (such as TS policies) which are sub-modules 
    of each component
    """
    device = config.device
    extractor = Extractor(env)
    net_args = initialize_gen_net_args(config, extractor)
    dynamics = initialize_dynamics(config, env, extractor).to(device)
    diayn_discriminator = initialize_diayn_discriminator(config, extractor, device)

    upper_policy, upper_action_size, lower_rewtermdone = initialize_upper_policy(config, env,
                                                                                 extractor,
                                                                                 dynamics,
                                                                                 diayn_discriminator,
                                                                                 net_args=net_args)
    upper_policy.to(device)

    lower_policy = initialize_lower_policy(config, env,
                                           upper_action_size, extractor,
                                           lower_rewtermdone, dynamics, diayn_discriminator, net_args=net_args)
    lower_policy.to(device)

    policy = HierarchicalModel(config, lower_policy, upper_policy, env.action_space).to(device)

    return dynamics, extractor, diayn_discriminator, policy

