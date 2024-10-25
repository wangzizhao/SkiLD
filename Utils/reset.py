from copy import deepcopy

from Causal import Dynamics
from Networks import DiaynDiscriminator
from Option.Terminate import RTTLowerGraph


def reset_module(model, name_filter_fn=lambda x: False, type_filter_fn=lambda x: False, copy_pair=None):
    for name, module in model.named_children():
        if name_filter_fn(name):
            continue
        if type_filter_fn(module):
            continue

        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        reset_module(module, name_filter_fn, type_filter_fn, copy_pair)

    if isinstance(model, RTTLowerGraph):
        model.reset_rew_schedule_and_adaption()

    # target network should be initialized with the same parameters as the source network
    # TODO: only handle rainbow for now, need to add for SAC, TD3, etc.
    if copy_pair is not None:
        for source_name, target_name in copy_pair:
            if hasattr(model, source_name) and hasattr(model, target_name):
                source_module = getattr(model, source_name)
                target_module = deepcopy(source_module).eval()
                setattr(model, target_name, target_module)


def reset_upper_model(hrl_policy, epoch):
    type_filter_fn = lambda x: isinstance(x, (Dynamics, DiaynDiscriminator))
    reset_module(hrl_policy.upper, type_filter_fn=type_filter_fn)
    hrl_policy.last_upper_reset_epoch = epoch


def reset_lower_model(config, hrl_policy, lower_replay_buffer, epoch, env_step):
    if config.train.reset.lower.reset_diayn:
        type_filter_fn = lambda x: isinstance(x, Dynamics)
    else:
        type_filter_fn = lambda x: isinstance(x, (Dynamics, DiaynDiscriminator))
    reset_module(hrl_policy.lower, type_filter_fn=type_filter_fn, copy_pair=[("model", "model_old")])

    # need to go through all the trackers and reset the parameters
    # lower_rtt: handled in reset_module
    # lower_buffer: reset PER weights
    lower_replay_buffer.reset_policy_PER()

    hrl_policy.lower_reset_env_step = env_step
    hrl_policy.last_lower_reset_epoch = epoch
