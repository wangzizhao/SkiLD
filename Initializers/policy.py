from typing import Union, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Independent, Normal
from omegaconf import DictConfig
from tianshou.exploration import GaussianNoise
from tianshou.policy import DDPGPolicy, TD3Policy, SACPolicy
from tianshou.policy import DQNPolicy, RainbowPolicy
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.continuous import Actor as ContActor, ActorProb, Critic as ContCritic
from tianshou.utils.net.discrete import Actor as DisActor, Critic as DisCritic

from Networks.network import Rainbow
from Networks.ppo_w_mask import PPOPolicyMaskEnabled
from Networks.gen_dqn_network import GenDQN, GenRainbow
from Networks.gen_actor_critic_network import GenNet


class InplaceOperator(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation(x)


def crelu(x, dim=1):
    return torch.cat((F.relu(x), F.relu(-x)), dim)


class CReLU(nn.Module):
    def __init__(self, dim=1):
        super(CReLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), self.dim)


class SinModule(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class SincModule(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return torch.sinc(x)


class TanhModule(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)


class CosModule(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return torch.cos(x)


def get_inplace_acti(acti):
    if acti == "relu":
        return nn.ReLU
    elif acti == "leakyrelu":
        return nn.LeakyReLU
    elif acti == "sin":
        return SinModule
    elif acti == "sinc":
        return SincModule
    elif acti == "sigmoid":
        return nn.Sigmoid
    elif acti == "tanh":
        return TanhModule
    elif acti == "cos":
        return CosModule
    elif acti == "none":
        return nn.Identity
    elif acti == "prelu":
        return nn.PReLU
    elif acti == "crelu":
        return CReLU


def get_normalization(norm_form_str):
    if norm_form_str == "none":
        return None
    elif norm_form_str == "layer":
        return nn.LayerNorm


def get_activation(acti_form_str):
    return get_inplace_acti(acti_form_str)


def initialize_policy(
        algo: str,
        input_dim: int,
        action_space: gym.Space,
        policy_config: DictConfig,
        decorator: Callable,
        device: torch.device,
        net_args: dict = None,
) -> Union[DQNPolicy, RainbowPolicy, DDPGPolicy, TD3Policy, SACPolicy, PPOPolicyMaskEnabled]:
    if algo == "rainbow":
        initialize_fn = initialize_rainbow
    elif algo == "ddpg":
        initialize_fn = initialize_ddpg
    elif algo == "td3":
        initialize_fn = initialize_td3
    elif algo == "sac":
        initialize_fn = initialize_sac
    elif algo == "ppo":
        initialize_fn = initialize_ppo
    else:
        raise NotImplementedError(f"unknown algo name: {algo}")
    return initialize_fn(input_dim, action_space, policy_config, decorator, device, net_args=net_args)


def initialize_rainbow(
        input_dim: int,
        action_space: Union[gym.spaces.Discrete, gym.spaces.MultiDiscrete],
        policy_config: DictConfig,
        decorator: Callable,
        device: torch.device,
        net_args: dict = None,
) -> RainbowPolicy:
    rainbow_config = policy_config.rainbow
    action_shape = action_space.n

    norm_layer = get_normalization(policy_config.rainbow.norm_form)
    activation = get_activation(policy_config.rainbow.acti_form)
    hidden_sizes = [int(hs) for hs in policy_config.rainbow.hidden_sizes]

    if net_args is None:
        net = decorator(Rainbow)(input_dim,
                                 action_shape,
                                 hidden_sizes,
                                 rainbow_config.num_atoms,
                                 device=device,
                                 norm_layer=norm_layer,
                                 activation=activation).to(device)
    else:
        net_args.num_inputs = input_dim
        net_args.input_dim = net_args.factor.object_dim
        net_args.num_outputs = hidden_sizes[-1]
        net_args.object_dim = net_args.factor.object_dim
        net_args.hidden_sizes = hidden_sizes
        net = decorator(GenRainbow)(input_dim,
                                    action_shape,
                                    net_args,
                                    rainbow_config.num_atoms,
                                    device=device,
                                    norm_layer=norm_layer,
                                    activation=activation).to(device)

    net_type = Rainbow if net_args is None else GenRainbow
    optim = torch.optim.Adam(net.parameters(), lr=rainbow_config.lr)
    # define policy
    policy = RainbowPolicy(
        net,
        optim,
        policy_config.gamma,
        rainbow_config.num_atoms,
        rainbow_config.v_min,
        rainbow_config.v_max,
        policy_config.n_step,
        rainbow_config.target_update_freq,
        action_space=action_space,
    )
    policy.set_eps(rainbow_config.eps)
    return policy


def initialize_actor_critic(
        input_dim: int,
        action_space: gym.Space,
        algorithm_config: DictConfig,
        decorator: Callable,
        device: torch.device,
        double: bool,
        use_actor_prob: bool,
        combine_optim: bool = False,
        q_critic: bool = True,
        net_args: dict = None,  # network arguments if using a GenNet
):
    if combine_optim:
        assert not double
    if q_critic:
        assert not isinstance(action_space, gym.spaces.Discrete)

    norm_layer = get_normalization(algorithm_config.norm_form)
    activation = get_activation(algorithm_config.acti_form)
    hidden_sizes = [int(hs) for hs in algorithm_config.hidden_sizes]

    if net_args is not None:
        net_args.hidden_sizes = hidden_sizes
        net_a = decorator(GenNet)(input_dim, net_args, device=device)
    else:
        net_a = decorator(Net)(input_dim, hidden_sizes=hidden_sizes, norm_layer=norm_layer, activation=activation, device=device)
    if isinstance(action_space, gym.spaces.Box):
        if use_actor_prob:
            actor = decorator(ActorProb)(net_a, action_space.shape,
                                         unbounded=True,
                                         conditioned_sigma=algorithm_config.conditioned_sigma,
                                         device=device).to(device)
        else:
            actor = decorator(ContActor)(net_a, action_space.shape, device=device).to(device)
    elif isinstance(action_space, gym.spaces.Discrete):
        actor = decorator(DisActor)(net_a, action_space.n, device=device, softmax_output=False).to(device)
    else:
        raise NotImplementedError("only Box and Discrete action space are supported")

    net_c = decorator(Net)(
        input_dim,
        action_space.shape or action_space.n,
        hidden_sizes=hidden_sizes,
        concat=q_critic,
        device=device,
    )
    if isinstance(action_space, gym.spaces.Box):
        critic = decorator(ContCritic)(net_c, device=device).to(device)
    elif isinstance(action_space, gym.spaces.Discrete):
        critic = decorator(DisCritic)(net_c, device=device).to(device)
    else:
        raise NotImplementedError("only Box and Discrete action space are supported")

    if combine_optim:
        actor_critic = ActorCritic(actor, critic)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=algorithm_config.lr)
        return actor, critic, optim

    actor_optim = torch.optim.Adam(actor.parameters(), lr=algorithm_config.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=algorithm_config.critic_lr)
    if not double:
        return actor, actor_optim, critic, critic_optim

    net_c2 = decorator(Net)(
        input_dim,
        action_space.shape,
        hidden_sizes=algorithm_config.hidden_sizes,
        concat=q_critic,
        device=device,
    )
    assert isinstance(action_space, gym.spaces.Box)
    critic2 = decorator(ContCritic)(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=algorithm_config.critic_lr)
    return actor, actor_optim, critic, critic_optim, critic2, critic2_optim


def initialize_ddpg(
        input_dim: int,
        action_space: gym.Space,
        policy_config: DictConfig,
        decorator: Callable,
        device: torch.device,
        net_args: dict = None,
) -> DDPGPolicy:
    ddqg_config = policy_config.ddpg
    actor, actor_optim, critic, critic_optim = \
        initialize_actor_critic(input_dim,
                                action_space,
                                ddqg_config,
                                decorator,
                                device,
                                double=False,
                                use_actor_prob=False,
                                combine_optim=False,
                                net_args=net_args, )
    policy = DDPGPolicy(
        actor,
        actor_optim,
        critic,
        critic_optim,
        tau=ddqg_config.tau,
        gamma=policy_config.gamma,
        exploration_noise=GaussianNoise(sigma=ddqg_config.exploration_noise),
        estimation_step=policy_config.n_step,
        action_space=action_space,
    )
    return policy


def initialize_td3(
        input_dim: int,
        action_space: gym.Space,
        policy_config: DictConfig,
        decorator: Callable,
        device: torch.device,
        net_args: dict = None,
) -> TD3Policy:
    td3_config = policy_config.td3
    actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim = \
        initialize_actor_critic(input_dim,
                                action_space,
                                td3_config,
                                decorator,
                                device,
                                double=True,
                                use_actor_prob=False,
                                combine_optim=False,
                                net_args=net_args, )
    policy = TD3Policy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=td3_config.tau,
        gamma=policy_config.gamma,
        exploration_noise=GaussianNoise(sigma=td3_config.exploration_noise),
        policy_noise=td3_config.policy_noise,
        update_actor_freq=td3_config.update_actor_freq,
        noise_clip=td3_config.noise_clip,
        estimation_step=policy_config.n_step,
        action_space=action_space,
    )
    return policy


def initialize_sac(
        input_dim: int,
        action_space: gym.Space,
        policy_config: DictConfig,
        decorator: Callable,
        device: torch.device,
        net_args: dict = None,
) -> SACPolicy:
    sac_config = policy_config.sac
    actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim = \
        initialize_actor_critic(input_dim,
                                action_space,
                                sac_config,
                                decorator,
                                device,
                                double=True,
                                use_actor_prob=True,
                                combine_optim=False,
                                net_args=net_args, )
    if sac_config.auto_alpha:
        target_entropy = -np.prod(action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=sac_config.alpha_lr)
        sac_config.alpha = (target_entropy, log_alpha, alpha_optim)
    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=sac_config.tau,
        gamma=policy_config.gamma,
        alpha=sac_config.alpha,
        estimation_step=policy_config.n_step,
        action_space=action_space,
    )
    return policy


def initialize_ppo(
        input_dim: int,
        action_space: gym.Space,
        policy_config: DictConfig,
        decorator: Callable,
        device: torch.device,
        net_args: dict = None,
) -> PPOPolicyMaskEnabled:
    ppo_config = policy_config.ppo
    actor, critic, optim = initialize_actor_critic(input_dim,
                                                   action_space,
                                                   ppo_config,
                                                   decorator,
                                                   device,
                                                   double=False,
                                                   use_actor_prob=True,
                                                   combine_optim=True,
                                                   q_critic=False,
                                                   net_args=net_args, )
    if isinstance(action_space, gym.spaces.Discrete):
        def dist(probs):
            return Categorical(probs=probs)
    elif isinstance(action_space, gym.spaces.Box):
        def dist(*logits):
            return Independent(Normal(*logits), 1)
    else:
        raise NotImplementedError

    policy = PPOPolicyMaskEnabled(
        actor,
        critic,
        optim,
        dist,
        discount_factor=policy_config.gamma,
        eps_clip=ppo_config.eps_clip,
        value_clip=True,
        dual_clip=None,
        advantage_normalization=True,
        recompute_advantage=ppo_config.recompute_advantage,
        gae_lambda=ppo_config.gae_lambda,
        max_grad_norm=ppo_config.max_grad_norm,
        vf_coef=ppo_config.vf_coef,
        ent_coef=ppo_config.ent_coef,
        reward_normalization=ppo_config.rew_norm,
        action_scaling=True,
        action_space=action_space,
    )
    return policy
