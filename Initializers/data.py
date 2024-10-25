from State.buffer import VectorHierarchicalReplayBufferManager, HierarchicalReplayBuffer
from State.collector import HRLCollector


def initialize_data(config, hrl_policy, dynamics, extractor, single_env, env, test_env, train=True):

    test_collector = HRLCollector(hrl_policy, dynamics, test_env,
                                  exploration_noise=False, name="test", root_dir=config.exp_path,
                                  num_factors=config.num_factors,
                                  upper_rew_aggregation=config.option.upper.rew_aggregation,
                                  save_gif_num=config.save.save_gif_num, scripted_lower=config.policy.lower.scripted)
    if not train:
        return test_collector

    upper_buffer = HierarchicalReplayBuffer(
        True,
        config.data.buffer_len,
        use_her=config.data.her.upper.use_her,
        horizon=config.data.her.upper.her_horizon,
        future_k=config.data.her.upper.future_k,
        alpha=config.data.prio.upper_policy.alpha,
        beta=config.data.prio.beta,
        decay_window=config.data.prio.upper_policy.decay_window,
        decay_rate=config.data.prio.upper_policy.decay_rate,
        max_prev_decay=config.data.prio.upper_policy.max_prev_decay,
        weight_norm=config.data.prio.weight_norm,
    )
    # TODO: allow for multi-buffer handling if multiple lower policies
    lower_buffer = VectorHierarchicalReplayBufferManager(
        single_env,
        config.data.buffer_len,
        extractor=extractor,
        buffer_num=len(env),
        num_lower_policies=config.num_lower_policies,
        lower_timeout=config.option.timeout,
        count_threshold_for_valid_graph=config.data.count_threshold_for_valid_graph,
        # HER params
        use_her=config.data.her.lower.use_her,
        her_use_count_select_goal=config.data.her.lower.her_use_count_select_goal,
        horizon=config.episode_length if config.option.lower.training_ignore_lower_done else config.option.timeout,
        future_k=float('inf') if config.option.lower.training_ignore_lower_done else config.data.her.lower.future_k,
        lower_reached_graph_threshold=config.option.lower.reached_graph_threshold,
        # PER params
        policy_per_alpha=config.data.prio.lower_policy.alpha,
        dynamics_per_alpha=config.data.prio.dynamics.alpha,
        beta=config.data.prio.beta,
        weight_norm=config.data.prio.weight_norm,
        dynamics_per_pred_error_scale=config.data.prio.dynamics.pred_error_scale,
        dynamics_per_change_count_scale=config.data.prio.dynamics.change_count_scale,
        policy_per_td_error_scale=config.data.prio.lower_policy.td_error_scale,
        policy_per_graph_count_scale=config.data.prio.lower_policy.graph_count_scale,
        policy_per_graph_count_power=config.data.prio.lower_policy.graph_count_power,
        # PSER params
        decay_window=config.data.prio.lower_policy.decay_window,
        decay_rate=config.data.prio.lower_policy.decay_rate,
        max_prev_decay=config.data.prio.lower_policy.max_prev_decay,
    )

    # collectors
    train_collector = HRLCollector(hrl_policy, dynamics, env, lower_buffer, upper_buffer,
                                   exploration_noise=True,
                                   name="train",
                                   num_factors=config.num_factors,
                                   lower_training_ignore_lower_done=config.option.lower.training_ignore_lower_done,
                                   reached_graph_threshold=config.option.lower.reached_graph_threshold,
                                   upper_rew_aggregation=config.option.upper.rew_aggregation,
                                   scripted_lower=config.policy.lower.scripted)

    return train_collector, test_collector, lower_buffer, upper_buffer
