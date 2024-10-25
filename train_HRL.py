import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from Initializers.data import initialize_data
from Initializers.init_utils import init_logistics, init_loading
from Initializers.model import initialize_models
from Option.ihrl_trainer import IHRLTrainer
from Option.Terminate import RTTUpperGraphCount

from Utils.reset import reset_upper_model, reset_lower_model


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def train_HRL(config: DictConfig):
    config = OmegaConf.structured(OmegaConf.to_yaml(config))
    single_env, train_env, test_env, logger, config = init_logistics(config)

    # Similar to code in tianshou.examples
    config.device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")

    dynamics, extractor, diayn_discriminator, hrl_policy = initialize_models(config, single_env)
    train_collector, test_collector, lower_replay_buffer, upper_replay_buffer = \
        initialize_data(config, hrl_policy, dynamics, extractor, single_env, train_env, test_env)

    lower_replay_buffer, upper_replay_buffer = init_loading(config,
                                                            dynamics,
                                                            hrl_policy,
                                                            lower_replay_buffer,
                                                            upper_replay_buffer)
    lower_replay_buffer.has_reset_graph_count = config.dynamics.type == "gt"

    # Need to assign the buffer to the collector
    train_collector.buffer = lower_replay_buffer
    train_collector.upper_buffer = upper_replay_buffer
    for rtt_func in hrl_policy.upper.rewtermdone.rtt_functions:
        if isinstance(rtt_func, RTTUpperGraphCount):
            rtt_func.buffer = lower_replay_buffer
            rtt_func.update_graph_count()

    def save_best_fn(policy):
        torch.save(policy.state_dict(), config.exp_path / "policy.pth")

    def stop_fn(mean_rewards):
        return False

    def train_fn(epoch, env_step):
        if config.train.mode == "task_learning_lower_frozen":
            hrl_policy.train_upper(True)
            hrl_policy.train_lower(False)
            return

        if env_step < config.train.dynamics_warmup_step:
            train_collector.random_all = True
            train_collector.random_upper_only = False
            hrl_policy.train_upper(False)
            hrl_policy.train_lower(False)
        else:
            if not lower_replay_buffer.has_reset_graph_count:
                lower_replay_buffer.reset_graph_count()
                lower_replay_buffer.has_reset_graph_count = True

            # dynamics is updated enough to generate (relatively) good graphs to put into history
            hrl_policy.upper.history_update_ready = True

            # do not update upper before lower converges
            lower_reset_env_step = hrl_policy.lower_reset_env_step
            lower_warmup_env_step = config.train.reset.lower.warmup_env_step
            if env_step < config.train.init_upper_random_step or env_step < (lower_reset_env_step + lower_warmup_env_step):
                train_collector.random_all = False
                train_collector.random_upper_only = True
                hrl_policy.train_upper(False)
                hrl_policy.train_lower(True, lower_replay_buffer)
            else:
                train_collector.random_all = False
                train_collector.random_upper_only = False
                hrl_policy.train_upper(True, upper_replay_buffer)
                hrl_policy.train_lower(True, lower_replay_buffer)

    def test_fn(epoch, env_step):
        # This is used for naming the gif file
        test_collector.epoch_num = epoch

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        if epoch == 0 or epoch % config.save.save_freq != 0:
            return None
        print("saving checkpoints")
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = config.exp_path / f"policy_{epoch}.pth"
        torch.save(hrl_policy.state_dict(), ckpt_path)

        upper_reset_freq = config.train.reset.upper.reset_freq
        lower_reset_freq = config.train.reset.lower.reset_freq
        if upper_reset_freq and epoch - hrl_policy.last_upper_reset_epoch >= upper_reset_freq:
            reset_upper_model(hrl_policy, epoch)

        if lower_reset_freq and epoch - hrl_policy.last_lower_reset_epoch >= lower_reset_freq:
            reset_lower_model(config, hrl_policy, lower_replay_buffer, epoch, env_step)

        if config.save.save_replay_buffer:
            print("saving replaybuffer")
            lower_buffer_path = config.replay_buffer_dir / "lower_buffer.hdf5"
            higher_buffer_path = config.replay_buffer_dir / "upper_buffer.hdf5"
            train_collector.buffer.save_hdf5(lower_buffer_path)
            train_collector.upper_buffer.save_hdf5(higher_buffer_path)
        return ckpt_path

    if config.load.load_rpb:
        print("skipping initial data collection because of replay buffer loading")
    elif config.train.init_random_step > 0:
        # start filling replay buffer
        train_collector.collect(n_step=config.train.init_random_step, random=True)
        print("finished initial data collection")

    result = IHRLTrainer(
        hrl_policy,
        diayn_discriminator,
        dynamics,
        train_collector,
        test_collector,
        config.train.epoch,
        config.train.env_step_per_epoch,
        config.train.env_step_per_collect,
        config.train.policy_update_per_env_step,
        config.train.diayn_update_per_env_step,
        config.train.dynamics_update_per_env_step,
        config.train.upper_policy_batch_size,
        config.train.lower_policy_batch_size,
        config.train.diayn_batch_size,
        config.train.dynamics_batch_size,
        config.train.test_ep_per_epoch,
        train_fn=train_fn,
        dynamics_warmup_step=config.train.dynamics_warmup_step,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        # resume_from_log=config.load.resume_from_log,  # Todo: this requires setting the logger correctly
        logger=logger
    ).run()


if __name__ == "__main__":
    train_HRL()
