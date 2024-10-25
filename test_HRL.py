import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from Initializers.data import initialize_data
from Initializers.init_utils import init_logistics, init_loading
from Initializers.model import initialize_models


@hydra.main(version_base=None, config_path="./configs", config_name="test_config")
def test_HRL(config: DictConfig):  # does the exact same thing as training until accessing the trainer
    config = OmegaConf.structured(OmegaConf.to_yaml(config))
    single_env, train_env, test_env, logger, config = init_logistics(config, saving=False, training=False)

    # Similar to code in tianshou.examples
    config.device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")

    dynamics, graph_encoding, diayn_discriminator, hrl_policy = initialize_models(config, single_env)
    test_collector = initialize_data(config, hrl_policy, dynamics, graph_encoding, single_env, train_env, test_env, train=False)

    init_loading(config,
                 dynamics,
                 hrl_policy,
                 None,
                 None)

    # Reuses config.train.test_ep_per_epoch to run that number of test episodes
    hrl_policy.eval()
    # test_collector.collect(n_episode=config.train.test_ep_per_epoch, render=0.01)
    test_collector.collect(n_episode=config.train.test_ep_per_epoch)


if __name__ == "__main__":
    test_HRL()
