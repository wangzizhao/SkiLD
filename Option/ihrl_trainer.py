import numpy as np
import time

from typing import Any, Callable, Dict, Optional, Union, Tuple
from tianshou.policy import BasePolicy
from tianshou.trainer.base import BaseTrainer
from tianshou.utils import BaseLogger, LazyLogger

from Option.hierarchical_model import HierarchicalModel
from Networks import DiaynDiscriminator
from Causal import Dynamics
from State.collector import HRLCollector
from Utils.logger import HRLLogger


class IHRLTrainer(BaseTrainer):
    """Create an iterator wrapper for interaction on-policy training procedure.
    THis code matches that for the OffpolicyTrainer in Tianshou, except for adding
    support for training the Interaction models on every epoch

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing. If it's None,
        then no testing will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is
        set.
    :param int env_step_per_epoch: the number of env transitions per epoch.
    :param int env_step_per_collect: the number of transitions the collector would
        collect before the network update, i.e., trainer will collect
        "env_step_per_collect" transitions and do some policy network update repeatedly
        in each epoch.
    :param int/float policy/dynamics_update_per_env_step: the number of times the policy/dynamics
        would be updated after (env_step_per_collect) transitions are collected,
        e.g., if policy/dynamics_update_per_env_step set to 0.3, and env_step_per_collect is 256
        , policy/dynamics will be updated round(256 * 0.3 = 76.8) = 77 times after 256
        transitions are collected by the collector. Default to 1.
    :param int batch_size: the batch size of sample data, which is going to feed in
        the policy network.
    :param test_ep_per_epoch: the number of episodes for one policy evaluation, the
        evaluation happens at every epoch
    :param function train_fn: a hook called at the beginning of training in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function save_best_fn: a hook called when the undiscounted average mean
        reward in evaluation phase gets better, with the signature
        ``f(policy: BasePolicy) ->  None``. It was ``save_fn`` previously.
    :param function save_checkpoint_fn: a function to save training process and
        return the saved checkpoint path, with the signature ``f(epoch: int,
        env_step: int, gradient_step: int) -> str``; you can save whatever you want.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata
        from existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature
        ``f(rewards: np.ndarray with shape (num_episode, agent_num)) ->
        np.ndarray with shape (num_episode,)``, used in multi-agent RL. We need to
        return a single scalar for each episode's result to monitor training in the
        multi-agent RL setting. This function specifies what is the desired metric,
        e.g., the reward of agent 1 or the average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool show_progress: whether to display a progress bar when training.
        Default to True.
    :param bool test_in_train: whether to test in the training phase.
        Default to True.
    """

    __doc__ = BaseTrainer.gen_doc("onpolicy") + "\n".join(__doc__.split("\n")[1:])

    def __init__(
        self,
        policy: HierarchicalModel,
        diayn_discriminator: DiaynDiscriminator,
        dynamics: Dynamics,
        train_collector: HRLCollector,
        test_collector: Optional[HRLCollector],
        max_epoch: int,
        env_step_per_epoch: int,
        env_step_per_collect: int,
        policy_update_per_env_step: Union[int, float] = 1,
        diayn_update_per_env_step: Union[int, float] = 1,
        dynamics_update_per_collect: Union[int, float] = 1,
        upper_policy_batch_size: int = 64,
        lower_policy_batch_size: int = 64,
        diayn_batch_size: int = 64,
        dynamics_batch_size: int = 64,
        test_ep_per_epoch: Union[int, float] = 1,
        dynamics_warmup_step: int = 0,
        train_fn: Optional[Callable[[int, int], None]] = None,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: HRLLogger = LazyLogger(),
        verbose: bool = True,
        show_progress: bool = True,
        test_in_train: bool = True,
        print_logs: bool = True,
        **kwargs: Any,
    ):
        self.diayn_discriminator = diayn_discriminator
        self.dynamics = dynamics
        self.print_logs = True

        self.policy_update_per_env_step = policy_update_per_env_step
        self.diayn_update_per_env_step = diayn_update_per_env_step
        self.dynamics_update_per_collect = dynamics_update_per_collect
        self.upper_policy_batch_size = upper_policy_batch_size
        self.lower_policy_batch_size = lower_policy_batch_size
        self.diayn_batch_size = diayn_batch_size
        self.dynamics_batch_size = dynamics_batch_size
        self.update_counter = 0

        self.dynamics_warmup_step = dynamics_warmup_step

        self.policy_gradient_step = self.diayn_gradient_step = self.dynamics_gradient_step = 0

        super().__init__(
            learning_type="offpolicy",
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=max_epoch,
            step_per_epoch=env_step_per_epoch,
            step_per_collect=env_step_per_collect,
            episode_per_test=test_ep_per_epoch,
            batch_size=1,                           # not used by BaseTrainer
            update_per_step=1,                      # not used by BaseTrainer
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=resume_from_log,
            reward_metric=reward_metric,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress,
            test_in_train=test_in_train,
            **kwargs,
        )

    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        result = super().__next__()
        # draw the high-level action sampling and achievement distribution
        self.logger.log_policy_data(self.policy, self.epoch)
        self.logger.log_lower_buffer_data(self.train_collector.buffer, self.epoch)
        return result

    def policy_update_fn(
        self, data: Dict[str, Any], result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Perform one on-policy update."""
        assert self.train_collector is not None
        if self.env_step > self.dynamics_warmup_step:
            for _ in range(round(self.policy_update_per_env_step * result["n/st"])):
                losses = self.policy.update(self.upper_policy_batch_size,
                                            self.lower_policy_batch_size,
                                            self.train_collector.buffer,
                                            self.train_collector.upper_buffer)
                self.policy_gradient_step += 1
                self.log_update_data(data, losses, self.policy_gradient_step)

            if self.diayn_discriminator is not None:
                for _ in range(round(self.diayn_update_per_env_step * result["n/st"])):
                    losses = self.diayn_discriminator.update(self.diayn_batch_size, self.train_collector.buffer)
                    self.diayn_gradient_step += 1
                    self.log_update_data(data, losses, self.diayn_gradient_step)

        for _ in range(round(self.dynamics_update_per_collect * result["n/st"])):
            # TODO: change this to the desired parameters
            losses = self.dynamics.update(self.dynamics_batch_size, self.train_collector.buffer)
            self.dynamics_gradient_step += 1
            self.log_update_data(data, losses, self.dynamics_gradient_step)
        
        self.policy.update_schedules(self.update_counter)
        self.update_counter += 1

    def log_update_data(self, data: Dict[str, Any], losses: Dict[str, Any], step: int) -> None:
        """Log losses to current logger."""
        for k in losses.keys():
            self.stat[k].add(losses[k])
            losses[k] = self.stat[k].get()
            data[k] = f"{losses[k]:.3f}"
        self.logger.log_update_data(losses, step)


# performance (reached vs sampled) per graph, quantile graph performance, diversity is all you need accuracy, historical coverage, task performance