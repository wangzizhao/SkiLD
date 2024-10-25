from typing import Callable, Optional
import numpy as np

from tianshou.utils.logger.tensorboard import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter


class HRLLogger(TensorboardLogger):
    _transition_logging_keys = [f"episode/{k}" for k in ["success", "rew", "lower_rew", "upper_rew"]] + \
                               [f"lower/{k}" for k in ["reached_graph", "reached_goal"]]

    def __init__(
        self,
        writer: SummaryWriter,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        policy_interval: int = 1,
        buffer_interval: int = 1,
        save_interval: int = 1,
        write_flush: bool = True,
    ) -> None:
        super().__init__(writer, train_interval, test_interval, update_interval, save_interval, write_flush)
        self.policy_interval = policy_interval
        self.buffer_interval = buffer_interval
        self.last_log_policy_step = 0
        self.last_log_buffer_step = 0
        self.last_log_update_step = {}

    def log_transition_data(self, log_prefix: str, collect_result: dict, step: int) -> None:
        # collector use v == np.nan as a flag that no data is available to log for the key
        collect_result = {f"{log_prefix}/{k}": collect_result[k] for k in self._transition_logging_keys
                          if np.isfinite(collect_result[k])}

        if log_prefix == "train":
            last_log_step = self.last_log_train_step
        elif log_prefix == "test":
            last_log_step = self.last_log_test_step
        else:
            raise ValueError(f"Unknown log_prefix: {log_prefix}")
        if len(collect_result) and step - last_log_step >= self.train_interval:
            self.write("", step, collect_result)
            if log_prefix == "train":
                self.last_log_train_step = step
            elif log_prefix == "test":
                self.last_log_test_step = step
            else:
                raise ValueError(f"Unknown log_prefix: {log_prefix}")

    def log_train_data(self, collect_result: dict, step: int) -> None:
        self.log_transition_data("train", collect_result, step)

    def log_test_data(self, collect_result: dict, step: int) -> None:
        assert collect_result["n/ep"] > 0
        self.log_transition_data("test", collect_result, step)

    def log_update_data(self, update_result: dict, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param update_result: a dict containing information of data collected in
            updating stage, i.e., returns of policy.update().
        :param int step: stands for the timestep the collect_result being logged.
        """
        log_data = {}
        for k, v in update_result.items():
            if k not in self.last_log_update_step:
                self.last_log_update_step[k] = -1
            if step - self.last_log_update_step[k] >= self.update_interval:
                log_data[f"update/{k}"] = v
                self.last_log_update_step[k] = step
        self.write("", step, log_data)

    def log_policy_data(self, policy, step: int) -> None:
        if step - self.last_log_policy_step >= self.policy_interval:
            has_written = policy.logging(self.writer, step)
            if has_written:
                self.last_log_policy_step = step

    def log_lower_buffer_data(self, buffer, step: int) -> None:
        if step - self.last_log_buffer_step >= self.buffer_interval:
            has_written = buffer.logging(self.writer, step)
            if has_written:
                self.last_log_buffer_step = step

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
    ) -> None:
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            path = save_checkpoint_fn(epoch, env_step, gradient_step)
            if path is not None:
                self.write("save/epoch", epoch, {"save/epoch": epoch})
                self.write("save/env_step", env_step, {"save/env_step": env_step})
                self.write(
                    "save/gradient_step", gradient_step,
                    {"save/gradient_step": gradient_step}
                )
