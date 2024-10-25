from typing import Any, Dict, Optional, Union

import numpy as np
import time
import tianshou as ts

from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Batch


class HierarchicalModel(ts.policy.BasePolicy):
    """
    Manages the hierarchical model, inheriting from tianshou.BasePolicy \
    Even though it inherits, it does NOT strictly act like a BasePolicy, \
    but as a composition of BasePolicies (for upper and lower)
    """

    def __init__(self, config, lower, upper, action_space):
        super().__init__(action_scaling=True, action_space=action_space, action_bound_method="clip")
        self.environment_goals = config.env.goal_based  # indicates if an environment is goal based already
        self.upper = upper
        self.lower = lower  # lowerPolicy handles intermediate policies
        self.update_upper = True
        self.update_lower = True
        self.lower_reset_env_step = 0
        self.last_upper_reset_epoch = 0
        self.last_lower_reset_epoch = 0
        self.fixed_graph = config.policy.upper.fixed_graph
        self.update_schedule_frequency = config.option.update_schedule_frequency
        self.domain = config.env.env_name

    def split_observation(self, batch, use_next_obs=False):
        if use_next_obs:
            observation = batch.obs_next["observation"] if self.environment_goals else batch.obs_next
        else:
            observation = batch.obs["observation"] if self.environment_goals else batch.obs
        achieved_goal = self.lower.get_achieved_goal(batch, use_next_obs=use_next_obs)
        return observation, achieved_goal

    def train_upper(self, train: bool = True, upper_buffer=None) -> None:
        if not self.update_upper and train:
            self.upper.reset_training(upper_buffer)
        self.update_upper = train

    def train_lower(self, train: bool = True, lower_buffer=None) -> None:
        if not self.update_lower and train:
            self.lower.reset_training(lower_buffer)
        self.update_lower = train

    def update(self, upper_batch_size: int, lower_batch_size: int,
               buffer: Batch, upper_buffer: Batch, **kwargs: Any) -> Dict[str, Any]:
        """
        Updates the upper policy with upper actions
        Updates the lower policy with lower actions
        """
        upper_losses = lower_losses = {}
        if self.update_upper:
            upper_losses = self.upper.update(upper_batch_size, upper_buffer, buffer, **kwargs)
        if self.update_lower:
            lower_losses = self.lower.update(lower_batch_size, buffer, **kwargs)

        upper_losses = {"upper/" + k: v for k, v in upper_losses.items()}
        lower_losses = {"lower/" + k: v for k, v in lower_losses.items()}
        return {**upper_losses, **lower_losses}

    def update_lower_stats(self, lower_metrics, desired_goal):
        self.upper.update_lower_stats(lower_metrics, desired_goal)
        self.lower.update_lower_stats(lower_metrics, desired_goal)

    def _new_actions(self, batch, upper):
        """
        @param batch is the original batch of data
        @param upper is a policy batch of newly generated upper data
        """
        needs_upper = self.upper.needs_upper(batch)
        upper_actions = list()
        for i, nu in enumerate(needs_upper):
            if nu or "upper" not in batch.sampled_action_chain:
                upper_actions.append(upper[i])
            else:
                new_sample = Batch(act=batch.action_chain.upper[i],
                                   sampled_act=batch.sampled_action_chain.upper[i],
                                   option_choice=batch.option_choice[i])
                if "all_desired_goals" in batch.obs:
                    new_sample.all_desired_goals = batch.obs.all_desired_goals[i]
                upper_actions.append(new_sample)
        return Batch.stack(upper_actions), needs_upper

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        """Update policy with a given batch of data.

        :return: A dict, including the data needed to be logged (e.g., loss).

        .. note::

            In order to distinguish the collecting state, updating state and
            testing state, you can check the policy state by ``self.training``
            and ``self.updating``. Please refer to :ref:`policy_state` for more
            detailed explanation.

        .. warning::

            If you use ``torch.distributions.Normal`` and
            ``torch.distributions.Categorical`` to calculate the log_prob,
            please be careful about the shape: Categorical distribution gives
            "[batch_size]" shape while Normal distribution gives "[batch_size,
            1]" shape. The auto-broadcasting of numerical operation with torch
            tensors will amplify this error.
        """
        return dict()

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which MUST have the following keys:

            * ``act`` an numpy.ndarray or a torch.Tensor, the action over \
                given batch data (primitive level).
            * ``state`` a dict, with upper and lower components, the \
                internal state of the policy, ``None`` as default (not used).
        The keyword ``policy`` is reserved and the corresponding data will be
        stored into the replay buffer. This must contain:
        * ``action_chain`` The sequence of actions taken for the whole \
            hierarchy (typically a 2-tuple of upper and lower)
        * ``option_choice`` The choice of lower level options by the upper
        Other instances,
        ::

            # some code
            return Batch(..., policy=Batch(log_prob=dist.log_prob(act)))
            # and in the sampled data batch, you can directly use
            # batch.policy.log_prob to get your data.

        .. note::

            In continuous action space, you should do another step "map_action" to get
            the real action:
            ::

                act = policy(batch).act  # doesn't map to the target action range
                act = policy.map_action(act, batch)
        """
        # upper action
        upper_action = self.upper(batch, state)
        upper_action, resampled = self._new_actions(batch, upper_action)  # applies temporal extension

        # prepare low obs, achieved goal will be updated
        batch.obs.desired_goal = upper_action.sampled_act
        batch.option_choice = upper_action.option_choice

        if self.fixed_graph:
            num_factors, num_edge_classes = self.upper.num_factors, self.upper.num_edge_classes
            batch.obs.desired_goal[..., :num_factors + (num_factors + 1) * num_edge_classes] = 0

            if self.domain == "installing_printer":
                # printer
                factor = 0
                agent, printer, table, action = 1, 0, 0, 1
                parents = [agent, printer, table, action]
            elif self.domain == "thawing":
                # thawing
                factor = 0
                agent, frig, sink, olive, fish, date, action = 1, 0, 0, 0, 0, 0, 1
                parents = [agent, frig, sink, olive, fish, date, action]
            elif self.domain == "cleaning_car":
                # cleaning a car
                try:
                    self.count += 1
                    self.count = self.count % 500
                except:
                    self.count = 0

                if self.count <= 30:
                    factor = 3
                    agent, car, bucket, soap, sink, rag, action = 1, 0, 1, 1, 0, 0, 0
                    factor = 5
                    agent, car, bucket, soap, sink, rag, action = 0, 0, 0, 0, 1, 1, 0
                else:
                    factor = 2
                    agent, car, bucket, soap, sink, rag, action = 0, 0, 1, 1, 0, 0, 0
                    factor = 5
                    agent, car, bucket, soap, sink, rag, action = 0, 1, 0, 0, 0, 1, 0
                parents = [agent, car, bucket, soap, sink, rag, action]
            elif self.domain == "igibson":
                try:
                    self.count += 1
                    self.count = self.count % 1000
                except:
                    self.count = 0
                # factors = [robot, peach, knife, sink]
                print(f"executing count {self.count}")

                if self.count <= 100:
                    factor = 2
                    robot, peach, knife, sink, action = 1, 0, 1, 0, 1
                else:
                    factor = 2
                    robot, peach, knife, sink, action = 0, 0, 1, 1, 0


                # TODO: these are for cutting in fridge
                # if self.count <= 200:
                #     factor = 1
                #     robot, peach, knife, sink, action = 1, 1, 0, 0, 1
                # elif self.count <= 400:
                #     factor = 1
                #     robot, peach, knife, sink, action = 0, 1, 0, 1, 0
                # elif self.count <= 600:
                #     factor = 2
                #     robot, peach, knife, sink, action = 1, 0, 1, 0, 1
                # else:
                #     factor = 1
                #     robot, peach, knife, sink, action = 1, 1, 1, 0, 1

                parents = [robot, peach, knife, sink, action]
            else:
                raise NotImplementedError

            # factor(n_fac), (n_fac+1) * n_edge_classes, diayn(4) # the +1 is for action
            batch.obs.desired_goal[..., factor] = 1
            for i, parent_i in enumerate(parents):
                batch.obs.desired_goal[..., num_factors + i * num_edge_classes + parent_i] = 1
            batch.option_choice[:] = factor

        # if using the wide lower, this is where we assign all_desired_goals
        if "all_desired_goals" in upper_action:
            batch.obs.all_desired_goals = upper_action.all_desired_goals

        # lower action
        lower_action = self.lower(batch, state)

        action_chain = Batch(lower=lower_action.action_chain, upper=upper_action.act)
        sampled_action_chain = Batch(lower=lower_action.sampled_action_chain, upper=upper_action.sampled_act)

        act = lower_action.act  # TODO: not use action_chain for now, revisit when action_chain is used
        state = None  # TODO: support recurrent policies

        return Batch(act=act, state=state, policy=Batch(),
                     option_choice=upper_action.option_choice, resampled=resampled,
                     action_chain=action_chain, sampled_action_chain=sampled_action_chain)

    def check_rew_term_trunc(self, data):
        # TODO: if upper terminates on a different signal than done
        # determines which skills in upper or lower (if any) have terminated
        # based on the termination properties of both

        upper_reward, upper_term, upper_trunc = self.upper.rewtermdone.check_rew_term_trunc(data)
        lower_reward, lower_term, lower_trunc = self.lower.rewtermdone.check_rew_term_trunc(data)
        return (Batch(upper=upper_reward, lower=lower_reward),
                Batch(upper=upper_term, lower=lower_term),
                Batch(upper=upper_trunc, lower=lower_trunc))

    def random_sample(self, random_upper_only=False, batch=None, lower_state=None):
        # returns a random sample with fixed upper or a randomly selected upper
        # if random_upper is true, then uses both random upper and lower, respecting hierarchy
        # otherwise, only the lower is random, and requires a batch input
        batch_size = len(batch) if batch is not None else 1

        upper_action = self.upper.sample_act(batch)
        upper_action = self.upper.sample_action(upper_action, random=True)
        if "option_choice" not in upper_action:
            upper_action.option_choice = np.random.randint(len(self.lower.policies), size=batch_size)

        upper_action, resampled = self._new_actions(batch, upper_action)
        batch.obs.desired_goal = upper_action.sampled_act
        batch.option_choice = upper_action.option_choice
        if "all_desired_goals" in upper_action:
            batch.obs.all_desired_goals = upper_action.all_desired_goals

        if random_upper_only:
            lower_action = self.lower(batch, lower_state)
            lower_act = lower_action.action_chain[:, 0]  # selects the lowest policy from the chain
        else:  # if random upper, use the model to get the lower action
            lower_acts, lower_act = self.lower.sample_action(upper_action.option_choice)
            lower_action = Batch(action_chain=lower_acts,
                                 sampled_action_chain=lower_acts)

        action_chain = Batch(lower=lower_action.action_chain, upper=upper_action.act)
        sampled_action_chain = Batch(lower=lower_action.sampled_action_chain, upper=upper_action.sampled_act)

        state = None  # TODO: support recurrent policies
        return_batch = Batch(act=lower_act, state=state, policy=Batch(),
                             option_choice=upper_action.option_choice, resampled=resampled,
                             action_chain=action_chain, sampled_action_chain=sampled_action_chain)
        return return_batch

    def logging(self, writer: SummaryWriter, step: int) -> bool:
        upper_has_written = self.upper.logging(writer, step)
        lower_has_written = self.lower.logging(writer, step)
        return upper_has_written or lower_has_written

    def update_schedules(self, i):
        if self.update_schedule_frequency > 0 and i % self.update_schedule_frequency == 0:
            self.upper.update_schedules()
            self.lower.update_schedules()
    
    def update_state_counts(self, data):
        self.upper.update_state_counts(data)
        self.lower.update_state_counts(data)

