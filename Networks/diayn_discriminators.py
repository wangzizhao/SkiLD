from typing import Any, Optional, Union, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import ReplayBuffer, to_numpy
from Option.utils import get_new_indices
import torch.nn.utils.spectral_norm as spectral_norm


class DiaynDiscriminator(nn.Module):
    def __init__(self, config, obs_dim, graph_encoding_size, device):
        super(DiaynDiscriminator, self).__init__()

        dcfg = config.policy.upper.diayn
        hidden_dim = dcfg.hidden_dim
        self.skill_dim = dcfg.num_classes
        self.use_state = dcfg.use_state

        # Directly concat the obs and the graph
        input_size = obs_dim + graph_encoding_size
        if self.use_state:
            input_size += obs_dim

        self.wide_diayn = config.policy.upper.diayn.wide_diayn
        self.spectral_norm = config.policy.upper.diayn.spectral_norm
        if self.wide_diayn:
            if self.spectral_norm:
                self.skill_pred_net = nn.ModuleList([nn.Sequential(spectral_norm(nn.Linear(input_size, hidden_dim)),
                                                                   nn.ReLU(),
                                                                   spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                                                                   nn.ReLU(),
                                                                   spectral_norm(nn.Linear(hidden_dim, dcfg.num_classes)))
                                                     for _ in range(config.num_factors)])
            else:
                self.skill_pred_net = nn.ModuleList([nn.Sequential(nn.Linear(input_size, hidden_dim),
                                                                   nn.ReLU(),
                                                                   nn.Linear(hidden_dim, hidden_dim),
                                                                   nn.ReLU(),
                                                                   nn.Linear(hidden_dim, dcfg.num_classes))
                                                     for _ in range(config.num_factors)])
        else:
            if self.spectral_norm:
                self.skill_pred_net = nn.Sequential(spectral_norm(nn.Linear(input_size, hidden_dim)),
                                                    nn.ReLU(),
                                                    spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                                                    nn.ReLU(),
                                                    spectral_norm(nn.Linear(hidden_dim, dcfg.num_classes)))
            else:
                self.skill_pred_net = nn.Sequential(nn.Linear(input_size, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, dcfg.num_classes))

        # loss criterion
        self.diayn_criterion = nn.CrossEntropyLoss()
        # optimizers
        self.diayn_opt = torch.optim.Adam(self.parameters(), lr=dcfg.lr)
        self.device = device

        # on-policy update
        self.on_policy_update = config.policy.upper.diayn.on_policy_update
        self.buffer_last_index = None
        self.on_policy_indices = None

        # graph info
        self.reached_graph = self.get_achieved_goal = None
        self.num_factors = num_factors = config.num_factors
        self.num_edge_classes = num_edge_classes = config.graph_encoding.num_edge_classes
        self.graph_type = config.policy.upper.graph_type

        if self.graph_type == "graph":
            self.graph_info_size = num_factors * (num_factors + 1) * num_edge_classes
        elif self.graph_type == "factor":
            self.graph_info_size = num_factors + (num_factors + 1) * num_edge_classes
        elif self.graph_type == "none":
            self.graph_info_size = 0
        else:
            raise NotImplementedError(f"unknown graph type: {self.graph_type}")

    def to_tensor(self, mat, idx_filter=None):
        if idx_filter is None:
            return torch.tensor(mat, device=self.device)
        else:
            mat = mat[idx_filter]
            return torch.tensor(mat, device=self.device)

    def get_achieved_z(self, graph, state, next_state):
        graph = self.to_tensor(graph)
        state = self.to_tensor(state)
        next_state = self.to_tensor(next_state)

        d_pred = self.forward(graph, state, next_state)
        z_onehot = F.one_hot(torch.argmax(d_pred, dim=-1), self.skill_dim)
        return to_numpy(z_onehot)

    def get_intrinsic_reward(self, graph, state, next_state, skill):
        graph = self.to_tensor(graph)
        state = self.to_tensor(state)
        next_state = self.to_tensor(next_state)
        skill = self.to_tensor(skill)

        z_hat = torch.argmax(skill, dim=1)

        d_pred = self.forward(graph, state, next_state)

        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]), z_hat] + np.log(self.skill_dim)
        reward = reward.flatten().detach().cpu().numpy()

        # from tianshou.data import to_numpy
        # softmax = to_numpy(F.softmax(d_pred, dim=1))
        # print("acc", softmax, z_hat.item(), softmax[torch.arange(d_pred.shape[0]), z_hat])
        return reward

    # Predict skill probability given
    def forward(self, graph, state, next_state):
        if self.use_state:
            input = torch.concat([graph, state, next_state], dim=-1)
        else:
            input = torch.concat([graph, next_state], dim=-1)
        if self.wide_diayn:
            # Loop through all networks
            # TODO: we can potentially prune out the factor indicator but not critical
            skill_pred = torch.zeros(input.shape[0], self.skill_dim, device=self.device)
            for i in range(self.num_factors):
                idices = torch.where(graph[:, i] == 1)
                filtered_batch = input[idices]
                if len(filtered_batch) > 0:
                    skill_pred[idices] = self.skill_pred_net[i](filtered_batch)
        else:
            skill_pred = self.skill_pred_net(input)
        return skill_pred

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> Dict[str, Any]:
        if buffer is None:
            return {}

        if self.on_policy_update:
            self.buffer_last_index, indices = get_new_indices(self.buffer_last_index, buffer)
            if len(indices) != 0:
                self.on_policy_indices = indices

            if self.on_policy_indices is None or len(self.on_policy_indices) == 0:
                return {}
            indices = np.random.choice(self.on_policy_indices, sample_size)
            buffer.restore_cache()
            batch = buffer[indices]
        else:
            batch, indices = buffer.sample(sample_size)

        assert self.reached_graph is not None, "diayn_discriminator.reached_graph should be set when rtt is initialized"
        assert self.get_achieved_goal is not None, "diayn_discriminator.get_achieved_goal should be set when rtt is initialized"
        reached_graph, desired_graph = self.reached_graph(batch, return_true_for_state_coverage_policy=False)
        state_value = self.get_achieved_goal(batch, desired_graph, use_next_obs=False)
        next_state_value = self.get_achieved_goal(batch, desired_graph, use_next_obs=True)
        graph_info = batch.obs.desired_goal[..., :self.graph_info_size]
        diayn_z = batch.obs.desired_goal[..., self.graph_info_size:]

        diayn_results = self.learn(graph_info, state_value, next_state_value, diayn_z, reached_graph)
        diayn_results = {f"diayn/{k}": v for k, v in diayn_results.items()}
        return diayn_results

    def learn(self, graph, state, next_state, skill, reached_graph=None):
        """
        Train the Diayn Discriminator
        return: dictionary indicating training results
        """
        # TODO: I think we should always train diayn with all experiences
        if reached_graph is not None and not np.any(reached_graph):
            return {}

        graph = self.to_tensor(graph, idx_filter=reached_graph)
        state = self.to_tensor(state, idx_filter=reached_graph)
        next_state = self.to_tensor(next_state, idx_filter=reached_graph)
        skill = self.to_tensor(skill, idx_filter=reached_graph)

        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.forward(graph, state, next_state)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        loss = self.diayn_criterion(d_pred, z_hat)

        df_accuracy = torch.sum(
            torch.eq(z_hat, pred_z.reshape(1, list(pred_z.size())[0])[0])).float() / list(pred_z.size())[0]

        self.diayn_opt.zero_grad()
        loss.backward()
        self.diayn_opt.step()

        loss_details = {"diayn_loss": loss.item(),
                        "diayn_acc": df_accuracy}

        return loss_details
