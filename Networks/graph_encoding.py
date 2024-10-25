import torch
import torch.nn.functional as F

from typing import Sequence
from torch import nn
from omegaconf import DictConfig


class Encoder(nn.Module):
    def __init__(self,
                 num_factors: int,
                 graph_type: str,
                 num_edge_classes: int,
                 hidden_sizes: Sequence[int],
                 latent_size: int,
                 device: torch.device,
                 v_min: float = -1,
                 v_max: float = 1,):
        super(Encoder, self).__init__()

        self.num_factors = num_factors
        self.graph_type = graph_type
        self.num_edge_classes = num_edge_classes
        self.device = device
        self.mean = (v_min + v_max) / 2
        self.scale = (v_max - v_min) / 2

        if graph_type == "graph":
            self.input_size = num_factors * (num_factors + 1) * num_edge_classes
        elif graph_type == "factor":
            self.input_size = num_factors + (num_factors + 1) * num_edge_classes
        else:
            raise NotImplementedError(f"unknown graph encoding type: {graph_type}")

        sizes = [self.input_size] + list(hidden_sizes) + [latent_size]
        net_list = sum([[nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)]
                        for in_dim, out_dim in zip(sizes[:-1], sizes[1:])],
                       start=list())
        self.net = nn.Sequential(*net_list[:-1])    # [:-1] to remove last ReLU

    def forward(self, x):
        """
        x:
            if graph_type == "graph":
                (bs, num_factors, num_factors + 1, num_edge_classes)
            elif graph_type == "factor":
                (factor, parents)
                factor: (bs, num_factors)
                parents: (bs, num_factors + 1, num_edge_classes)
        z: (bs, num_edge_classes)
        """

        if self.graph_type == "graph":
            x = torch.as_tensor(x, device=self.device, dtype=torch.get_default_dtype())
            if x.shape[-1] != self.num_edge_classes:
                x = F.one_hot(x, self.num_edge_classes)

            # (bs, num_factors * (num_factors + 1) * num_edge_classes)
            x = torch.flatten(x, start_dim=-3).float()
        elif self.graph_type == "factor":
            factor, parents = x
            factor = torch.as_tensor(factor, device=self.device, dtype=torch.get_default_dtype())
            parents = torch.as_tensor(parents, device=self.device, dtype=torch.get_default_dtype())

            if factor.shape[-1] != self.num_factors:
                factor = F.one_hot(factor, self.num_factors)
            if parents.shape[-1] != self.num_edge_classes:
                parents = F.one_hot(parents, self.num_edge_classes)

            # (bs, (num_factors + 1) * num_edge_classes)
            parents = torch.flatten(parents, start_dim=-2)

            # (bs, num_factors + (num_factors + 1) * num_edge_classes)
            x = torch.cat([factor.float(), parents.float()], dim=-1)
        else:
            raise NotImplementedError(f"unknown graph encoding type: {self.graph_type}")

        z = self.net(x)
        z = self.mean + self.scale * F.tanh(z)
        return z


class Decoder(nn.Module):
    def __init__(self,
                 num_factors: int,
                 graph_type: str,
                 num_edge_classes: int,
                 hidden_sizes: Sequence[int],
                 latent_size: int,
                 device: torch.device,):
        super(Decoder, self).__init__()
        self.num_factors = num_factors
        self.graph_type = graph_type
        self.num_edge_classes = num_edge_classes
        self.device = device

        if graph_type == "graph":
            self.output_size = num_factors * (num_factors + 1) * num_edge_classes
        elif graph_type == "factor":
            self.output_size = num_factors + (num_factors + 1) * num_edge_classes
        else:
            raise NotImplementedError(f"unknown graph encoding type: {graph_type}")

        sizes = [latent_size] + list(hidden_sizes) + [self.output_size]
        net_list = sum([[nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)]
                        for in_dim, out_dim in zip(sizes[:-1], sizes[1:])],
                       start=list())
        self.net = nn.Sequential(*net_list[:-1])    # [:-1] to remove last ReLU

    def forward(self, z, output_class=False):
        """
        z: (bs, num_edge_classes)
        x:
            if graph_type == "graph":
                (bs, num_factors, num_factors + 1, num_edge_classes)
            elif graph_type == "factor":
                (factor, parents)
                factor: (bs, num_factors)
                parents: (bs, num_factors + 1, num_edge_classes)
        """
        z = torch.as_tensor(z, device=self.device, dtype=torch.get_default_dtype())
        x_logits = self.net(z)

        num_factors = self.num_factors
        num_edge_classes = self.num_edge_classes
        if self.graph_type == "graph":
            x_logits = x_logits.reshape(*z.shape[:-1], num_factors, num_factors + 1, num_edge_classes)

            if output_class:
                return x_logits.argmax(dim=-1)
            else:
                return x_logits
        elif self.graph_type == "factor":
            factor_logits, parents_logits = x_logits.split([num_factors, (num_factors + 1) * num_edge_classes], dim=-1)
            parents_logits = parents_logits.view(*parents_logits.shape[:-1], num_factors + 1, num_edge_classes)

            if output_class:
                return factor_logits.argmax(dim=-1), parents_logits.argmax(dim=-1)
            else:
                return factor_logits, parents_logits
        else:
            raise NotImplementedError(f"unknown graph encoding type: {self.graph_type}")


class GraphEncoding(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        device = config.device
        self.num_factors = num_factors = config.num_factors
        graph_encoding_config = config.graph_encoding
        self.graph_type = graph_type = config.policy.upper.graph_type
        self.latent_size = latent_size = graph_encoding_config.latent_size
        self.num_edge_classes = num_edge_classes = graph_encoding_config.num_edge_classes
        hidden_sizes = graph_encoding_config.hidden_sizes

        self.encoder = Encoder(num_factors, graph_type, num_edge_classes, hidden_sizes, latent_size, device)
        self.decoder = Decoder(num_factors, graph_type, num_edge_classes, hidden_sizes, latent_size, device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=graph_encoding_config.lr)

        self.cur_latent_l2_reg_coef = self.latent_l2_reg_coef = graph_encoding_config.latent_l2_reg_coef

    def anneal(self, step: int, total_step: int):
        self.cur_latent_l2_reg_coef = self.latent_l2_reg_coef * step / total_step

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def preprocess(self, x):
        if self.graph_type == "graph":
            x = F.one_hot(x, self.num_edge_classes)
        elif self.graph_type == "factor":
            factor, parents = x
            factor = F.one_hot(factor, self.num_factors)
            parents = F.one_hot(parents, self.num_edge_classes)
            x = (factor, parents)
        else:
            raise NotImplementedError(f"unknown graph encoding type: {self.graph_type}")
        return x

    def reconstruction_loss(self, x, x_logits):
        """
        x, x_logits:
            if graph_type == "graph":
                (bs, num_factors, num_factors + 1, num_edge_classes)
            elif graph_type == "factor":
                (factor, parents)
                factor: (bs, num_factors)
                parents: (bs, num_factors + 1, num_edge_classes)
        """
        if self.graph_type == "graph":
            x_log_prob = F.log_softmax(x_logits, dim=-1)
            recon_loss = -(x * x_log_prob).sum(dim=-1).mean(dim=(-2, -1)).mean()
            accuracy = (x.argmax(dim=-1) == x_log_prob.argmax(dim=-1)).float().mean()
        elif self.graph_type == "factor":
            factor, parents = x
            factor_logits, parents_logits = x_logits
            factor_log_prob = F.log_softmax(factor_logits, dim=-1)
            parents_log_prob = F.log_softmax(parents_logits, dim=-1)
            factor_recon_loss = -(factor * factor_log_prob).sum(dim=-1).mean()
            parents_recon_loss = -(parents * parents_log_prob).sum(dim=-1).mean()
            recon_loss = factor_recon_loss + parents_recon_loss

            factor_accuracy = (factor.argmax(dim=-1) == factor_logits.argmax(dim=-1)).float()
            parents_accuracy = (parents.argmax(dim=-1) == parents_logits.argmax(dim=-1)).float()
            accuracy = torch.cat([factor_accuracy.unsqueeze(dim=-1), parents_accuracy], dim=-1).mean()
        else:
            raise NotImplementedError(f"unknown graph encoding type: {self.graph_type}")

        return recon_loss, accuracy

    def update(self, x):
        """
        x:
            if graph_type == "graph":
                (bs, num_factors, num_factors + 1)
            elif graph_type == "factor":
                (factor, parents)
                factor: (bs, )
                parents: (bs, num_factors + 1)
        """
        x = self.preprocess(x)
        z, x_logits = self.forward(x)
        recon_loss, accuracy = self.reconstruction_loss(x, x_logits)

        z_l2 = z.pow(2).sum(dim=-1).mean()

        loss_details = {"latent_norm": z_l2,
                        "reconstruction": recon_loss,
                        "accuracy": accuracy}
        loss = recon_loss + z_l2 * self.cur_latent_l2_reg_coef

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_details

    def test(self, x, x_pos):
        """
        x, x_pos:
            if graph_type == "graph":
                (bs, num_factors, num_factors + 1)
            elif graph_type == "factor":
                (factor, parents)
                factor: (bs, )
                parents: (bs, num_factors + 1)
        """
        x = self.preprocess(x)
        x_pos = self.preprocess(x_pos)

        z = self.encoder(x)                                             # (bs, latent_size)
        z_pos = self.encoder(x_pos)                                     # (bs, latent_size)

        bs = z.shape[0]
        z_z_pos_diff = (z - z_pos).pow(2).sum(dim=-1)                   # (bs,)
        z_repeat = z.unsqueeze(dim=0).repeat(bs, 1, 1)                  # (bs, bs, latent_size)
        z_z_diff = (z_repeat - z[:, None, :]).pow(2).sum(dim=-1)        # (bs, bs)

        rank = (z_z_diff < z_z_pos_diff[:, None]).float().sum(dim=-1).mean()
        return rank
