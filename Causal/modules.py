import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from Causal.dynamics_utils import forward_network


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[channel_size, seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(-3)]
        return x


class Attention(nn.Module):
    def __init__(self, attention_dim, num_queries, query_dim, num_keys, key_dim, out_dim=None, use_bias=False):
        super(Attention, self).__init__()
        self.temperature = np.sqrt(attention_dim)
        self.use_bias = use_bias

        if out_dim is None:
            out_dim = attention_dim

        b = 1 / self.temperature
        b_v = 1 / np.sqrt(out_dim)
        self.query_weight = nn.Parameter(torch.FloatTensor(num_queries, query_dim, attention_dim).uniform_(-b, b))
        self.query_bias = nn.Parameter(torch.zeros(num_queries, 1, attention_dim))
        self.key_weight = nn.Parameter(torch.FloatTensor(num_keys, key_dim, attention_dim).uniform_(-b, b))
        self.key_bias = nn.Parameter(torch.zeros(num_keys, 1, attention_dim))
        self.value_weight = nn.Parameter(torch.FloatTensor(num_keys, key_dim, out_dim).uniform_(-b_v, b_v))
        self.value_bias = nn.Parameter(torch.zeros(num_keys, 1, out_dim))

    def forward_score(self, q, k):
        """
        :param q: (num_queries, bs, query_dim)
        :param k: (num_keys, bs, key_dim)
        :return: logits (bs, num_queries, num_keys)
        """
        query = torch.bmm(q, self.query_weight)  # (num_queries, bs, attention_dim)
        key = torch.bmm(k, self.key_weight)  # (num_keys, bs, attention_dim)
        if self.use_bias:
            query += self.query_bias  # (num_queries, bs, attention_dim)
            key += self.key_bias  # (num_keys, bs, attention_dim)

        query = query.permute(1, 0, 2)  # (bs, num_queries, attention_dim)
        key = key.permute(1, 2, 0)  # (bs, attention_dim, num_keys)

        logits = torch.bmm(query, key) / self.temperature  # (bs, num_queries, num_keys)
        return logits

    def forward(self, q, k, return_logits=False, gumbel_select=False, tau=1.0):
        """
        :param q: (num_queries, bs, query_dim)
        :param k: (num_keys, bs, key_dim)
        :return:
        """
        value = torch.bmm(k, self.value_weight)  # (num_keys, bs, attention_value_dim)
        if self.use_bias:
            value += self.value_bias  # (num_keys, bs, attention_value_dim)

        logits = self.forward_score(q, k)  # (bs, num_queries, num_keys)

        if gumbel_select:
            attn = F.gumbel_softmax(logits, dim=-1, hard=True, tau=tau)  # (bs, num_queries, num_keys)
        else:
            attn = F.softmax(logits, dim=-1)  # (bs, num_queries, num_keys)

        attn = attn.permute(1, 2, 0).unsqueeze(dim=-1)  # (num_queries, num_keys, bs, 1)

        output = (value * attn).sum(dim=1)  # (num_queries, bs, attention_value_dim)

        if return_logits:
            return output, logits
        else:
            return output


class ChannelMHAttention(nn.Module):
    def __init__(self, chan_shape, attention_dim, num_heads, num_queries, query_dim, num_keys, key_dim,
                 out_dim=None, use_bias=False, residual=False, share_weight_across_kqv=False, post_fc_dims=[]):
        super(ChannelMHAttention, self).__init__()

        if out_dim is None:
            out_dim = attention_dim
        self.temperature = np.sqrt(attention_dim)
        self.use_bias = use_bias
        self.residual = residual
        if residual:
            assert query_dim == out_dim
            if post_fc_dims:
                assert query_dim == post_fc_dims[-1]

        self.chan_shape = chan_shape
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_keys = num_keys
        self.embed_dim = embed_dim = num_heads * attention_dim

        b = 1 / self.temperature
        b_p = 1 / np.sqrt(out_dim)
        self.post_fc_weights = nn.ParameterList()
        self.post_fc_biases = nn.ParameterList()

        self.share_weight_across_kqv = share_weight_across_kqv
        if share_weight_across_kqv:
            # overwrite for weight and bias initialization
            num_queries, num_keys = 1, 1

        self.query_weight = nn.Parameter(torch.FloatTensor(*chan_shape, num_queries, query_dim, embed_dim).uniform_(-b, b))
        self.key_weight = nn.Parameter(torch.FloatTensor(*chan_shape, num_keys, key_dim, embed_dim).uniform_(-b, b))
        self.value_weight = nn.Parameter(torch.FloatTensor(*chan_shape, num_keys, key_dim, embed_dim).uniform_(-b, b))
        self.proj_weight = nn.Parameter(torch.FloatTensor(*chan_shape, num_queries, embed_dim, out_dim).uniform_(-b_p, b_p))
        if use_bias:
            self.query_bias = nn.Parameter(torch.zeros(*chan_shape, num_queries, 1, embed_dim))
            self.key_bias = nn.Parameter(torch.zeros(*chan_shape, num_queries, 1, embed_dim))
            self.value_bias = nn.Parameter(torch.zeros(*chan_shape, num_queries, 1, embed_dim))
            self.proj_bias = nn.Parameter(torch.zeros(*chan_shape, num_queries, 1, out_dim))
        else:
            self.query_bias = self.key_bias = self.value_bias = self.proj_bias = None

        in_dim = out_dim
        for fc_dim in post_fc_dims:
            b = 1 / np.sqrt(fc_dim)
            self.post_fc_weights.append(
                nn.Parameter(torch.FloatTensor(*chan_shape, num_queries, in_dim, fc_dim).uniform_(-b, b)))
            self.post_fc_biases.append(nn.Parameter(torch.zeros(*chan_shape, num_queries, 1, fc_dim)))
            in_dim = fc_dim

        self.q_pos_enc = PositionalEncoding(d_model=query_dim)
        self.k_pos_enc = PositionalEncoding(d_model=key_dim)

    def forward(self, q, k, return_attn=False, gumbel_select=False, tau=1.0, log_attn_mask=None):
        """
        :param q: (chan_shape, num_queries, bs, query_dim)
        :param k: (chan_shape, num_keys, bs, key_dim)
        :param log_attn_mask: (chan_shape, bs, num_queries, num_keys)
        :return:
        """
        bs = q.shape[-2]
        attention_dim = self.attention_dim
        num_heads = self.num_heads
        embed_dim = self.embed_dim
        num_queries = self.num_queries
        num_keys = self.num_keys

        q_pe = self.q_pos_enc(q)
        k_pe = self.k_pos_enc(k)

        query = forward_network(q_pe, self.query_weight, self.query_bias)   # (chan_shape, num_queries, bs, embed_dim)
        key = forward_network(k_pe, self.key_weight, self.key_bias)         # (chan_shape, num_keys, bs, embed_dim)

        chan_shape = query.shape[:-3]
        chan_shape_prod = np.prod(chan_shape)

        query = query.reshape(chan_shape_prod, num_queries, bs, -1)         # (chan_shape_prod, num_queries, bs, embed_dim)
        key = key.reshape(chan_shape_prod, num_keys, bs, -1)                # (chan_shape_prod, num_keys, bs, embed_dim)

        query = query.transpose(1, 0)                                       # (num_queries, chan_shape_prod, bs, embed_dim)
        key = key.transpose(1, 0)                                           # (num_keys, chan_shape_prod, bs, embed_dim)

        query = query.reshape(num_queries, -1, attention_dim)   # (num_queries, chan_shape_prod * bs * num_heads, attention_dim)
        key = key.reshape(num_keys, -1, attention_dim)          # (num_keys, chan_shape_prod * bs * num_heads, attention_dim)

        query = query.permute(1, 0, 2)                          # (chan_shape_prod * bs * num_heads, num_queries, attention_dim)
        key = key.permute(1, 2, 0)                              # (chan_shape_prod * bs * num_heads, attention_dim, num_keys)

        logits = torch.bmm(query, key) / self.temperature       # (chan_shape_prod * bs * num_heads, num_queries, num_keys)

        if log_attn_mask is not None:
            # (chan_shape, bs, num_queries, num_keys) -> (chan_shape_prod * bs, num_queries, num_keys)
            log_attn_mask = log_attn_mask.reshape(-1, num_queries, num_keys)
            # (chan_shape_prod * bs, num_heads, num_queries, num_keys)
            log_attn_mask = log_attn_mask.unsqueeze(dim=1).expand(-1, num_heads, -1, -1)
            log_attn_mask = log_attn_mask.reshape(-1, num_queries, num_keys)
            logits = logits + log_attn_mask

        if gumbel_select:
            attn = F.gumbel_softmax(logits, dim=-1, hard=True, tau=tau)
        else:
            attn = F.softmax(logits, dim=-1)                    # (chan_shape_prod * bs * num_heads, num_queries, num_keys)

        attn = attn.reshape(chan_shape_prod, bs * num_heads, num_queries, num_keys)
        attn = attn.permute(0, 2, 3, 1)                         # (chan_shape_prod, num_queries, num_keys, bs * num_heads)
        attn = attn.reshape(chan_shape_prod, num_queries, num_keys, bs * num_heads, 1)

        value = forward_network(k_pe, self.value_weight, self.value_bias)       # (chan_shape, num_keys, bs, embed_dim)
        value = value.reshape(chan_shape_prod, 1, num_keys, bs * num_heads, attention_dim)

        output = (value * attn).sum(dim=2)                      # (chan_shape_prod, num_queries, bs * num_heads, attention_dim)
        output = output.reshape(*chan_shape, num_queries, bs, embed_dim)
        output = forward_network(output, self.proj_weight, self.proj_bias)

        if self.residual:
            output += q

        # post attention fully-connected layers
        fc_output = forward_network(output, self.post_fc_weights, self.post_fc_biases)

        if self.residual:
            fc_output += output

        if not return_attn:
            return fc_output

        attn = attn.reshape(chan_shape_prod, num_queries, num_keys, bs, num_heads)
        attn = attn.permute(3, 0, 4, 1, 2)  # (bs, chan_shape_prod, num_heads, num_queries, num_keys)
        attn = attn.reshape(bs, *chan_shape, num_heads, num_queries, num_keys)
        return fc_output, attn
