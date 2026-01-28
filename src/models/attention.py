import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalAttentionPooling(nn.Module):
    def __init__(self, query_dim, input_dim, hidden_dim=64):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, query, other_semesters, mask):
        Q = self.query_proj(query)
        K = self.key_proj(other_semesters)
        V = self.value_proj(other_semesters)
        scores = torch.einsum('bh, bsjh -> bsj', Q, K) / (K.size(-1) ** 0.5)
        scores.masked_fill_(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = torch.einsum('bsj, bsjh -> bsh', weights, V)
        return out