import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

class DKT(nn.Module):
    def __init__(self, num_skills, input_dim, hidden_dim = 128, layers = 4, dropout = 0.1):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.PReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.LSTM(hidden_dim, hidden_dim, layers, batch_first = True, dropout = dropout)
        self.post = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                                  nn.Linear(hidden_dim, num_skills), nn.Sigmoid())

    def forward(self, X, skill_idx):
        corrects = self.post(self.net(self.pre(X))[0])
        skill_idx = torch.where(skill_idx == -1000, 0, skill_idx)
        if skill_idx.max() >= corrects.shape[-1]:
            import pdb; pdb.set_trace()
        return torch.gather(corrects, dim = -1, index = skill_idx.long().unsqueeze(-1))

