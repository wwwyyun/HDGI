import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class Assign(nn.Module):
    def __init__(self, k, in_dim, device, hard):
        super(Assign, self).__init__()

        self.k = k
        self.in_dim = in_dim
        self.community_embed = nn.Parameter(torch.zeros(self.k, self.in_dim))
        self.reset_parameters()
        #self.set2set = Set2Set()
        #self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.hard = hard
        self.device = device

    def reset_parameters(self):
        #gain = nn.init.calculate_gain()
        torch.nn.init.xavier_uniform_(self.community_embed)

    def communityCal(self, node_embed):
        # input node_embed.shape = [1, 2708, 512], after squeeze node_embed.shape = [2708, 512]
        # node_embed = self.bn1(torch.squeeze(node_embed))
        node_embed = torch.squeeze(node_embed)
        # community_embed.shape = [7, 512]
        # unsqueeze(community_embed) to [1, 7, 512]
        # unsqueeze(node_embed) to [2708, 1, 512]
        # scores.shape = [2708, 7]
        scores = torch.norm(torch.unsqueeze(node_embed, 1)-torch.unsqueeze(self.community_embed, 0), dim=2)
        #scores = self.bn2(scores)
        if self.training:
            assignmat = F.gumbel_softmax(logits=scores, tau=1., hard=self.hard) 
        else:
            ind = scores.argmax(dim=-1).reshape(scores.shape[0], 1)
            assignmat = torch.zeros(scores.shape).to(self.device).scatter_(1, ind, 1.)

        return self.community_embed, assignmat 

    def forward(self, seq, msk, training):
        self.training = training
        if msk is None:
            return self.communityCal(seq)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(centroids * msk, 1) / torch.sum(msk)

