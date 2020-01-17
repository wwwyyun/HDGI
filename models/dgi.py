import torch
import torch.nn as nn
from layers import GCN, Assign, Discriminator
import numpy as np
class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, k, device, hard, batch_size):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.assign = Assign(k, n_h, device, hard)

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)
        
        self.k = k
    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)
        c, assignmat = self.assign(h_1.detach(), msk, training=True)
        h_2 = self.gcn(seq2, adj, sparse)

        #ret = self.disc(c,assignmat, h_1, h_2, samp_bias1, samp_bias2)
        ret = self.disc(c,assignmat.detach(), h_1, h_2, samp_bias1, samp_bias2)
        return ret, c, assignmat, h_1

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        # get node embedding
        h_1 = self.gcn(seq, adj, sparse)
        # get community embedding form node embedding
        c, _ = self.assign(h_1, msk, training=False)
        return h_1.detach(), c.detach()
    
