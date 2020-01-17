import torch
import torch.nn as nn
import torch.nn.functional as F
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, assignmat, h_pl, h_mi, s_bias1=None, s_bias2=None):
        with torch.no_grad():
            community_embed_ = torch.div(torch.mm(torch.t(assignmat), torch.squeeze(h_pl)),\
                torch.sum(torch.t(assignmat), dim=1, keepdim=True)+1e-8)
#        community_embed_ = torch.div(torch.mm(torch.t(assignmat), torch.squeeze(h_pl)),\
#            torch.sum(torch.t(assignmat), dim=1, keepdim=True)+1e-8)
            #s = torch.mean(h_pl, 1)
            c_x = torch.unsqueeze(torch.mm(assignmat, community_embed_), 0)
            #s = s.expand_as(c_x)
            #c_x = torch.cat((c_x, s), dim=-1)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        print(sc_1.shape)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
        print(logits.shape)

        return logits

