import torch
import torch.nn as nn

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

    def forward(self, c, cl, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.zeros((h_pl.shape[1], h_pl.shape[2]))
        #print('cl: ', cl)
        #print('c.shape: ', c.shape)
        for i in range(h_pl.shape[1]):
            c_x[i] = c[cl[i]]
        c_x = torch.unsqueeze(c_x, 0)
        c_x = c_x.to("cuda:0")
        #print(c_x.size())
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        #for i in range(h_pl.shape[])

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

