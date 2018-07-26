
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTM(nn.Module):
    def __init__(self, nc):
        super(ConvLSTM, self).__init__()
        self.i_x = nn.Conv2d(nc, nc, 3, padding=1)
        self.o_x = nn.Conv2d(nc, nc, 3, padding=1)
        self.g_x = nn.Conv2d(nc, nc, 3, padding=1)
        self.f_x = nn.Conv2d(nc, nc, 3, padding=1)

        self.i_h = nn.Conv2d(nc, nc, 3, padding=1)
        self.o_h = nn.Conv2d(nc, nc, 3, padding=1)
        self.g_h = nn.Conv2d(nc, nc, 3, padding=1)
        self.f_h = nn.Conv2d(nc, nc, 3, padding=1)

    def forward(self, x, h_prev, c_prev):
        i = F.sigmoid(self.i_x(x) + self.i_h(h_prev))
        o = F.sigmoid(self.o_x(x) + self.o_h(h_prev))
        g = F.sigmoid(self.g_x(x) + self.g_h(h_prev))
        f = F.sigmoid(self.f_x(x) + self.f_h(h_prev))
        c_t = i.mul(g).add(f.add(c_prev))
        h_t = o.mul(F.tanh(c_t))
        return h_t, c_t
