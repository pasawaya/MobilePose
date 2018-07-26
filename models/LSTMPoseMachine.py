
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.ConvLSTM import ConvLSTM
from utils.train_utils import initialize_weights_kaiming


class Processor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Processor, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_channels, 128, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 128, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 128, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 512, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),
            nn.Conv2d(512, 512, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),
            nn.Conv2d(512, out_channels, 1, padding=0),
        )

    def forward(self, x_t):
        return self.process(x_t)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, 128, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 128, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 128, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, out_channels, 5, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_t):
        return self.encode(x_t)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.generate = nn.Sequential(
            nn.Conv2d(in_channels, 128, 11, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 1, padding=0),
        )

    def forward(self, h_t):
        return self.generate(h_t)


class Stage(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super(Stage, self).__init__()

        lstm_size = hidden_channels + out_channels + 1

        self.encode = Encoder(in_channels, hidden_channels)
        self.lstm = ConvLSTM(lstm_size, 3, 1, True, device)
        self.generate = Generator(lstm_size, out_channels)

    def forward(self, f_t, b_t_1, h_t_1, c_t_1, centers):
        f_t = self.encode(f_t)
        f_t = torch.cat([f_t, b_t_1, centers], dim=1)

        h_t, c_t = self.lstm(f_t, h_t_1, c_t_1)

        b_t = self.generate(h_t)
        return b_t, h_t, c_t


class PretrainLPM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device, T=5):
        super(PretrainLPM, self).__init__()

        self.T = T

        self.process = Processor(in_channels, out_channels)
        self.stage_1 = Stage(in_channels, hidden_channels, out_channels, device)
        self.stage_t = Stage(in_channels, hidden_channels, out_channels, device)

        self.apply(initialize_weights_kaiming)

        # Per http://proceedings.mlr.press/v37/jozefowicz15.pdf
        self.stage_t.lstm.f_x.bias.data.fill_(1.0)
        self.stage_t.lstm.f_h.bias.data.fill_(1.0)

    def forward(self, x, centers):
        centers = F.avg_pool2d(centers, 9, stride=8)

        b_0 = self.process(x)
        b_1, h_1, c_1 = self.stage_1(x, b_0, None, None, centers)
        beliefs = [b_0, b_1]

        b_t_1, h_t_1, c_t_1 = b_1, h_1, c_1
        for t in range(self.T - 1):
            b_t, h_t, c_t = self.stage_t(x, b_t_1, h_t_1, c_t_1, centers)
            beliefs.append(b_t)
            b_t_1, h_t_1, c_t_1 = b_t, h_t, c_t

        out = torch.stack(beliefs, 1)
        return out
