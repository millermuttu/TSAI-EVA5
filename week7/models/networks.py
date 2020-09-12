import torch.nn.functional as F
from __future__ import print_function
import torch
import torch.nn as nn

gbn_splits = 4
dropout_value = 0.05

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)

class NetS6(nn.Module):
    def __init__(self, batch_normalization_type="BN"):
        super(NetS6, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8) if batch_normalization_type == "BN" else GhostBatchNorm(8, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16) if batch_normalization_type == "BN" else GhostBatchNorm(16, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        )  # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16) if batch_normalization_type == "BN" else GhostBatchNorm(16, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16) if batch_normalization_type == "BN" else GhostBatchNorm(16, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16) if batch_normalization_type == "BN" else GhostBatchNorm(16, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16) if batch_normalization_type == "BN" else GhostBatchNorm(16, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )  # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

gbn_splits = 4
dropout_value = 0.05
hidden_units = [16,32,64,128,200]
class Net(nn.Module):
  def __init__(self, batch_normalization_type="BN"):
    super(Net, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=hidden_units[0], kernel_size=(3, 3), padding=2, bias=False, padding_mode = 'reflect', dilation=2), # Input=32x32x3 Kernel=3x3x1x16 Output=32x32x16 RF=3x3
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[0]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[0], gbn_splits),
        nn.Dropout(dropout_value),
        nn.Conv2d(in_channels=hidden_units[0], out_channels=hidden_units[1], kernel_size=(3, 3), padding=2, bias=False, padding_mode = 'reflect'), # Input=32x32x16 Kernel=3x3x10x32 Output=32x32x32 RF=5x5
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[1]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[1], gbn_splits),
        nn.Dropout(dropout_value),
        nn.Conv2d(in_channels=hidden_units[1], out_channels=hidden_units[1], kernel_size=(3, 3), padding=2, bias=False,padding_mode = 'reflect'), # Input=32x32x16 Kernel=3x3x10x32 Output=32x32x32 RF=5x5
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[1]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[1], gbn_splits),
        nn.Dropout(dropout_value),
        nn.MaxPool2d(kernel_size=2, stride=2)
        ) #output_size=16x16x32

    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units[1], out_channels=hidden_units[1], kernel_size=(3, 3), padding=2, bias=False, groups=hidden_units[1], padding_mode = 'reflect'), #Input=14x14x10 Kernel=3x3x10x10 Output=12x12x10 RF=10x10
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[1]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[1], gbn_splits),
        nn.Dropout(dropout_value),
        nn.Conv2d(in_channels=hidden_units[1], out_channels=hidden_units[2], kernel_size=(1,1), padding=1, bias=False, padding_mode = 'reflect'), #Input=12x12x10 Kernel=3x3x10x12 Output=10x10x12 RF=14x14
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[2]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[2], gbn_splits),
        nn.Dropout(dropout_value),
        nn.Conv2d(in_channels=hidden_units[2], out_channels=hidden_units[2], kernel_size=(3, 3), padding=2, bias=False, groups = hidden_units[2], padding_mode = 'reflect'), #Input=10x10x12 Kernel=3x3x12x16 Output=8x8x16 RF=18x18
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[2]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[2], gbn_splits),
        nn.Dropout(dropout_value),
        nn.Conv2d(in_channels=hidden_units[2], out_channels=hidden_units[3], kernel_size=(1, 1), padding=1, bias=False, padding_mode = 'reflect'), #Input=10x10x12 Kernel=3x3x12x16 Output=8x8x16 RF=18x18
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[3]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[2], gbn_splits),
        nn.Dropout(dropout_value),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )#output_size=8x8x32

    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units[3], out_channels=hidden_units[4], kernel_size=(3, 3), padding=1, bias=False, padding_mode = 'reflect'), #Input=14x14x10 Kernel=3x3x10x10 Output=12x12x10 RF=10x10
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[4]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[2], gbn_splits),
        nn.Dropout(dropout_value),
        nn.Conv2d(in_channels=hidden_units[4], out_channels=hidden_units[4], kernel_size=(3, 3), padding=1, bias=False, padding_mode = 'reflect'), #Input=14x14x10 Kernel=3x3x10x10 Output=12x12x10 RF=10x10
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[4]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[2], gbn_splits),
        nn.Dropout(dropout_value),
        nn.Conv2d(in_channels=hidden_units[4], out_channels=hidden_units[3], kernel_size=(1,1), padding=1, bias=False, padding_mode = 'reflect'), #Input=12x12x10 Kernel=3x3x10x12 Output=10x10x12 RF=14x14
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[3]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[1], gbn_splits),
        nn.Dropout(dropout_value),
        # nn.Conv2d(in_channels=hidden_units[3], out_channels=hidden_units[3], kernel_size=(3, 3), padding=1, bias=False, padding_mode = 'reflect'), #Input=10x10x12 Kernel=3x3x12x16 Output=8x8x16 RF=18x18
        # nn.ReLU(),
        # nn.BatchNorm2d(hidden_units[3]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[2], gbn_splits),
        # nn.Dropout(dropout_value),
        nn.Conv2d(in_channels=hidden_units[3], out_channels=hidden_units[3], kernel_size=(3, 3), padding=1, bias=False, padding_mode = 'reflect'), #Input=10x10x12 Kernel=3x3x12x16 Output=8x8x16 RF=18x18
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[3]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[2], gbn_splits),
        nn.Dropout(dropout_value),
        nn.Conv2d(in_channels=hidden_units[3], out_channels=hidden_units[2], kernel_size=(1, 1), padding=0, bias=False, padding_mode = 'reflect'), #Input=10x10x12 Kernel=3x3x12x16 Output=8x8x16 RF=18x18
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[2]) if batch_normalization_type=="BN" else GhostBatchNorm(hidden_units[1], gbn_splits),
        nn.Dropout(dropout_value),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )#output_size=8x8x32

    self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units[2], out_channels=hidden_units[2], kernel_size=(3, 3), padding=1, bias=False, padding_mode = 'reflect'), #Input=6x6x16 Kernel=3x3x16x16 Output=4x4x16 RF=26x26
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units[2]) if batch_normalization_type=="BN" else GhostBatchNorm(16, gbn_splits),
        nn.AvgPool2d(kernel_size=(7)), #Input=4x4x16 Kernel=4x4x16x16 Output=1x1x16
        nn.Conv2d(in_channels=hidden_units[2], out_channels=hidden_units[1], kernel_size=(1, 1), padding=0, bias=False), #Input=1x1x16 Kernel=1x1x16x10 Output=1x1x10
        nn.Conv2d(in_channels=hidden_units[1], out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )


  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=1)
