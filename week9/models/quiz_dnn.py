from __future__ import print_function
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils.normalization import *
gbn_splits = 4
dropout_value = 0.05

hidden_units = [16,32,64,128,200]
class Quiz_net(nn.Module):
    def __init__(self, batch_normalization_type="BN"):
        super(Quiz_net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32) if batch_normalization_type == "BN" else GhostBatchNorm(8, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32) if batch_normalization_type == "BN" else GhostBatchNorm(16, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 24

        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64) if batch_normalization_type == "BN" else GhostBatchNorm(32, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 10
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64) if batch_normalization_type == "BN" else GhostBatchNorm(16, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 8
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64) if batch_normalization_type == "BN" else GhostBatchNorm(16, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 6

        self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 12

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128) if batch_normalization_type == "BN" else GhostBatchNorm(16, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 6

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128) if batch_normalization_type == "BN" else GhostBatchNorm(16, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 6

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128) if batch_normalization_type == "BN" else GhostBatchNorm(16, gbn_splits),
            nn.Dropout(dropout_value)
        )  # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )  # output_size = 1

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool1(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = self.convblock9(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)