"""
Refined U-Net architecture for skin lesion segmentation.
Improvements over original:
- Added Dropout for regularization
- Added residual-style attention gate connections
- Better weight initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block with BatchNorm, ReLU and optional Dropout."""
    def __init__(self, in_c, out_c, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate to focus on relevant regions during upsampling."""
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        # Align spatial dims
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi


class UNet(nn.Module):
    """
    Refined U-Net with attention gates and dropout regularization.
    Input:  (B, 3, 256, 256)
    Output: (B, 1, 256, 256) - raw logits (apply sigmoid for mask probability)
    """
    def __init__(self, use_attention=True, dropout=0.1):
        super().__init__()
        self.use_attention = use_attention

        # Encoder
        self.d1 = DoubleConv(3, 64, dropout)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128, dropout)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(128, 256, dropout)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(256, 512, dropout)
        self.p4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bridge = DoubleConv(512, 1024, dropout)

        # Decoder
        self.u1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.c1 = DoubleConv(1024, 512, dropout)
        self.u2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c2 = DoubleConv(512, 256, dropout)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c3 = DoubleConv(256, 128, dropout)
        self.u4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c4 = DoubleConv(128, 64, dropout)

        # Attention gates
        if use_attention:
            self.att1 = AttentionGate(512, 512, 256)
            self.att2 = AttentionGate(256, 256, 128)
            self.att3 = AttentionGate(128, 128, 64)
            self.att4 = AttentionGate(64, 64, 32)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder path
        s1 = self.d1(x)
        s2 = self.d2(self.p1(s1))
        s3 = self.d3(self.p2(s2))
        s4 = self.d4(self.p3(s3))

        # Bottleneck
        b = self.bridge(self.p4(s4))

        # Decoder path with skip connections (+ optional attention)
        u1 = self.u1(b)
        if self.use_attention:
            s4 = self.att1(u1, s4)
        u1 = self.c1(torch.cat([u1, s4], dim=1))

        u2 = self.u2(u1)
        if self.use_attention:
            s3 = self.att2(u2, s3)
        u2 = self.c2(torch.cat([u2, s3], dim=1))

        u3 = self.u3(u2)
        if self.use_attention:
            s2 = self.att3(u3, s2)
        u3 = self.c3(torch.cat([u3, s2], dim=1))

        u4 = self.u4(u3)
        if self.use_attention:
            s1 = self.att4(u4, s1)
        u4 = self.c4(torch.cat([u4, s1], dim=1))

        return self.out(u4)  # raw logits
