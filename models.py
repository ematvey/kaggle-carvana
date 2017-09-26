import torch
import torch.nn as nn


def conv(*a, **kwa):
    return nn.Conv2d(*a, **kwa)


def convlayer(in_c, out_c, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        conv(in_c, out_c, kernel_size=kernel_size,
             stride=stride, padding=padding, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def unet_block(in_c, out_c, **kwa):
    return nn.Sequential(
        convlayer(in_c, in_c, kernel_size=3, stride=1, padding=1),
        convlayer(in_c, out_c, kernel_size=3, stride=1, padding=1),
    )


def unet_block_slim(in_c, out_c, **kwa):
    return convlayer(in_c, out_c, kernel_size=1, stride=1, padding=0)


import torch.nn.functional as F


class MyUNet(nn.Module):
    def __init__(self, **kwa):
        super().__init__()

        block = unet_block
        in_channels = 3
        S0 = 16
        S1 = 32
        S2 = 64
        S3 = 128
        S4 = 256
        S5 = 512
        S6 = 1024
        S7 = 1024

        F0 = 32
        F1 = 32
        F2 = 32

        # inputs = 1920 x 1280, immediately downsampled 2x
        self.down0 = convlayer(
            in_channels, S0, kernel_size=3, stride=2, padding=1)

        self.down1 = block(S0, S1)  # 960 x 640
        self.down2 = block(S1, S2)  # 480 x 320
        self.down3 = block(S2, S3)  # 240 x 160
        self.down4 = block(S3, S4)  # 120 x 80
        self.down5 = block(S4, S5)  # 60 x 40
        self.down6 = block(S5, S6)  # 30 x 20

        self.center = block(S6, S7)  # 15 x 10

        self.up6 = block(S7 + S6, S6)
        self.up5 = block(S6 + S5, S5)
        self.up4 = block(S5 + S4, S4)
        self.up3 = block(S4 + S3, S3)
        self.up2 = block(S3 + S2, S2)
        self.up1 = block(S2 + S1, S1)

        self.f1 = nn.Sequential(
            convlayer(S1 + S0, F0, kernel_size=3, stride=1, padding=1),
            convlayer(F0, F0, kernel_size=3, stride=1, padding=1),

            nn.ConvTranspose2d(F0, F1, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),

            convlayer(F1, F1, kernel_size=3, stride=1, padding=1),
        )
        self.f2 = block(F1 + in_channels, F2)
        self.out = nn.Conv2d(F2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        d0 = self.down0(inputs)

        d1 = self.down1(d0)
        d2 = self.down2(F.max_pool2d(d1, kernel_size=2, stride=2))
        d3 = self.down3(F.max_pool2d(d2, kernel_size=2, stride=2))
        d4 = self.down4(F.max_pool2d(d3, kernel_size=2, stride=2))
        d5 = self.down5(F.max_pool2d(d4, kernel_size=2, stride=2))
        d6 = self.down6(F.max_pool2d(d5, kernel_size=2, stride=2))

        out = self.center(F.max_pool2d(d6, kernel_size=2, stride=2))

        out = self.up6(
            torch.cat([F.upsample(out, scale_factor=2, mode='bilinear'), d6], dim=1))
        out = self.up5(
            torch.cat([F.upsample(out, scale_factor=2, mode='bilinear'), d5], dim=1))
        out = self.up4(
            torch.cat([F.upsample(out, scale_factor=2, mode='bilinear'), d4], dim=1))
        out = self.up3(
            torch.cat([F.upsample(out, scale_factor=2, mode='bilinear'), d3], dim=1))
        out = self.up2(
            torch.cat([F.upsample(out, scale_factor=2, mode='bilinear'), d2], dim=1))
        out = self.up1(
            torch.cat([F.upsample(out, scale_factor=2, mode='bilinear'), d1], dim=1))

        out = self.f1(torch.cat([out, d0], dim=1))
        out = self.f2(torch.cat([out, inputs], dim=1))
        out = self.out(out)
        out = out.squeeze(1)  # remove logits dim
        return out
