import torch
import torch.nn as nn
import torch.nn.functional as f


class ResConv(nn.Module):

    def __init__(self, input_channels, output_channels, ):
        super(ResConv, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),
                                   )
        self.BN_Relu = nn.Sequential(
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),
                                 )
        self.conv2 = nn.Sequential(nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   )
        self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        out = self.conv2(x1) + x2
        out = self.BN_Relu(out)
        return out


class StartConv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(StartConv, self).__init__()
        self.conv = ResConv(input_channels, output_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownConv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                  ResConv(input_channels, output_channels),
                                  )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(input_channels, input_channels//2, kernel_size=2, stride=2)

        self.conv = ResConv(input_channels, output_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff1 = x2.shape[2] - x1.shape[2]
        diff2 = x2.shape[3] - x1.shape[3]
        x1 = f.pad(x1, pad=(diff2 // 2, diff2 - diff2 // 2, diff1 // 2, diff1 - diff1 // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class StopConv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(StopConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=1))

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.inc = StartConv(1, 16)
        self.down1 = DownConv(16, 32)
        self.down2 = DownConv(32, 64)
        self.down3 = DownConv(64, 128)
        self.down4 = DownConv(128, 256)
        self.up1 = UpConv(256, 128)
        self.up2 = UpConv(128, 64)
        self.up3 = UpConv(64, 32)
        self.up4 = UpConv(32, 16)
        self.outc = StopConv(16, 1)

    def forward(self, x):

        xin = self.inc(x)
        xd1 = self.down1(xin)
        xd2 = self.down2(xd1)
        xd3 = self.down3(xd2)
        xd4 = self.down4(xd3)
        xu1 = self.up1(xd4, xd3)
        xu2 = self.up2(xu1, xd2)
        xu3 = self.up3(xu2, xd1)
        xu4 = self.up4(xu3, xin)
        xout = self.outc(xu4)
        return xout
