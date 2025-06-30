import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self):  # <-- fixed here
        super(UNet, self).__init__()  # <-- fixed here
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.upconv_block4 = conv_block(1024, 512)
        self.upconv_block3 = conv_block(512, 256)
        self.upconv_block2 = conv_block(256, 128)
        self.upconv_block1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = F.max_pool2d(c1, 2)

        c2 = self.conv2(p1)
        p2 = F.max_pool2d(c2, 2)

        c3 = self.conv3(p2)
        p3 = F.max_pool2d(c3, 2)

        c4 = self.conv4(p3)
        p4 = F.max_pool2d(c4, 2)

        b = self.bottleneck(p4)

        u4 = self.upconv4(b)
        u4 = torch.cat((u4, c4), dim=1)
        u4 = self.upconv_block4(u4)

        u3 = self.upconv3(u4)
        u3 = torch.cat((u3, c3), dim=1)
        u3 = self.upconv_block3(u3)

        u2 = self.upconv2(u3)
        u2 = torch.cat((u2, c2), dim=1)
        u2 = self.upconv_block2(u2)

        u1 = self.upconv1(u2)
        u1 = torch.cat((u1, c1), dim=1)
        u1 = self.upconv_block1(u1)

        out = self.final_conv(u1)
        return torch.sigmoid(out)  # pixel values in [0, 1]