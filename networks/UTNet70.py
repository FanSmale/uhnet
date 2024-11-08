import math
from math import ceil
import torch
import torch.nn as nn

from networks.Transformer_open import Transformer_open
from networks.Transformer_seg import Transformer_seg

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn', output_lim=[18,18]):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.LeakyReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.output_lim = output_lim

    def forward(self, x):
        x = self.layers(x)
        x = nn.functional.interpolate(x, size=self.output_lim, mode='bilinear', align_corners=False)
        return x

class DeconvBlock_skip(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn', output_lim=[18,18]):
        super(DeconvBlock_skip, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.LeakyReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.output_lim = output_lim
        self.conv1 = ConvBlock(in_fea, in_fea, kernel_size = 3)
        self.conv2 = ConvBlock(2*in_fea, in_fea, kernel_size = 3)

    def forward(self, x, input):
        input = self.conv1(input)
        x = torch.cat([x, input], 1)
        x = self.conv2(x)
        x = self.layers(x)
        x = nn.functional.interpolate(x, size=self.output_lim, mode='bilinear', align_corners=False)

        return x

# 刚刚创建了跳跃连接版的上采样，现在该写网络部分了

class UTNet70(nn.Module):
    def __init__(self, dim0 = 16,dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, dim6=1024, sample_spatial=1.0, config=None, is_open_FWI=False, **kwargs):
        super(UTNet70, self).__init__()

        n_patch = [4, 4]
        # 单降
        self.convblock1 = ConvBlock(5, dim0, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim0, dim1, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim1, dim1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(8, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        # 双降
        self.convblock5_1 = ConvBlock(dim3, dim3, kernel_size=(3, 3), stride=(2, 2))
        self.convblock5_2 = ConvBlock(dim3, dim3, kernel_size=(3, 3), stride=(1, 1))
        self.convblock6_1 = ConvBlock(dim3, dim4, kernel_size=(3, 3), stride=(2, 2))
        self.convblock6_2 = ConvBlock(dim4, dim4, kernel_size=(3, 3), stride=(1, 1))
        self.convblock7_1 = ConvBlock(dim4, dim5, kernel_size=(3, 3), stride=(2, 2))
        self.convblock7_2 = ConvBlock(dim5, dim5, kernel_size=(3, 3), stride=(1, 1))


        # self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        # self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        # # 这里按照inversionnet，连着两个3*1卷积，但我论文里设计的是一个3*1一个3*3
        # self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        # self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        # self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        # self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        # self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        # self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        # self.convblock5_2 = ConvBlock(dim3, dim3)
        # self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        # self.convblock6_2 = ConvBlock(dim4, dim4)

        # 用于代替
        # self.convblock8_1 = ConvBlock(dim5, dim6, stride=2)
        # self.convblock8_2 = ConvBlock(dim6, dim6)

        # self.transformer = Transformer_open(config, n_patch, is_open_FWI)
        self.transformer = Transformer_seg(config, n_patch, is_open_FWI)

        self.deconv0_1 = DeconvBlock(dim6, dim5, kernel_size=3, stride=2, output_lim=[9, 9])
        self.deconv0_2 = ConvBlock(dim5, dim5)
        self.deconv1_1 = DeconvBlock_skip(dim5, dim4, kernel_size=3, stride=2, output_lim=[18, 18])
        self.deconv1_2 = ConvBlock(dim4, dim4)
        self.deconv2_1 = DeconvBlock_skip(dim4, dim3, kernel_size=3, stride=2, output_lim=[35, 35])
        self.deconv2_2 = ConvBlock(dim3, dim3)
        self.deconv3_1 = DeconvBlock_skip(dim3, dim2, kernel_size=3, stride=2, output_lim=[70, 70])
        self.deconv3_2 = ConvBlock(dim2, dim2)
        self.deconv4 = ConvBlock(dim2, dim1)
        self.deconv5_1 = ConvBlock(dim1, dim0)
        self.deconv5_2 = ConvBlock(dim0, dim0)
        # self.deconv6 = ConvBlock_Tanh(dim0, 1)
        # self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=2, stride=2, output_lim=[18,18])
        # self.deconv2_2 = ConvBlock(dim4, dim4)
        # self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=2, stride=2, output_lim=[35,35])
        # self.deconv3_2 = ConvBlock(dim3, dim3)
        # self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=2, stride=2, output_lim=[70,70])
        # self.deconv4_2 = ConvBlock(dim2, dim2)
        # self.deconv5_1 = ConvBlock(dim2, dim1)
        # self.deconv5_2 = ConvBlock(dim1, dim1)
        # self.deconv6 = ConvBlock_Tanh(dim1, 1)
        self.deconv6 = ConvBlock(dim0, 1)

    def forward(self, x):
        # Encoder Part
        # 单维度降维
        down1 = self.convblock1(x)  # (None, 16, 500, 70)
        down2_1 = self.convblock2_1(down1)  # (None, 32, 250, 70)
        down2_2 = self.convblock2_2(down2_1)  # (None, 32, 250, 70)
        down3_1 = self.convblock3_1(down2_2)  # (None, 64, 125, 70)
        down3_2 = self.convblock3_2(down3_1)  # (None, 64, 125, 70)
        down4_1 = self.convblock4_1(down3_2)  # (None, 128, 70, 70)
        down4_2 = self.convblock4_2(down4_1)  # (None, 128, 70, 70)
        # 同时降维
        down5_1 = self.convblock5_1(down4_2)  # (None, 128, 35, 35)
        down5_2 = self.convblock5_2(down5_1)  # (None, 128, 35, 35)

        down6_1 = self.convblock6_1(down5_2)  # (None, 256, 18, 18)
        down6_2 = self.convblock6_2(down6_1)  # (None, 256, 18, 18)
        down7_1 = self.convblock7_1(down6_2)  # (None, 512, 9, 9)
        down7_2 = self.convblock7_2(down7_1)  # (None, 512, 9, 9)
        # ViT
        center, features = self.transformer(down7_2)  # (None, 1024, 4, 4)
        # 代替vit
        # temp = self.convblock8_1(down7_2)  # (None, 512, 8, 9)
        # center = self.convblock8_2(temp)  # (None, 512, 8, 9)

        up0_1 = self.deconv0_1(center)  # (None, 512, 9, 9)
        up0_2 = self.deconv0_2(up0_1)  # (None, 512, 9, 9)

        up1_1 = self.deconv1_1(up0_2, down7_2)  # (None, 256, 18, 18)
        up1_2 = self.deconv1_2(up1_1)  # (None, 256, 18, 18)
        # skip
        up2_1 = self.deconv2_1(up1_2, down6_2)  # (None, 128, 35, 35)
        up2_2 = self.deconv2_2(up2_1)  # (None, 128, 35, 35)
        # skip
        up3_1 = self.deconv3_1(up2_2, down5_2)  # (None, 64, 70, 70)
        up3_2 = self.deconv3_2(up3_1)  # (None, 64, 70, 70)
        up4 = self.deconv4(up3_2)  # (None, 32, 70, 70)
        up5_1 = self.deconv5_1(up4)  # (None, 16, 70, 70)
        up5_2 = self.deconv5_2(up5_1)  # (None, 16, 70, 70)

        up6 = self.deconv6(up5_2)  # (None, 1, 70, 70)
        return up6


