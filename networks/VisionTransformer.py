import copy
import math
import torch
import torch.nn as nn


from .Transformer_seg import *

from .Unet import *


# 该类封装了完整的网络结构，从UNet下采样到transformer的embedding、encoder和decoder，最后到UNet上采样和1.5倍池化调整和最终输出
'''
data_dim和label_dim 地震数据的原始尺寸和速度模型的原始尺寸
n_patch 是地震数据降维的最终尺寸，SEGSalt是(6,5)

'''


class VisionTransformer(nn.Module):
    def __init__(self, config, data_dim, is_open_FWI=False):
        super().__init__()
        temp_size0 = data_dim[0]
        temp_size1 = data_dim[1]
        # 如果是OpenFWI数据，那么需要在时间域单方面降维，如下是手动模拟计算时间域降维的过程
        if is_open_FWI:
            self.pre_conv_block = PreConvBlock(config.in_channel)
            temp_size0 = math.floor((temp_size0 - 7 + 2 * 3) / 2 + 1)
            temp_size0 = math.floor((temp_size0 - 3 + 2 * 2) / 2 + 1)
            temp_size0 = math.floor((temp_size0 - 3 + 2 * 2) / 2 + 1)
        # 手动模拟两个维度同时降维
        for i in range(len(config.down_sample_list)):
            temp_size0 = math.ceil((temp_size0) / 2)
            temp_size1 = math.ceil((temp_size1) / 2)
        temp_size0 = math.ceil((temp_size0 - 1) / config.patch_size[0])
        temp_size1 = math.ceil((temp_size1 - 1) / config.patch_size[1])
        n_patch = [temp_size0, temp_size1]
        # 在pre-processing过程中openfwi数据的通道数已经改变，因此在下采样之前需要改变
        if is_open_FWI:
            temp_in_channel = 32
        else:
            temp_in_channel = config.in_channel
        # 封装UNet下采样过程
        self.UNet_Down = UnetDownModel(in_channel=temp_in_channel, is_batchnorm=True,
                                       down_sample_list=config.down_sample_list)
        # 封装transformer过程
        # self.transformer = Transformer(config, n_patch, is_open_FWI)
        self.transformer = Transformer_seg(config, n_patch, is_open_FWI)
        # self.transformer = Transformer_MultiScale(config, n_patch, is_open_FWI)
        # 封装UNet上采样过程
        self.UNet_Up = DecoderCup(config, n_patch)
        self.config = config
        # 把上采样的结果卷积成为模型输出
        self.out_put_layer = nn.Conv2d(
            config.decoder_list[-1],
            1,
            1
        )
        self.is_open_FWI = is_open_FWI

    def forward(self, x):
        # 根据场景执行pre_processing过程
        if self.is_open_FWI:
            x = self.pre_conv_block(x)
        # 执行UNet下采样
        x, features = self.UNet_Down(x)
        # 执行Transformer过程
        x, features = self.transformer(x, features)
        # 取消skip-connection
        # x, features = self.transformer(x)
        # 执行UNet上采样过程
        x = self.UNet_Up(x, features)
        # 此时尺寸相比于最终输出偏大，截取中间的部分
        temp_offset1 = int((x.shape[2] - self.config.label_size[0]) / 2)
        temp_offset1 = temp_offset1 - 1 if temp_offset1 > 0 else temp_offset1
        temp_offset2 = int((x.shape[3] - self.config.label_size[1]) / 2)
        temp_offset2 = temp_offset2 - 1 if temp_offset2 > 0 else temp_offset2
        x = x[:, :, temp_offset1:temp_offset1 + self.config.label_size[0],
            temp_offset2: temp_offset2 + self.config.label_size[1]].contiguous()
        # 最后一次卷积，为模型输出
        result = self.out_put_layer(x)
        return result
