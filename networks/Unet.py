import torch.nn as nn
from .Transformer_seg import *

# unet的下采样部分

# 多头混合卷积
# class MHMC(nn.Module):
#     def __init__(self, in_channel, out_channel, num_heads=3, kernel_sizes=[3, 5, 7]):
#         super(MHMC, self).__init__()
#         self.num_heads = num_heads
#         self.kernel_sizes = kernel_sizes
#
#         self.convs = nn.ModuleList()
#         for i in range(num_heads):
#             for k in kernel_sizes:
#                 self.convs.append(nn.Sequential(
#                     nn.Conv2d(in_channel, out_channel // (num_heads * len(kernel_sizes)), k, 1, k // 2),
#                     nn.BatchNorm2d(out_channel // (num_heads * len(kernel_sizes))),
#                     nn.ReLU(inplace=True)
#                 ))
#
#     def forward(self, x):
#         outputs = []
#         for conv in self.convs:
#             outputs.append(conv(x))
#         return torch.cat(outputs, dim=1)

# 两次尺寸不变卷积



class unetConv2(nn.Module):
    def __init__(self, in_channel, out_channel, is_batchnorm=True):
        super(unetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                                       nn.BatchNorm2d(out_channel),
                                       nn.ReLU(inplace=True))
            # self.MHMC = MHMC(out_channel, out_channel)
            self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                                       nn.BatchNorm2d(out_channel),
                                       nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                                       nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                                       nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        # outputs = self.MHMC(outputs)
        outputs = self.conv2(outputs)
        return outputs

# 两次尺寸不变卷积 + 一次最大池化，尺寸缩小一半（下采样）
class unetDown(nn.Module):
    def __init__(self, in_channel, out_channel, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_channel, out_channel, is_batchnorm)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        conv_outputs = self.conv(inputs)
        outputs = self.down(conv_outputs)
        return conv_outputs, outputs

# UNet完整的下采样，从(B, 29, 400, 301)变成(B, 512, 13, 10)
# 对于openfwi数据，inchannels应该是32，不是5！
class UnetDownModel(nn.Module):
    def __init__(self, in_channel, is_batchnorm, down_sample_list=[]):
        super().__init__()
        self.in_channel = in_channel
        self.is_batchnorm = is_batchnorm
        self.unet_down_list = nn.ModuleList()
        for i in range(0, len(down_sample_list)):
            if i == 0:
                layer = unetDown(in_channel, down_sample_list[i], self.is_batchnorm)
            else:
                layer = unetDown(down_sample_list[i - 1], down_sample_list[i], self.is_batchnorm)
            self.unet_down_list.append(layer)

    def forward(self, x):
        features = []
        for module in self.unet_down_list:
            conv_output, _output = module(x)
            features.append(_output)
            x = _output
        return x, features[::-1]


# 一个卷积操作，封装了一次卷积+批归一化+激活函数
class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_fea,
                      out_channels=out_fea,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_fea),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        outputs = self.layer(x)
        return outputs

# 激活函数改变
class ConvBlock2(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(ConvBlock2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_fea,
                      out_channels=out_fea,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            # nn.BatchNorm2d(out_fea),
            # nn.InstanceNorm2d(out_fea),
            # nn.LayerNorm(out_fea),
            nn.LeakyReLU(inplace=True),
            # nn.Tanh()
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        outputs = self.layer(x)
        return outputs
# 针对于OpenFWI的pre_processing过程（SEGSalt不需要）
class PreConvBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.pre1 = ConvBlock(in_channel, 8, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.pre2 = ConvBlock(8, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre2_1 = ConvBlock(16, 16, kernel_size=3, stride=1, padding=1)
        self.pre3 = ConvBlock(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre3_1 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temp_layer = self.pre1(x)
        temp_layer = self.pre2(temp_layer)
        temp_layer = self.pre2_1(temp_layer)
        temp_layer = self.pre3(temp_layer)
        temp_layer = self.pre3_1(temp_layer)
        return temp_layer

# 上采样过程，包括多次反向池化
class DecoderCup(nn.Module):
    def __init__(self, config, n_patch):
        super().__init__()
        self.config = config
        self.n_patch = n_patch
        # 对应对底层下采样的通道数head_channels
        head_channels = config.decoder_list[0]
        # 通过一次反卷积，将transformer部分的encoder输出的通道数2048变为1024
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_node_list = config.decoder_list
        blocks = [
            DecoderBlock(decoder_node_list[i], decoder_node_list[i + 1]) for i in range(len(decoder_node_list) - 1)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.final_block = DecoderBlock(decoder_node_list[-1], decoder_node_list[-1], False)

    def forward(self, hidden_states, features=None):
        # UNet上采样，尺寸变为(B, 32, 192, 160)
        x = hidden_states
        for i, decoder_block in enumerate(self.blocks):
            feature_ = features[i]
            # feature_ = None
            x = decoder_block(x, feature_)
        # 多次1.5倍反池化，直到尺寸达到要求(B, 32, 201, 301)
        while x.shape[2] <= self.config.label_size[0] or x.shape[3] <= self.config.label_size[1]:
            x = self.final_block(x)
        return x

# 两次尺寸不变卷积+一次反卷积（或反池化）+跳跃连接（上采样）
class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_deconv=True):
        super().__init__()
        self.conv = unetConv2(in_channel, out_channel, True)

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=1.5)

    # x：上层网络输出 feature：待skip connection数据
    def forward(self, x, feature=None):
        if feature is not None:
            outputs2 = self.up(x)
            offset1 = (outputs2.size()[2] - feature.size()[2])
            offset2 = (outputs2.size()[3] - feature.size()[3])
            padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
            outputs1 = nn.functional.pad(feature, padding)
            return self.conv(torch.cat([outputs1, outputs2], dim=1))
        else:
            outputs2 = self.up(x)
            return outputs2
            # return self.conv(outputs2)

class Output(nn.Module):
    def __init__(self):
        super().__init__()
        # self.config = config
        # self.n_patch = n_patch

        # 通过一次反卷积，将transformer部分的encoder输出的通道数2048变为1024
        # self.conv_more1 = Conv2dReLU(
        #     29,
        #     16,
        #     kernel_size=3,
        #     padding=1,
        #     use_batchnorm=True,
        # )
        # 为什么没报错？
        self.conv_more1 = ConvBlock2(32, 16, kernel_size=1, stride=(1, 1), padding=0)
        self.conv_more2 = ConvBlock2(16, 4, kernel_size=1, stride=(1, 1), padding=0)
        self.conv_more3 = ConvBlock2(4, 1, kernel_size=1, stride=(1, 1), padding=0)
        # self.conv_more1 = nn.Conv2d(32, 16, kernel_size=3, stride=(1, 1), padding=1)
        # self.conv_more2 = nn.Conv2d(16, 4, kernel_size=3, stride=(1, 1), padding=1)
        # self.conv_more3 = nn.Conv2d(4, 1, kernel_size=3, stride=(1, 1), padding=1)
        # self.conv_more2 = Conv2dReLU(
        #     16,
        #     4,
        #     kernel_size=3,
        #     padding=1,
        #     use_batchnorm=True,
        # )
        # self.conv_more3 = Conv2dReLU(
        #     4,
        #     1,
        #     kernel_size=3,
        #     padding=1,
        #     use_batchnorm=True,
        # )


    def forward(self, x):
        temp = self.conv_more1(x)
        temp2 = self.conv_more2(temp)
        temp3 = self.conv_more3(temp2)

        return temp3

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)