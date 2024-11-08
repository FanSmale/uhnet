import copy
import math
import torch
import torch.nn as nn



# 完整的Transformer过程包括Embedding、Encoder和Decoder

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

class Transformer_open(nn.Module):
    def __init__(self, config, n_patch, is_open_FWI=False):
        super().__init__()
        self.hidden_size = config.patch_size[0] * config.patch_size[1] * config.down_sample_list[-1]
        # Embedding过程
        self.embeddings = Embedding(config, n_patch)
        # Encoder过程
        self.encoder = Encoder(config)
        # Decoder过程
        self.decoder = Decoder(config, n_patch)

    def forward(self, x, features = None):
        embedding_out = self.embeddings(x)
        encoded = self.encoder(embedding_out)
        decoded, features = self.decoder(encoded, features)
        return decoded, features

# Embedding过程是将(B, 512, 13, 10)卷积成(B, 2048, 6, 5)，并通过扁平、转置等操作，变成(B, 30, 2048)
# 把原本的(C,H,W)图片数据处理成(H/2 * W/2, 2*2*C)的序列数据
# Hidden_size的含义是：在transformer过程中序列数据的长度
class Embedding(nn.Module):
    def __init__(self, config, n_patche):
        super().__init__()
        # 将(B, 512, 13, 10)卷积成(B, 2048, 6, 5)
        # self.patch_embeddings = nn.Conv2d(config.down_sample_list[-1],
        #                                   config.hidden_size,
        #                                   kernel_size=config.patch_size,
        #                                   stride=config.patch_size)
        self.patch_embeddings = ConvBlock(config.down_sample_list[-1], config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size, padding=0)
        # 生成待嵌入的位置信息（这里的1是固定的吗？若batch_size不为1会出错吗？）
        # self.position_embeddings = nn.Parameter(torch.zeros(1, 475, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patche[0] * n_patche[1], config.hidden_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.patch_embeddings(x)
        # 扁平化，由[batch_size, height, width, channels]变为[batch_size, height*width*channels]
        x = x.flatten(2)
        # 转置
        x = x.transpose(-1, -2)
        # 将原图片与位置信息相加
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

# Encoder过程包含了若干个transformer层，将其封装在Block()方法中
class Encoder(nn.Module):
    def __init__(self, config, relu_slop = 0.2):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.LKRelu = nn.LeakyReLU(relu_slop, inplace=True)
        # 添加多个连续的transformer层
        for _ in range(config.num_transformer_layer):
            layer_ = Block(config)
            self.layer.append(copy.deepcopy(layer_))

    def forward(self, hidden_states):
        # 连续经过多个transformer层
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)  # (B, n_patch, hidden)
        # 经过批归一化后再输出
        encoded = self.encoder_norm(hidden_states)
        encoded = self.LKRelu(encoded)
        return encoded

# transformer层包括Attention多头自注意力机制、残差连接和Mlp前馈网络
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        # 封装好的多头自注意力机制
        self.attn = Attention(config, 2)
        # 封装好的前馈网络
        self.ffn = Mlp(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

# 多头自注意力机制，用于计算输入的注意力权重，并生成一个带有编码信息的输出向量，指示中的每个部分如何关注其他所有部分
class Attention(nn.Module):
    def __init__(self, config, band_width):
        super().__init__()
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.band_width = band_width
        self.weight = nn.Parameter(torch.zeros(1))
        # 稀疏带状注意力的权重上界
        self.bound = config.upper_bound

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def generate_band_mask(self, seq_length, band_width, device):
        mask = torch.zeros((seq_length, seq_length), device=device)
        for i in range(seq_length):
            start = max(0, i - band_width)
            end = min(seq_length, i + band_width + 1)
            mask[i, :start] = float('-inf')
            mask[i, end:] = float('-inf')
        return mask

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        weight = self.weight.clamp(0, self.bound)

        # 全局
        ###### 矩阵相乘得到权重矩阵，并归一化处理
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        # context_layer (B, num_attention_heads, n_patch, attention_head_size)

        ###### 权重矩阵与value相乘，生成一个带有编码信息的输出向量
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer (B, n_patch, num_attention_heads, attention_head_size)
        # contiguous一般与transpose，permute，view搭配使用：使用transpose或permute进行维度变换后，调用contiguous，然后方可使用view对维度进行变形
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape (B, n_patch,all_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # attention_output (B, n_patch,hidden_size)
        # 小细节 attention_head_size = int(hidden_size / num_attention_heads)，all_head_size = num_attention_heads * attention_head_size
        # 所以应该满足hidden_size能被num_attention_heads整除
        attention_output = self.out(context_layer)


        # 带状
        attention_scores1 = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)

        seq_length = hidden_states.size(1)
        band_mask = self.generate_band_mask(seq_length, self.band_width, hidden_states.device)
        band_mask = band_mask.unsqueeze(0).unsqueeze(0)  # For batch and head dimensions
        attention_scores1 = attention_scores1 + band_mask

        attention_probs1 = self.softmax(attention_scores1)

        context_layer1 = torch.matmul(attention_probs1, value_layer)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer1.view(*new_context_layer_shape1)
        attention_output1 = self.out(context_layer2)

        return attention_output + attention_output1 * weight


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act_fn = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Decoder负责将上层的序列输出(B, 30, 2048)变成图片形式(B, 1024, 6, 5)
class Decoder(nn.Module):
    def __init__(self, config, n_patch):
        super().__init__()
        self.config = config
        self.n_patch = n_patch
        # 通过一次卷积，将transformer部分的encoder输出的通道数2048变为1024
        # self.conv_more = Conv2dReLU(
        #     config.hidden_size,
        #     config.decoder_list[0],
        #     kernel_size=3,
        #     padding=1,
        #     use_batchnorm=True,
        # )
        self.conv_more = ConvBlock(config.hidden_size, config.decoder_list[0], kernel_size=3, padding=1)

    def forward(self, hidden_states, features):
        B, n_patch, hidden = hidden_states.size()  # hidden_states: (B, n_patch, hidden)
        x = hidden_states.permute(0, 2, 1)  # x: (B, hidden, n_patch)
        x = x.contiguous().view(B, hidden, self.n_patch[0], self.n_patch[1])  # x: (B, hidden, h, w)
        # 变成最终的图片形式(B, 1024, 6, 5)
        x = self.conv_more(x)
        return x, features

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# 封装了一次卷积操作，包括卷积+ReLU激活+归一化
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


class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
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