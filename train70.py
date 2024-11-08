from PathConfig import *
from LibConfig import *
import numpy as np
import argparse

from func.loss import CombinedLoss
from networks.UTNet70 import UTNet70


parser = argparse.ArgumentParser()

parser.add_argument('--down_sample_list', type=list,
                    default=DownSamplingList, help='downsampling node list')
parser.add_argument('--decoder_list', type=list,
                    default=DecoderList, help='decoder node list')
parser.add_argument('--image_size', type=list,
                    default=DataDim, help='image_size')
parser.add_argument('--label_size', type=list,
                    default=ModelDim, help='label_size')
parser.add_argument('--patch_size', type=list,
                    default=PatchSize, help='patch_size')
parser.add_argument('--in_channel', type=int,
                    default=Inchannels, help='input channel number')
parser.add_argument('--hidden_size', type=int,
                    default=HiddenSize, help='size of the hidden layer')
parser.add_argument('--mlp_dim', type=int,
                    default=MLPDim, help='dim of the mlp')
parser.add_argument('--num_heads', type=int,
                    default=NumHeads, help='number of transformer heads')
parser.add_argument('--num_transformer_layer', type=int,
                    default=NumTransformerLayer, help='number of the transformer layer')
parser.add_argument('--upper_bound', type=int,
                    default=UpperBound1, help='upper_bound')
args = parser.parse_args()


# 读取数据_data_set[:instance_size]
if SimulateData:
    train_set, label_set = SEGSimulate(data_dir, 0, TrainSize, Inchannels, split="train",
                                                                data_dsp_blk=data_dsp_blk,
                                                                label_dsp_blk=label_dsp_blk, model_dim=ModelDim)

    # 读取.mat文件使用
    # train_set, label_set, data_dim, label_dsp_dim = SEGSimulate(data_dir, 0, TrainSize, Inchannels, split="train",
    #                                                                 data_dsp_blk=data_dsp_blk,
    #                                                                 label_dsp_blk=label_dsp_blk, model_dim=ModelDim)
    data_dim, label_dsp_dim = args.image_size, args.label_size
elif OpenFWIData:
    train_set, label_set = OpenFWI(data_dir, TrainSize, Inchannels, split="train", start=1)
    data_dim, label_dsp_dim = args.image_size, args.label_size
    # 速度模型归一化:
    print("正在进行速度模型归一化")
    for i in range(label_set.shape[0]):
        for j in range(label_set.shape[1]):
            temp = label_set[i, j, ...]
            label_set[i, j, ...] = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
else:
    train_set, label_set, data_dim, label_dsp_dim = SEGSalt(data_dir, 0, TrainSize,
                                                                Inchannels,
                                                                split="train",
                                                                data_dsp_blk=data_dsp_blk,
                                                                label_dsp_blk=label_dsp_blk, model_dim=ModelDim)
datasets = data_utils.TensorDataset(torch.from_numpy(train_set), torch.from_numpy(label_set))
train_loader = data_utils.DataLoader(datasets, batch_size=BatchSize, shuffle=True)

# 选择GPU or CPU
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
# 引入网络
# net = VisionTransformer(config=args, data_dim=data_dim, is_open_FWI=OpenFWIData)
net = UTNet70(config=args, is_open_FWI=OpenFWIData)
if cuda_available:
    net = net.to(device)

# If ReUse, it will load saved model from premodelfilepath and continue to train
if ReUse:
    print('***************** Loading the pre-trained model *****************')
    premodel_file = premodel_dir + premodelname + '.pkl'
    print(premodel_file)
    # Load generator parameters
    net.load_state_dict(torch.load(premodel_file))

    net = net.to(device)
    print('Finish downloading:', str(premodel_file))


# 定义一个优化器对象
optimizer = torch.optim.Adam(net.parameters(), lr=LearnRate)

################################################
########            TRAINING            ########
################################################
print('*******************************************')
print('*******************************************')
print('           START TRAINING                  ')
print('*******************************************')
print('*******************************************')

print('Original data dimention:%s' % str(DataDim))
print('Original label dimention:%s' % str(ModelDim))
print('Training size:%d' % int(TrainSize))
print('Traning batch size:%d' % int(BatchSize))
print('Number of epochs:%d' % int(Epochs))
print('Learning rate:%.5f' % float(LearnRate))

# Initialization
loss1 = 0.0
step = np.int16(TrainSize / BatchSize)
start = time.time()


for epoch in range(Epochs):

    epoch_loss = 0.0
    # loss的动态权重
    canny_weight = round(c_weight(epoch), 2)
    # 稀疏带状注意力的动态权重
    if epoch == 180:
        args.upper_bound = UpperBound2
    since = time.time()
    for i, (images, labels) in enumerate(train_loader):
        iteration = epoch * step + i + 1
        net.train()

        # 修改数据尺寸，由(B, 29, 120400)转换成(B, 29, 400, 301)
        images = images.view(BatchSize, Inchannels, data_dim[0], data_dim[1])
        labels = labels.view(BatchSize, Nclasses, label_dsp_dim[0], label_dsp_dim[1])
        images = images.to(device)
        labels = labels.to(device)

        # 参数梯度置零
        optimizer.zero_grad()

        # 当前batch的地震数据投入网络
        outputs = net(images)

        # 以样本损失平均值来计算均方误差
        # loss = F.mse_loss(outputs, labels, reduction='mean')
        # loss = F.huber_loss(outputs, labels, delta=50000, reduction='mean')
        # loss = F.huber_loss(outputs, labels, reduction='mean')

        combined_loss = CombinedLoss(mse_weight=1-canny_weight,  canny_weight=canny_weight, low_threshold=5,high_threshold=20)
        loss = combined_loss(outputs, labels)

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')
        epoch_loss += loss.item()

        # 计算梯度
        loss.backward()
        # 更新权重
        optimizer.step()

        # 打印损失信息
        if iteration % DisplayStep == 0:
            print('Epoch: {}/{}, Iteration: {}/{} --- Training Loss:{:.6f}'.format(epoch + 1, \
                                                                                   Epochs, iteration, \
                                                                                   step * Epochs, loss.item()))

    # 打印损失和耗时的批次信息
    if (epoch + 1) % 1 == 0:
        print('Epoch: {:d} finished ! Loss: {:.5f}'.format(epoch + 1, epoch_loss / i))
        loss1 = np.append(loss1, epoch_loss / i)
        time_elapsed = time.time() - since
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 每十个批次保存一次
    if (epoch + 1) % SaveEpoch == 0:
        torch.save(net.state_dict(), models_dir + modelname + '_epoch' + str(epoch + 1) + '.pkl')
        print(models_dir + modelname + '_epoch' + str(epoch + 1) + '.pkl')
        print('Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))

# 打印总耗时
time_elapsed = time.time() - start
print('Training complete in {:.0f}m  {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# 保存损失
if not OpenFWIData:
    SaveTrainResults(loss=loss1 / 1e4, SavePath=results_dir, epoch=Epochs, start=PainStart)
else:
    SaveTrainResults(loss=loss1, SavePath=results_dir, epoch=Epochs, start=PainStart)
