from PathConfig import *
from LibConfig import *
import argparse


from networks.UTNet70 import UTNet70
from networks.VisionTransformer import VisionTransformer

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

################################################
########    LOADING TESTING DATA       ########
################################################
print('***************** Loading Testing DataSet *****************')

if SimulateData:
    test_set, label_set, data_dim, label_dim = SEGSimulate(data_dir, 1600, TestSize, Inchannels, split="test",
                                                                   data_dsp_blk=data_dsp_blk,
                                                                   label_dsp_blk=label_dsp_blk, model_dim=ModelDim)
elif OpenFWIData:
    # test_set, label_set = OpenFWI(data_dir, TestSize, Inchannels, split="test", start=11)
    test_set, label_set = OpenFWI(data_dir, TestSize, Inchannels, split="test", start=1)
    data_dim, label_dim = args.image_size, args.label_size
    # 速度模型归一化:
    print("正在进行速度模型归一化")
    for i in range(label_set.shape[0]):
        for j in range(label_set.shape[1]):
            temp = label_set[i, j, ...]
            label_set[i, j, ...] = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
else:
    test_set, label_set, data_dim, label_dim = SEGSalt(data_dir, 0, TestSize, Inchannels, split="test",
                                                               data_dsp_blk=data_dsp_blk,
                                                               label_dsp_blk=label_dsp_blk, model_dim=ModelDim)

datasets = data_utils.TensorDataset(torch.from_numpy(test_set), torch.from_numpy(label_set))
test_loader = data_utils.DataLoader(datasets, batch_size=TestBatchSize, shuffle=False)

################################################
########         LOAD    NETWORK        ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
# model_file = models_dir + modelname + '_epoch' + str(Epochs) + '.pkl'
test_epoch = Epochs
# 设置不同的epoch进行r测试
test_epoch = 10
model_file = models_dir + modelname + '_epoch' + str(test_epoch) + '.pkl'
# model_file = 'E:/fwi/UTNet01/models/SimulataModelTest05/SEGSimulationModel.pkl'
temp_w_file = '/res_mean_' + str(test_epoch) + '.txt'

net = VisionTransformer(config=args, data_dim=data_dim,is_open_FWI=OpenFWIData)
# net = UTNet703(config=args,is_open_FWI=OpenFWIData)
#net = VisionTransformer_SEGSalt(config=args, data_dim=data_dim,is_open_FWI=OpenFWIData)
# net = VisionTransformer_SEGSalt(config=args, data_dim=data_dim,is_open_FWI=OpenFWIData)
net.load_state_dict(torch.load(model_file))
if cuda_available:
    net=net.to(device)

# net = torch.nn.DataParallel(net,device_ids=[1])
################################################
########            TESTING             ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('            START TESTING                  ')
print('*******************************************')
print('*******************************************')
print()

# Initialization
since = time.time()
TotPSNR = np.zeros((1, TestSize), dtype=float)
TotSSIM = np.zeros((1, TestSize), dtype=float)
MSE = np.zeros((1, TestSize), dtype=float)
MAE = np.zeros((1, TestSize), dtype=float)
UQI = np.zeros((1, TestSize), dtype=float)
LPIPS = np.zeros((1, TestSize), dtype=float)

Prediction = np.zeros((TestSize, ModelDim[0], ModelDim[1]), dtype=float)
GT = np.zeros((TestSize, ModelDim[0], ModelDim[1]), dtype=float)
total = 0

lpips_object = lpips.LPIPS(net='alex', version="0.1")

for i, (images, labels) in enumerate(test_loader):
    images = images.view(TestBatchSize, Inchannels, data_dim[0], data_dim[1])
    labels = labels.view(TestBatchSize, Nclasses, label_dim[0], label_dim[1])
    images = images.to(device)
    labels = labels.to(device)

    # Predictions
    net.eval()
    outputs = net(images)
    outputs = outputs.view(TestBatchSize, ModelDim[0], ModelDim[1])
    outputs = outputs.data.cpu().numpy()
    gts = labels.data.cpu().numpy()

    # Calculate the PSNR, SSIM
    for k in range(TestBatchSize):
        pd = outputs[k, :, :].reshape(ModelDim[0], ModelDim[1])
        gt = gts[k, :, :].reshape(ModelDim[0], ModelDim[1])
        pd = turn(pd)
        gt = turn(gt)
        Prediction[i * TestBatchSize + k, :, :] = pd
        GT[i * TestBatchSize + k, :, :] = gt
        psnr = PSNR(pd, gt)
        mse = run_mse(pd, gt)
        mae = run_mae(pd, gt)
        uqi = run_uqi(gt, pd)
        lpips = run_lpips(gt, pd, lpips_object)

        TotPSNR[0, total] = psnr
        MSE[0, total] = mse
        MAE[0, total] = mae
        UQI[0, total] = uqi
        LPIPS[0, total] = lpips
        ssim = SSIM(pd.reshape(-1, 1, ModelDim[0], ModelDim[1]),
                    gt.reshape(-1, 1, ModelDim[0], ModelDim[1]))
        TotSSIM[0, total] = ssim
        print('The %d testing psnr: %.2f, SSIM: %.4f, MSE: %.4f ,MAE: %.4f ' % (total, psnr, ssim, mse, mae))
        total = total + 1
print("MSE average: ", MSE.mean() ,
      "\nMAE average: ", MAE.mean(),
      "\nUQI average: ", UQI.mean(),
      "\nLPIPS average: ", LPIPS.mean(),
      )

with open(results_dir + temp_w_file, 'w') as f:
    f.write("MSE average: {:.8f}\n".format(MSE.mean()))
    f.write("MAE average: {:.5f}\n".format(MAE.mean() ))
    f.write("UQI average: {:.5f}\n".format(UQI.mean()))
    f.write("LPIPS average: {:.5f}\n".format(LPIPS.mean()))
    f.write("PSNR average: {:.5f}\n".format(TotPSNR.mean()))
    f.write("SSIM average: {:.5f}\n".format(TotSSIM.mean()))

# Save Results
SaveTestResults(TotPSNR, TotSSIM, Prediction, GT, results_dir, MSE, MAE, UQI, LPIPS)

# Plot one prediction and ground truth
num = 0
temp_min = np.min(GT[num, :, :])
temp_max = np.max(GT[num, :, :])

PlotComparison(Prediction[num, :, :], GT[num, :, :], ModelDim, label_dsp_blk, dh, temp_min, temp_max, SavePath=results_dir, epoch=test_epoch)

# Record the consuming time
time_elapsed = time.time() - since
print('Testing complete in  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
