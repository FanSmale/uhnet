# -*- coding: utf-8 -*-
"""
Created on Feb 2018

@author: fangshuyang (yfs2016@hit.edu.cn)

"""
import matplotlib
from scipy.ndimage import uniform_filter

from ParamConfig import canny_upperBound

matplotlib.use('Agg')  # 使用Agg可以控制不显示绘图
import torch
import numpy as np
import torch.nn as nn
from math import log10
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import matplotlib.pyplot as plt

import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 21,
         }

def turn(GT):
    dim = GT.shape
    for j in range(0, dim[1]):
        for i in range(0, dim[0] // 2):
            temp = GT[i, j]
            GT[i, j] = GT[dim[0] - 1 - i, j]
            GT[dim[0] - 1 - i, j] = temp
    return GT


def PSNR(prediction, target):
    prediction = Variable(torch.from_numpy(prediction))
    target = Variable(torch.from_numpy(target))
    zero = torch.zeros_like(target)
    criterion = nn.MSELoss(size_average=True)
    MSE = criterion(prediction, target)
    total = criterion(target, zero)
    psnr = 10. * log10(total.item() / MSE.item())
    return psnr


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    L = 255
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(img1, img2, window_size=11, size_average=True):
    img1 = Variable(torch.from_numpy(img1))
    img2 = Variable(torch.from_numpy(img2))
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def SaveTrainResults(loss, SavePath, epoch,start=0):
    data = {}
    data['loss'] = loss
    scipy.io.savemat(SavePath + 'TrainLoss.mat', data)

    fig, ax = plt.subplots()
    # plt.plot(loss[1:], linewidth=2)
    plt.plot(np.arange(start, epoch+1, 1),loss[start:], linewidth=2)
    plt.xlabel('Num. of epochs', font2)
    plt.ylabel('MSE Loss', font2)
    plt.title('Training', font3)
    plt.xticks(np.arange(start, epoch+1, 10))

    plt.savefig(SavePath + 'TrainLoss', transparent=True)
    plt.show()
    plt.close()
    # fig, ax = plt.subplots()
    # plt.plot(loss[1:], linewidth=2)
    # ax.set_xlabel('Num. of epochs', font2)
    # ax.set_ylabel('MSE Loss', font2)
    # ax.set_title('Training', font3)
    # ax.set_xlim([1, 6])
    # ax.set_xticklabels(('0', '20', '40', '60', '80', '100'))
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     label.set_fontsize(12)
    # ax.grid(linestyle='dashed', linewidth=0.5)
    #
    # plt.savefig(SavePath + 'TrainLoss', transparent=True)
    # data = {}
    # data['loss'] = loss
    # scipy.io.savemat(SavePath + 'TrainLoss', data)
    # plt.show(fig)
    # plt.close()


def SaveTestResults(TotPSNR, TotSSIM, Prediction, GT, SavePath, MSE, MAE, UQI, LPIPS):
    data = {}
    data['TotPSNR'] = TotPSNR
    data['TotSSIM'] = TotSSIM
    data['GT'] = GT
    data['Prediction'] = Prediction
    data['Mse'] = MSE
    data['Mae'] = MAE
    data['UQI'] = UQI
    data['Lpips'] = LPIPS
    print("SaveTestResults~~~~~~~~~~~~~~~~~~~~~~~~~~")
    scipy.io.savemat(SavePath + 'TestResults.mat', data)


def PlotComparison(pd, gt, label_dsp_dim, label_dsp_blk, dh, minvalue, maxvalue, SavePath, epoch):
    PD = pd.reshape(label_dsp_dim[0], label_dsp_dim[1])
    GT = gt.reshape(label_dsp_dim[0], label_dsp_dim[1])
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    im1 = ax1.imshow(GT, extent=[0, label_dsp_dim[1] * label_dsp_blk[1] * dh / 1000., \
                                 0, label_dsp_dim[0] * label_dsp_blk[0] * dh / 1000.], vmin=minvalue, vmax=maxvalue)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, ax=ax1, cax=cax1).set_label('Velocity (m/s)')
    plt.tick_params(labelsize=12)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(14)
    ax1.set_xlabel('Position (km)', font2)
    ax1.set_ylabel('Depth (km)', font2)
    ax1.set_title('Ground truth', font3)
    ax1.invert_yaxis()
    plt.subplots_adjust(bottom=0.15, top=0.92, left=0.08, right=0.98)
    plt.savefig(SavePath + 'GT', transparent=True)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    im2 = ax2.imshow(PD, extent=[0, label_dsp_dim[1] * label_dsp_blk[1] * dh / 1000., \
                                 0, label_dsp_dim[0] * label_dsp_blk[0] * dh / 1000.], vmin=minvalue, vmax=maxvalue)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, ax=ax2, cax=cax2).set_label('Velocity (m/s)')
    plt.tick_params(labelsize=12)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontsize(14)
    ax2.set_xlabel('Position (km)', font2)
    ax2.set_ylabel('Depth (km)', font2)
    ax2.set_title('Prediction', font3)
    ax2.invert_yaxis()
    plt.subplots_adjust(bottom=0.15, top=0.92, left=0.08, right=0.98)
    plt.savefig(SavePath + 'PD' + str(epoch), transparent=True)
    # plt.show(fig1)
    # plt.show(fig2)
    print("PlotComparison~~~~~~~~~~~~~~~~~~~~~~~~~~")

    plt.close()


# def run_lpips(GT, P, lp):
#     '''
#     Evaluation metric: LPIPS
#
#     :param GT:      The ground truth
#     :param P:       The velocity model predicted by the network
#     :param lp:      LPIPS related objects
#     :return:
#     '''
#     GT_tensor = torch.from_numpy(GT)
#     P_tensor = torch.from_numpy(P)
#     return lp.forward(GT_tensor, P_tensor).item()
#
# """
# 调用方式:
# lpips_object = lpips.LPIPS(net='alex', version="0.1")
# lpi = run_lpips(velocity_model, predicted_vmod, lpips_object)
# """


def _uqi_single(GT, P, ws):
    '''
    a component of UQI metric

    :param GT:          The ground truth
    :param P:           The velocity model predicted by the network
    :param ws:          Window size
    :return:
    '''
    N = ws ** 2
    window = np.ones((ws, ws))

    GT_sq = GT * GT
    P_sq = P * P
    GT_P = GT * P

    GT_sum = uniform_filter(GT, ws)
    P_sum = uniform_filter(P, ws)
    GT_sq_sum = uniform_filter(GT_sq, ws)
    P_sq_sum = uniform_filter(P_sq, ws)
    GT_P_sum = uniform_filter(GT_P, ws)

    GT_P_sum_mul = GT_sum * P_sum
    GT_P_sum_sq_sum_mul = GT_sum * GT_sum + P_sum * P_sum
    numerator = 4 * (N * GT_P_sum - GT_P_sum_mul) * GT_P_sum_mul
    denominator1 = N * (GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
    denominator = denominator1 * GT_P_sum_sq_sum_mul

    q_map = np.ones(denominator.shape)
    index = np.logical_and((denominator1 == 0), (GT_P_sum_sq_sum_mul != 0))
    q_map[index] = 2 * GT_P_sum_mul[index] / GT_P_sum_sq_sum_mul[index]
    index = (denominator != 0)
    q_map[index] = numerator[index] / denominator[index]

    s = int(np.round(ws / 2))
    return np.mean(q_map[s:-s, s:-s])


def run_uqi(GT, P, ws=8):
    '''
    Evaluation metric: UQI

    :param P:       The velocity model predicted by the network
    :param GT:      The ground truth
    :param ws:      Size of window
    :return:
    '''
    if len(GT.shape) == 2:
        GT = GT[:, :, np.newaxis]
        P = P[:, :, np.newaxis]

    GT = GT.astype(np.float64)
    P = P.astype(np.float64)
    return np.mean([_uqi_single(GT[:, :, i], P[:, :, i], ws) for i in range(GT.shape[2])])


def run_mse(prediction, target):
    '''
    Evaluation metric: MSE

    :param prediction:  The velocity model predicted by the network
    :param target:      The ground truth
    :return:
    '''
    prediction = Variable(torch.from_numpy(prediction))
    target = Variable(torch.from_numpy(target))
    criterion = nn.MSELoss(reduction='mean')
    result = criterion(prediction, target)
    return result.item()


def run_mae(prediction, target):
    '''
    Evaluation metric: MAE

    :param prediction:  The velocity model predicted by the network
    :param target:      The ground truth
    :return:
    '''

    prediction = Variable(torch.from_numpy(prediction))
    target = Variable(torch.from_numpy(target))
    criterion = nn.L1Loss(reduction='mean')
    result = criterion(prediction, target)
    return result.item()


def _uqi_single(GT,P,ws):
    '''
    a component of UQI metric

    :param GT:          The ground truth
    :param P:           The velocity model predicted by the network
    :param ws:          Window size
    :return:
    '''
    N = ws**2
    window = np.ones((ws,ws))

    GT_sq = GT*GT
    P_sq = P*P
    GT_P = GT*P

    GT_sum = uniform_filter(GT, ws)
    P_sum =  uniform_filter(P, ws)
    GT_sq_sum = uniform_filter(GT_sq, ws)
    P_sq_sum = uniform_filter(P_sq, ws)
    GT_P_sum = uniform_filter(GT_P, ws)

    GT_P_sum_mul = GT_sum*P_sum
    GT_P_sum_sq_sum_mul = GT_sum*GT_sum + P_sum*P_sum
    numerator = 4*(N*GT_P_sum - GT_P_sum_mul)*GT_P_sum_mul
    denominator1 = N*(GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
    denominator = denominator1*GT_P_sum_sq_sum_mul

    q_map = np.ones(denominator.shape)
    index = np.logical_and((denominator1 == 0) , (GT_P_sum_sq_sum_mul != 0))
    q_map[index] = 2*GT_P_sum_mul[index]/GT_P_sum_sq_sum_mul[index]
    index = (denominator != 0)
    q_map[index] = numerator[index]/denominator[index]

    s = int(np.round(ws/2))
    return np.mean(q_map[s:-s,s:-s])

def run_uqi(GT,P,ws=8):
    '''
    Evaluation metric: UQI

    :param P:       The velocity model predicted by the network
    :param GT:      The ground truth
    :param ws:      Size of window
    :return:
    '''
    if len(GT.shape) == 2:
        GT = GT[:, :, np.newaxis]
        P = P[:, :, np.newaxis]

    GT = GT.astype(np.float64)
    P = P.astype(np.float64)
    return np.mean([_uqi_single(GT[:,:,i],P[:,:,i],ws) for i in range(GT.shape[2])])


def run_lpips(GT, P, lp):
    '''
    Evaluation metric: LPIPS

    :param GT:      The ground truth
    :param P:       The velocity model predicted by the network
    :param lp:      LPIPS related objects
    :return:
    '''
    GT_tensor = torch.from_numpy(GT)
    P_tensor = torch.from_numpy(P)
    return lp.forward(GT_tensor, P_tensor).item()


def c_weight(x):
    return canny_upperBound * (1 / (1 + np.exp(-(x - 160))))


def add_gaussian_noise(seismic_data, mu=0, sigma=0.05):
    noise = np.random.normal(mu, sigma, seismic_data.shape)
    noisy_seismic_data = seismic_data + noise
    return noisy_seismic_data

def randomly_missing_trace(seismic_data, trace_rate = 0.2, gun_rate = 0.2):
    trace_num = len(seismic_data[0, 0, 0, :])
    gun_num = len(seismic_data[0, :, 0, 0])
    zeros_trace_num = int(trace_num * trace_rate)
    zeros_gun_num = int(gun_num * gun_rate)
    random_gun_indices = np.random.choice(gun_num, zeros_gun_num, replace=False)
    # random_trace_indices = np.random.choice(trace_num, zeros_trace_num, replace=False)
    for i1 in range(len(seismic_data)):
        for i2 in random_gun_indices:
            random_trace_indices = np.random.choice(trace_num, zeros_trace_num, replace=False)
            for i3 in random_trace_indices:
                seismic_data[i1, i2, :, i3] = 0
    return seismic_data