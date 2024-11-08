# import scipy.io
# import numpy as np
# from skimage.measure import block_reduce
#
# """
# 通过.mat文件读取数据
# """
#
# # main_data_dir = "C:/Users/hp/Desktop/UTNet01/data/SEGSaltData/"
# '''
# base_dir SEGSalt数据的目录
# start 读取数据的起始编号（从0开始）
# instance_size 读取数据的数量
# split 取值为"train"或"test"
# data_dsp_blk 地震数据的下采样块
# label_dsp_blk 速度模型的下采样块
# DataDim 地震数据的尺寸
# ModelDim 速度模型的尺寸
# '''
# def SEGSalt(base_dir, start, instance_size, in_channels, split, data_dsp_blk, label_dsp_blk, model_dim):
#     print("data_dir: ", base_dir)
#     # 数据路径拼接
#     data_path = base_dir + "{}_data/georec_{}".format(split, split)
#     label_path = base_dir + "{}_data/vmodel_{}".format(split, split)
#
#     # 遍历所有的地震数据及速度模型
#     # data_set和label_set是还未处理的地震数据与速度模型
#     for i in range(start, start+instance_size):
#         data_set_ = scipy.io.loadmat(data_path + '/srec' + str(i + 1))['Rec']
#         label_set_ = scipy.io.loadmat(label_path + '/svmodel' + str(i + 1))['svmodel']
#
#         # 打印周期为10
#         if (i + 1) % 10 == 0:
#             print(data_path + '/srec' + str(i + 1))
#
#         # Change the dimention [h, w, c] --> [c, h, w]
#         # data1_set和label1_set是已经处理好的单张地震剖面数据或者一份速度模型
#         # data2_set和label2_set是处理好的一份的地震数据和速度模型
#         for j in range(0, in_channels):
#             # 拿到其中一张地震剖面
#             data1_set = np.float32(data_set_[:, :, j])
#             data1_set = np.float32(data1_set)
#             # 下采样
#             data1_set = block_reduce(data1_set, block_size=data_dsp_blk, func=decimate)
#             # 将下采样后的图片尺寸保存
#             data_dim = data1_set.shape
#             # 一维化
#             data1_set = data1_set.reshape(1, data_dim[0] * data_dim[1])
#             # 将下采样后的每一张地震剖面加入train_set
#             if j == start:
#                 data2_set = data1_set
#             else:
#                 data2_set = np.append(data2_set, data1_set, axis=0)
#         # 同理，对速度模型做同样操作
#         label1_set = np.float32(label_set_).reshape(model_dim)
#         # Label downsampling
#         label1_set = block_reduce(label1_set, block_size=label_dsp_blk, func=np.max)
#         label_dim = label1_set.shape
#         label1_set = label1_set.reshape(1, label_dim[0] * label_dim[1])
#         label2_set = np.float32(label1_set)
#
#         # data3_set和label3_set是处理好的完整的地震数据和速度模型
#         # 把每次循环处理后的数据一个个加入data3_set和label3_set
#         data3_set = data2_set
#         label3_set = label2_set
#         if i == start:
#             data3_set = data2_set
#             label3_set = label2_set
#         else:
#             data3_set = np.append(data3_set, data2_set, axis=0)
#             label3_set = np.append(label3_set, label2_set, axis=0)
#
#     data3_set = data3_set.reshape((instance_size, in_channels, data_dim[0] * data_dim[1]))
#     label3_set = label3_set.reshape((instance_size, 1, label_dim[0] * label_dim[1]))
#
#     return data3_set, label3_set, data_dim, label_dim
#
# # downsampling function by taking the middle value
# def decimate(a, axis):
#     idx = np.round((np.array(a.shape)[np.array(axis).reshape(1, -1)] + 1.0) / 2.0 - 1).reshape(-1)
#     downa = np.array(a)[:, :, idx[0].astype(int), idx[1].astype(int)]
#     return downa

import os
import random
import h5py
import numpy as np
import torch
import os

import numpy as np
from skimage.measure import block_reduce
import skimage
import scipy.io
from IPython.core.debugger import set_trace

"""
通过.mat文件读取数据
"""


# main_data_dir = "D:/Zhang-Xingyi/Data for FWI/SEGSaltData/"
def SEGSalt(base_dir, start, instance_size, in_channels, split, data_dsp_blk, label_dsp_blk, model_dim):

    if split == 'train':
        data_path = base_dir+'train_data/seismic'
        label_path = base_dir+'train_data/vmodel'
    else:
        data_path = base_dir+"test_data/seismic"
        label_path = base_dir+"test_data/vmodel"

    for i in range(start, start + instance_size):
        # data_set_ = scipy.io.loadmat(data_path + '/srec' + str(i + 1))
        data_set_ = scipy.io.loadmat(data_path + '/seismic' + str(i + 1))['data']
        label_set_ = scipy.io.loadmat(label_path + '/vmodel' + str(i + 1))['data']
        if (i + 1) % 10 == 0:
            print(data_path + '/srec' + str(i + 1))
        # Change the dimention [h, w, c] --> [c, h, w]
        for k in range(0, in_channels):
            data1_set = np.float32(data_set_[:, :, k])
            data1_set = np.float32(data1_set)
            # Data downsampling
            # note that the len(data11_set.shape)=len(block_size.shape)=2
            data1_set = block_reduce(data1_set, block_size=data_dsp_blk, func=decimate)
            data_dsp_dim = data1_set.shape
            data1_set = data1_set.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
            if k == 0:
                train1_set = data1_set
            else:
                train1_set = np.append(train1_set, data1_set, axis=0)

        data2_set = np.float32(label_set_).reshape(model_dim)
        # Label downsampling
        data2_set = block_reduce(data2_set, block_size=label_dsp_blk, func=np.max)
        label_dsp_dim = data2_set.shape
        data2_set = data2_set.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
        data2_set = np.float32(data2_set)
        if i == start:
            train_set = train1_set
            label_set = data2_set
        else:
            train_set = np.append(train_set, train1_set, axis=0)
            label_set = np.append(label_set, data2_set, axis=0)

    train_set = train_set.reshape((instance_size, in_channels, data_dsp_dim[0] * data_dsp_dim[1]))
    label_set = label_set.reshape((instance_size, 1, label_dsp_dim[0] * label_dsp_dim[1]))

    return train_set, label_set, data_dsp_dim, label_dsp_dim


# downsampling function by taking the middle value
def decimate(a, axis):
    idx = np.round((np.array(a.shape)[np.array(axis).reshape(1, -1)] + 1.0) / 2.0 - 1).reshape(-1)
    downa = np.array(a)[:, :, idx[0].astype(int), idx[1].astype(int)]
    return downa
