# -*- coding: utf-8 -*-
"""
Import libraries

Created on Feb 2018

@author: fangshuyang (yfs2016@hit.edu.cn)

"""

################################################
########            LIBARIES            ########
################################################

import numpy as np
import torch
import os, sys

sys.path.append(os.getcwd())
import time
import pdb
import argparse
import torch
import torch.nn as nn
import scipy.io
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from func.utils import *
from func.SEGSimulate import SEGSimulate
from func.SEGSalt import SEGSalt
from func.OpenFWI import OpenFWI
# from func.DataLoad_Test import DataLoad_Test
from func.utils import turn, PSNR, SSIM,run_mae,run_mse,run_lpips,run_uqi
import lpips
# from networks.Unet import *
# from networks.Transformer import *
# from networks.VisionTransformer import *
