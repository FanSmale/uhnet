# -*- coding: utf-8 -*-
from LibConfig import *
from ParamConfig import *


main_dir = ''

# datapath
main_data_dir = ""
data_dir = ''
if SimulateData:
    data_dir = main_data_dir + 'SEGSimulate/'
elif OpenFWIData:
    data_dir = main_data_dir + open_data_type +'/'
else:
    data_dir = main_data_dir + 'SEGSalt/'


## results and models path
if os.path.exists('./results/') and os.path.exists('./models/'):
    results_dir = main_dir + 'results/'
    models_dir = main_dir + 'models/'
else:
    os.makedirs('./results/')
    os.makedirs('./models/')
    results_dir = main_dir + 'results/'
    models_dir = main_dir + 'models/'

if SimulateData:
    results_dir = results_dir + 'SimulateResults' + Test_order
    models_dir = models_dir + 'SimulataModel' + Test_order
elif OpenFWIData:

    results_dir = results_dir+'ForBatchSize50/' + 'OpenFWIResults' + open_data_type + Test_order
    models_dir = models_dir + 'ForBatchSize50/' + 'OpenFWIModel' + open_data_type + Test_order
else:
    results_dir = results_dir + 'SEGSaltResults' + Test_order
    models_dir = models_dir + 'SEGSalt' + Test_order

if os.path.exists(results_dir) and os.path.exists(models_dir):
    results_dir = results_dir
    models_dir = models_dir
else:
    os.makedirs(results_dir)
    os.makedirs(models_dir)
    results_dir = results_dir
    models_dir = models_dir

# models name
if SimulateData:
    tagM = 'Simulate'
elif OpenFWIData:
    tagM = 'OpenFWI'
else:
    tagM = 'SEGSalt'
tagM0 = '_TranUNetModel'
tagM1 = '_TrainSize' + str(TrainSize)
tagM2 = '_Epoch' + str(Epochs)
tagM3 = '_BatchSize' + str(BatchSize)
tagM4 = '_LR' + str(LearnRate)
tagM5 = '_AttentionHeads' + str(NumHeads)
tagM6 = '_TransLayer' + str(NumTransformerLayer)
modelname = tagM + tagM0 + tagM1 + tagM2 + tagM3 + tagM4 + tagM5 + tagM6

premodel_dir=''
premodelname = ''
