"""
#@Author:   Frankln Kenghagho
#@Date:     04.04.2019
#@Project:  RobotVA
"""


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
 
#select a GPU if working on Multi-GPU Systems
#Several GPUs can also be selected

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from TaskManager import *

"""Template To Train RobotVQA
"""


#start the model loader
# modeldir='logs1': the location of the RobotVQA's Weight File .h5
#rootdir='/mnt/Datadisk/franklin/test/RobotVQA': absolute path to parent directory of modeldir

tkm=TaskManager(modeldir='logs1',rootdir='../../RobotVQA')
#load the training set
test_set=tkm.getDataset(binary_dataset='../dataset/virtual_training_dataset(51000_Images).data')
#load the validation set
val_set=tkm.getDataset(binary_dataset='../dataset/virtual_validation_dataset(10105_Images).data')

#depth='float32': the format of the depth image if working in RGBD mode
#op_type= training or validation
#start the training or validation
tkm.train(test_set,val_set,depth='float32', op_type='training')




