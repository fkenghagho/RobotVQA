"""
#@Author:   Frankln Kenghagho
#@Date:     04.04.2019
#@Project:  RobotVA
"""

import os
#Select a GPU if working on Multi-GPU Systems
#Several GPUs can also be selected
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
from TaskManager import *

"""Template For Testing RobotVQA
"""



#start the model loader
tkm=TaskManager(modeldir='logs1',rootdir='../../RobotVQA')

#load the training set's cartography
train_set=tkm.getDataset(binary_dataset='../dataset/virtual_training_dataset(51000_Images).data')

#result_path: Where should the output of RobotVQA be saved?
result_path=tkm.ROOT_DIR+'/result'

#'../../realtestdataset/litImage*.jpg': Mask of the images that RobotVQA should process
#images=[[color-image1,depth-image1],...,[color-imageN,depth-imageN]]
#N=1: Online Processing
#N>1: Batch Processing

img=glob.glob('../testsamples/litImage*.jpg')
images=[[e,'../testsamples/depthImage'+e.split('litImage')[1].split('.')[0]+'.jpg'] for e in img]

#Start Inference
tkm.inference(train_set,images,result_path=result_path)

