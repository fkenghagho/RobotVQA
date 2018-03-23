#This program is frontend and multi-task, namely
#   1- Setting of model paths 
#   2- Setting of model hyperparameters
#   3- Data preparation and loading into memory
#   4- Training
#   5- Inference
#   6- Validation
#   7- Testing
#   8- Result visualization

#@Author:   Frankln Kenghagho
#@Date:     19.03.2018

import glob
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from  robotVQAConfig import RobotVQAConfig
import utils
import visualize
from visualize import get_ax
import skimage
import json


################# Extended DatasetLoader Class(EDLC) ###################

class ExtendedDatasetLoader(utils.Dataset):
    """Extend the generic Dataset class and override the following methods:
    load_mask()
    """
    
    def __init__(self):
        self.OBJECT_NAME_DICO=['Tea','Juice','Plate','Mug','Bowl','Tray','Tomato','Ketchup','Salz','Milch','Spoon','Spatula','Milk','Coffee','Cookie','Knife','Cornflakes','Cornflake','Eggholder','EggHolder', 'Cube']#Any other is part of background
        super(self.__class__,self).__init__()

    def register_images(self,folder,imgNameRoot,annotNameRoot):
        """get all image files that pass the filter
        """
        image_filter=folder+'/'+imgNameRoot+'*.*'
        annotation_filter=folder+'/'+annotNameRoot+'*.json'
        images=glob.glob(image_filter)
        annotations=glob.glob(annotation_filter)
        classes=[]
        #Add classes
        print('\nLoading classes from dataset ...\n')
        for anot in annotations:
            try:
                with open(anot,'r') as infile:
                    jsonImage=json.load(infile)
                infile.close()
                for obj in jsonImage['objects']:
                    if (obj['objectName'] not in classes) and (obj['objectName'] in self.OBJECT_NAME_DICO):
                        classes.append(obj['objectName'])
            except Exception as e:
                print('Data '+str(anot)+' could not be processed:'+str(e))
        print('\nClasses found:',classes, '\n')
        print('\nRegistering classes ...\n')
        for i in range(len(classes)):
            self.add_class("robotVQA",i+1,classes[i])
        print('\nAdding images ...\n')
        #Add images      
        for i in range(len(images)):
            index=images[i].split(imgNameRoot)[1].split('.')[0]
            annotationPath=folder+'/'+annotNameRoot+index+'.json'
            try:
                image = skimage.io.imread(images[i])
                if os.path.exists(annotationPath):
                    self.add_image("robotVQA",i,images[i],annotPath=annotationPath,dataFolder=folder,shape=image.shape)
            except Exception as e:
                print('Image '+str(images[i])+' could not be registered:'+str(e))
        del classes[:]
        print('\nImages found:',len(images), '\n')
           
        
    
    def load_mask(self, image_id):
        """Generate instance masks for objects of the given image ID.
        """
        info = self.image_info[image_id]
        annotationPath = info['annotPath']
        shape=info['shape']
        shape=[shape[0],shape[1]]
        mask=[]
        classes=[]
        try:
            with open(annotationPath,'r') as infile:
                jsonImage=json.load(infile)
            img=np.zeros(shape,dtype='uint8')
            for obj in jsonImage['objects']:
                if obj['objectName'] in self.OBJECT_NAME_DICO:#not a background
                    classes.append(obj['objectName'])#add class of the current instance
                    img=img*0
                    for cord in obj['objectSegmentationPixels']:
                        img[cord[0]][cord[1]]=1
                    mask.append(img.copy())
            nbInstances=len(mask)
            shape.append(nbInstances)
            print(shape)
            masks=np.zeros(shape,dtype='uint8')
            for i in range(nbInstances):
                masks[:,:,i]=mask[i].copy()
            del mask[:]
            for i in range(len(classes)):
                classes[i]=self.class_names.index(classes[i])
            return masks,np.array(classes,dtype='int32') 
        except Exception as e:
            print('Data '+str(annotationPath)+' could not be processed:'+str(e))
            return super(self.__class__,self).load_mask(image_id)
      

################ EXtended Model Configuration Class(EMCC)##############
class ExtendedRobotVQAConfig(RobotVQAConfig):
    """Configuration for training on the specific robotVQA  dataset.
    Derives from the base Config class and overrides values specific
    to the robotVQA dataset.
    """
    # Give the configuration a recognizable name
    NAME = "robotVQA"

    # Train on 1 GPU and 1 image per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 16  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    






################ Task Manager Class(TMC)##############
class TaskManager(object):
    def __init__(self,modeldir=None,rootdir=None,weightpath=None,pythonpath=None):

        try:
            #setting python paths
            if pythonpath!=None:
                for path in pythonpath:
                    sys.path.insert(0,path)
            # Root directory of the project
            if rootdir==None:
                self.ROOT_DIR = os.getcwd()
            else:
                self.ROOT_DIR=rootdir
            # Directory to save logs and trained model
            if modeldir==None:
                modeldir='logs'
            self.MODEL_DIR = os.path.join(self.ROOT_DIR, modeldir)
            
            # Local path to trained weights file
            if weightpath==None:
                weightpath="mask_rcnn_coco.h5"
            self.ROBOTVQA_WEIGHTS_PATH = os.path.join(self.ROOT_DIR,weightpath)
            self.config = ExtendedRobotVQAConfig()#Model config
            if not os.path.exists(self.ROBOTVQA_WEIGHTS_PATH):
               print('\nThe weight path '+str(self.ROBOTVQA_WEIGHTS_PATH)+' does not exists!!!\n')
            print('Root directory:'+str(self.ROOT_DIR) ) 
            print('Model directory:'+str(self.MODEL_DIR) ) 
            print('Weight path:'+str(self.ROBOTVQA_WEIGHTS_PATH)) 
            self.config.display()
            print('\nTaskManager started successfully\n')
        except Exception as e:
            print('\n Starting TaskManager failed: ',e.args[0],'\n')
            sys.exit(-1)
    
    
    
    def getDataset(self,folder, imgNameRoot, annotNameRoot):
        try:
            dataset=ExtendedDatasetLoader()
            dataset.register_images(folder,imgNameRoot,annotNameRoot)
            dataset.prepare()
            return dataset
        except Exception as e:
            print('Dataset creation failed: '+str(e))
            return None
    
    
    def visualize_dataset(self,dataset,nbImages):
        try:
            image_ids = np.random.choice(dataset.image_ids, nbImages)
            for image_id in image_ids:
                image = dataset.load_image(image_id)
                mask, class_ids = dataset.load_mask(image_id)
                visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
        except Exception as e:
            print('Error-Could not visualize dataset: '+str(e))
    
    def train(self,dataset,init_with='coco'):
        from robotVQA import log
        import robotVQA as modellib
        #config= should be adequately set for training
        model = modellib.MaskRCNN(mode="training", config=self.config,
                          model_dir=self.MODEL_DIR)
                          
        #Weights initialization imagenet, coco, or last
        if init_with == "imagenet":
            model_path=model.get_imagenet_weights()
            model.load_weights(model_path, by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model_path=self.ROBOTVQA_WEIGHTS_PATH
            model.load_weights(model_path, by_name=True,
                            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model_path=model.find_last()[1]
            model.load_weights(model_path, by_name=True)
        print('Weights loaded successfully from '+str(model_path))
                          
        #first training phase: only dataset specific layers(error) are trained
        #the second dataset is dedicated to evaluation and consequenty needs to be set differently than the first one
        model.train(dataset, dataset,learning_rate=self.config.LEARNING_RATE, epochs=5,layers='heads')
        #second training phase:all layers are revisited for fine-tuning
        #the second dataset is dedicated to evaluation and consequenty needs to be set differently than the first one
        #model.train(dataset, dataset,learning_rate=self.config.LEARNING_RATE/10, epochs=1,layers='all')
        #save weights after training
        model_path = os.path.join(self.MODEL_DIR, "robotVQA.h5")
        model.keras_model.save_weights(model_path)
        print('Training terminated successfully!')
    
    def inference(self,input_image_path,init_with='last'):
        from robotVQA import log
        import robotVQA as modellib
        #set config for inference properly
        self.config.GPU_COUNT = 1
        self.config.IMAGES_PER_GPU = 1
        model = modellib.MaskRCNN(mode="inference",config=self.config,model_dir=self.MODEL_DIR)
        #Weights initialization imagenet, coco, or last
        if init_with == "imagenet":
            model_path=model.get_imagenet_weights()
            model.load_weights(model_path, by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model_path=self.ROBOTVQA_WEIGHTS_PATH
            model.load_weights(model_path, by_name=True,
                            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model_path=model.find_last()[1]
            model.load_weights(model_path, by_name=True)
        print('Weights loaded successfully from '+str(model_path))
        #load image
        image = skimage.io.imread(input_image_path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        #predict
        results = model.detect([image], verbose=1)
        r = results[0]
        dst=self.getDataset('c:/MSCOCO/val2014','litImage','annotation')
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dst.class_names, r['scores'], ax=get_ax())




