#!/usr/bin/env python
"""
#@Author:   Frankln Kenghagho
#@Date:     04.04.2019
#@Project:  RobotVA
"""



#This program is frontend and multi-task, namely
#   1- Setting of model paths 
#   2- Setting of model hyperparameters
#   3- Data preparation and loading into memory
#   4- Training
#   5- Inference
#   6- Validation
#   7- Testing
#   8- Result visualization

#setting python paths
import sys
import os
import roslib
import rospkg
rospack = rospkg.RosPack()
packname=''
packname=rospack.get_path('robotvqa_visualizer')
sys.path.append(os.path.join(packname,'../models'))
sys.path.append(os.path.join(packname,'../tools'))

import visualize
from visualize import get_ax
import pickle
import glob

import random
import math
import re
import time
import numpy as np
import cv2
import mutex
import rospy
from PIL import Image
from std_msgs.msg import( String )
import cv_bridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


from DatasetClasses import DatasetClasses
from  robotVQAConfig import RobotVQAConfig
import utils
import skimage
import json

#Select a GPU if working on Multi-GPU Systems
#Several GPUs can also be selected
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob

 
from keras import backend as K
import robotVQA as modellib

#Scene Graph Server
from rs_robotvqa_msgs.srv import *

        

################# Extended DatasetLoader Class(EDLC) ###################

class ExtendedDatasetLoader(utils.Dataset):
    """Extend the generic Dataset class and override the following methods:
    load_mask()
    """
    
    def __init__(self):
        
        super(self.__class__,self).__init__()
        
    def normalize(self,s):
        """Capitalize first letter of string s
        """
        s=s.rstrip().lstrip()
        return s[0].upper()+s[1:]
        
            
    def register_images(self,folders,imgNameRoot,annotNameRoot,depthNameRoot,config,with_depth=True,high_depth=True):
        """get all image files that pass the filter
            
           inputs:
                  mode: how to build the dataset: from a dataset file(file) or from a raw dataset(data) made up of images and annotations
                        For a quick access to large dataset, the latter is preloaded into a binary file
        """
	classes=[[],[],[],[],[]]#for 5 output_features
        nbfails=0
        nbsuccess=0
	for folder in folders:
		annotation_filter=folder+'/'+annotNameRoot+'*.json'
		annotations=glob.glob(annotation_filter)
		#Add classes
		rospy.loginfo('\nLoading classes from dataset ...\n')
		for anot in annotations:
		    try:
		        with open(anot,'r') as infile:
		            jsonImage=json.load(infile)
		        infile.close()
		        for obj in jsonImage['objects']:
		            try:
		                    cat=self.normalize(obj['objectName'])
		                    col=self.normalize(obj['objectColor'])
		                    sha=self.normalize(obj['objectShape'])
		                    mat=self.normalize(obj['objectExternMaterial'])
		                    opn=self.normalize(obj['objectOpenability'])
		                    opn=self.normalize(config.OBJECT_OPENABILITY_DICO[opn])
		                    if((cat in config.OBJECT_NAME_DICO) and (col in config.OBJECT_COLOR_DICO) and (sha in config.OBJECT_SHAPE_DICO) and \
		                        (mat in config.OBJECT_MATERIAL_DICO) and (opn in list(config.OBJECT_OPENABILITY_DICO.values()))):
		                            if cat not in classes[0]:
		                                classes[0].append(cat)
		                            if col not in classes[1]:
		                                classes[1].append(col)
		                            if sha not in classes[2]:
		                                classes[2].append(sha)
		                            if mat not in classes[3]:
		                                classes[3].append(mat)
		                            if opn not in classes[4]:
		                                classes[4].append(opn)
		                            nbsuccess+=1
		            except Exception as e:
		                rospy.logwarn('Data '+str(anot)+': An object could not be processed:'+str(e))
		                nbfails+=1    
		    except Exception as e:
		        rospy.logwarn('Data '+str(anot)+' could not be processed:'+str(e))
		        nbfails+=1
        rospy.loginfo('\n',nbsuccess,' Objects successfully found and ',nbfails,' Objects failed!', '\n')
        rospy.loginfo('\nClasses found:',classes, '\n')
        rospy.loginfo('\nRegistering classes ...\n')
        for feature_id in range(config.NUM_FEATURES-2):
            for i in range(len(classes[feature_id])):
                self.add_class(feature_id,"robotVQA",i+1,classes[feature_id][i])
        
        rospy.loginfo('\nAdding object relationships ...\n')        
        #Add relationships
        feature_id=5
        for i in range(config.NUM_CLASSES[feature_id]-1):
                 self.add_class(feature_id,"robotVQA",i+1,self.normalize(list(config.OBJECT_RELATION_DICO.keys())[i]))
                 
        rospy.loginfo('\nAdding relationship categories ...\n')        
        #Add relationship categories
        feature_id=6
        for i in range(config.NUM_CLASSES[feature_id]-1):
                 self.add_class(feature_id,"robotVQA",i+1,self.normalize(list(config.RELATION_CATEGORY_DICO.values())[i]))
        
        rospy.loginfo('\nAdding images ...\n')
	k=-1
	for folder in folders:
		image_filter=folder+'/'+imgNameRoot+'*.*'
		images=glob.glob(image_filter)
		#Add images      
		for i in range(len(images)):
		    k+=1
		    index=images[i].split(imgNameRoot)[1].split('.')[0]
		    annotationPath=folder+'/'+annotNameRoot+index+'.json'
		    if high_depth:
		    	depthPath=folder+'/'+depthNameRoot+index+'.exr'
		    else:
			depthPath=folder+'/'+depthNameRoot+index+'.jpg'
		    try:
		        image = skimage.io.imread(images[i])
		        if (os.path.exists(depthPath) or (not with_depth) ) and os.path.exists(annotationPath):
		            self.add_image("robotVQA",k,images[i],depthPath=depthPath,annotPath=annotationPath,dataFolder=folder,shape=image.shape)
		    except Exception as e:
		        rospy.logerr('Image '+str(images[i])+' could not be registered:'+str(e))
		rospy.loginfo('\nImages found in'+folder+':',len(images), '\n')
        del classes[:]
        
    
    def reduce_relation(self,relation):
        x=np.count_nonzero(relation,axis=2)
        x=np.count_nonzero(x,axis=0)+np.count_nonzero(x,axis=1)
        return x.nonzero()
           
    def make_transition(self,relation):
        N,C=relation.shape[1:]   
        for c in range(C):
            stable=False
            while not stable:
                stable=True
                for i in range(N):
                    for j in range(N):
                        for k in range(N):
                                if(relation[i][j][c]==relation[j][k][c] and relation[i][j][c]!=0 and relation[i][j][c]!=relation[i][k][c]):
                                    relation[i][k][c]=relation[i][j][c]
                                    stable=False
        return relation
        
    def load_mask(self, image_id,config):
        """Generate instance masks for objects of the given image ID.
        """
        info = self.image_info[image_id]
        
        annotationPath = info['annotPath']
        shape=info['shape']
        shape=[shape[0],shape[1]]
        mask=[]
        pose=[]
        nbfail=0
        nbsuccess=0
        classes=[[],[],[],[],[]]
        id_name_map=[]
        try:
            with open(annotationPath,'r') as infile:
                jsonImage=json.load(infile)
            infile.close()
            img=np.zeros(shape,dtype='uint8')
            for obj in jsonImage['objects']:
                try:
                        cat=self.normalize(obj['objectName'])
                        col=self.normalize(obj['objectColor'])
                        sha=self.normalize(obj['objectShape'])
                        mat=self.normalize(obj['objectExternMaterial'])
                        opn=self.normalize(obj['objectOpenability'])
                        ori=np.array(obj['objectLocalOrientation'],dtype='float32')
                        #normalize angles to principal ones
                        ori[0]=utils.principal_angle(ori[0])
                        ori[1]=utils.principal_angle(ori[1])
                        ori[2]=utils.principal_angle(ori[2])
                        pos=np.array(obj['objectLocalPosition'],dtype='float32')
                        opn=self.normalize(config.OBJECT_OPENABILITY_DICO[opn])
                        #check that objects are defined in the right bound
                        assert abs(pos[0])<=config.MAX_OBJECT_COORDINATE and \
                               abs(pos[1])<=config.MAX_OBJECT_COORDINATE and \
                               abs(pos[2])<=config.MAX_OBJECT_COORDINATE
                        if((cat in config.OBJECT_NAME_DICO) and (col in config.OBJECT_COLOR_DICO) and (sha in config.OBJECT_SHAPE_DICO) and \
                            (mat in config.OBJECT_MATERIAL_DICO) and (opn in list(config.OBJECT_OPENABILITY_DICO.values()))):
                                    id_name_map.append(obj['objectId'])
                                    classes[0].append(cat)
                                    classes[1].append(col)
                                    classes[2].append(sha)
                                    classes[3].append(mat)
                                    classes[4].append(opn)
                                    img=img*0
                                    for cord in obj['objectSegmentationPixels']:
                                            img[cord[0]][cord[1]]=1
                                    mask.append(img.copy())
                                    #register poses with normalization
                                    pose.append(np.array(list(ori)+list(utils.getPositionFromCamToImg(pos)),dtype='float32'))
                                    nbsuccess+=1
                except Exception as e:
                                    nbfail+=1
            rospy.loginfo('\n\n',nbsuccess,'/',nbsuccess+nbfail,' Object(s) found!')
            nbInstances=len(mask)
            shape.append(nbInstances)
            rospy.loginfo('\nShape:\n',shape)
            masks=np.zeros(shape,dtype='uint8')
            poses=np.zeros([nbInstances,6],dtype='float32')
            relations=np.zeros([nbInstances,nbInstances,DatasetClasses.NUM_CLASSES[6]-1],dtype='int32')
            for i in range(nbInstances):
                masks[:,:,i]=mask[i].copy()
                poses[i,:]=pose[i].copy()
            del mask[:]
            del pose[:]
            for j in range(len(classes)):
                for i in range(len(classes[j])):
                    classes[j][i]=self.class_names[j].index(classes[j][i])
                classes[j]=np.array(classes[j],dtype='int32')
            for rel in jsonImage['objectRelationship']:
                try:
                    if(rel['object1'] in id_name_map) and (rel['object2'] in id_name_map):
                        relations[id_name_map.index(rel['object1'])][id_name_map.index(rel['object2'])][self.class_names[6].index(config.OBJECT_RELATION_DICO[self.normalize(rel['relation'])])-1]=self.class_names[5].index(self.normalize(rel['relation']))
                   
                except Exception as e:
                    rospy.logerr('An object relationship could not be processed: '+str(e))
            del id_name_map[:]
            #Further processing if there are relations
            if relations.sum()!=0.:
                #augment dataset through transitivity property of relations
                #relations=self.make_transition(relations)
                #only select objects participating in a relationship                
                valid_obj=self.reduce_relation(relations)
                #take away all non valid objects masks,poses,...
                relations=relations.take(valid_obj[0],axis=1).take(valid_obj[0],axis=0)
                masks=masks.take(valid_obj[0],axis=2)
                poses=poses.take(valid_obj[0],axis=0)
                for i in range(len(classes)):
                    classes[i]=classes[i].take(valid_obj[0],axis=0)
                #merge all relation categories into a single one
                z=np.where(relations[:,:,2]>0)
                relations[:,:,1][z]=(relations[:,:,2][z])
                z=np.where(relations[:,:,1]>0)
                relations[:,:,0][z]=(relations[:,:,1][z])
            return masks,classes,poses,relations[:,:,0]
        except Exception as e:
            rospy.logerr('\n\n Data '+str(annotationPath)+' could not be processed:'+str(e))
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
    
    #Number of target feature
    NUM_FEATURES=DatasetClasses.NUM_FEATURES
    #Target features
    FEATURES_INDEX=DatasetClasses.FEATURES_INDEX
    # Number of classes per features(object's category/name, color, shape, material, openability) (including background)
    NUM_CLASSES =DatasetClasses.NUM_CLASSES # background + 3 shapes
    #categories
    OBJECT_NAME_DICO=DatasetClasses.OBJECT_NAME_DICO
    #colors
    OBJECT_COLOR_DICO=DatasetClasses.OBJECT_COLOR_DICO
    #shape
    OBJECT_SHAPE_DICO=DatasetClasses.OBJECT_SHAPE_DICO
    #material
    OBJECT_MATERIAL_DICO=DatasetClasses.OBJECT_MATERIAL_DICO
    #openability
    OBJECT_OPENABILITY_DICO=DatasetClasses.OBJECT_OPENABILITY_DICO
    #object relationships
    OBJECT_RELATION_DICO=DatasetClasses.OBJECT_RELATION_DICO
    
    #relationship categories
    RELATION_CATEGORY_DICO=DatasetClasses.RELATION_CATEGORY_DICO

    #Max Object Coordinate in cm
    MAX_OBJECT_COORDINATE=DatasetClasses.MAX_OBJECT_COORDINATE
    
    #Max CAMERA_CENTER_TO_PIXEL_DISTANCE in m for attaching useless(not in the system's focus: reduce scope of view) objects
    MAX_CAMERA_CENTER_TO_PIXEL_DISTANCE=DatasetClasses.MAX_CAMERA_CENTER_TO_PIXEL_DISTANCE
    
    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.000001
  
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = DatasetClasses.IMAGE_MIN_DIM
    IMAGE_MAX_DIM = DatasetClasses.IMAGE_MAX_DIM

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128,256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =20

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 20
    
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 20
    

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 10
    
    #Number of epochs
    NUM_EPOCHS=1000
    

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 500

    #LEARNING RATE CONTROLLER
    REL_ALPHA1=0.9999
    REL_ALPHA2=0.0001
    
    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True


    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.6

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    
    # Input image size:RGBD-Images
    IMAGE_SHAPE = [DatasetClasses.IMAGE_MAX_DIM, DatasetClasses.IMAGE_MAX_DIM, DatasetClasses.IMAGE_MAX_CHANNEL]
    
    #Object Poses' Boundaries  for normalizing objects'poses
    #poses are normalized to [0,1[
    MAX_OBJECT_POSES=DatasetClasses.MAX_OBJECT_POSES

    #Object Orientation Normalization Factor. Angles belong to [0,2pi[
    #Amgles are normalized to [0,1[
    ANGLE_NORMALIZED_FACTOR=DatasetClasses.ANGLE_NORMALIZED_FACTOR
    
    # Image mean (RGB)
    MEAN_PIXEL = DatasetClasses.MEAN_PIXEL
    
    # With depth information?
    WITH_DEPTH_INFORMATION=False
    
    #processor's names
    GPU0='/gpu:0'
    GPU1='/gpu:1'
    GPU2='/gpu:2'
    GPU3='/gpu:3'
    CPU0='/cpu:0'
    
    #Numbers of threads
    NUMBER_THREADS=14
    #Layers to exclude on very first training with a new weights file
    EXCLUDE=None
    """                          
    EXCLUDE=["robotvqa_class_logits0", "robotvqa_class_logits1","robotvqa_class_logits2","robotvqa_class_logits3","robotvqa_class_logits4",
                                        "robotvqa_class_logits5_1",'robotvqa_class_logits5_2','robotvqa_class_bn2','robotvqa_class_conv2',
                                        "mrcnn_bbox_fc","mrcnn_bbox","robotvqa_poses_fc", "robotvqa_poses",
                                        "robotvqa_poses_fc0","robotvqa_poses_fc1","robotvqa_poses_fc2",
                                        "mrcnn_mask","robotvqa_class0","robotvqa_class1","robotvqa_class2","robotvqa_class3","robotvqa_class4","robotvqa_class5"]
    """

################ Task Manager Class(TMC)##############
class TaskManager(object):
    ###################################################################
    def __init__(self):

        try:
               
            #Ros Node
	    	
            rospy.init_node('robotvqa')
	    rospy.loginfo('Starting Ros Node RobotVQA ...')
	    rospy.on_shutdown(self.cleanup)  
            self.TempImageFile=rospy.get_param('sharedImageFile','TempImageFile.jpg')
            self.TempImageFile1=rospy.get_param('sharedImageFile1','TempImageFile1.jpg')         
	    self.mainTempImageFile=rospy.get_param('sharedmainImageFile','mainTempImageFile.jpg')
	    #attributes
	    self.cvImageBuffer=[]
	    self.INDEX=-1
	    self.INSTANCEINDEX=0
	    if rospy.get_param('videomode','local')=='local':
			self.cvMode='rgb8'
	    else:
			self.cvMode='bgr8'
            self.wait=True
            self.mutex=mutex.mutex()
            self.mutex2=mutex.mutex()
            self.mutex3=mutex.mutex()
            self.mutex4=mutex.mutex()
	    self.total=0
	    self.frequency=30
            self.counter=0
	    self.success=0
            self.currentImage1=[] #current image: Server
	    self.currentImage =[] #current image: Pervasive
	    self.iheight=rospy.get_param('input_height',480)
	    self.iwidth=rospy.get_param('input_width',640)
	    self.height=rospy.get_param('output_height',1000)
	    self.width=rospy.get_param('output_width',1000)
            self.color_hint=rospy.get_param('color_hint',"")
            self.model=None
            self.color_hints={"":Image, "Compressed":CompressedImage, "raw":Image, "Raw":Image, "compressed":CompressedImage}
	    self.bridge = CvBridge()
	    rospy.logwarn('RobotVQA initialized!!!')
            

            # Root directory of the project
            self.ROOT_DIR = rospy.get_param('root_dir',os.path.join(packname,'../../RobotVQA'))
          
            # Directory to save logs and trained model
            self.MODEL_DIR = rospy.get_param('model_dir',os.path.join(self.ROOT_DIR, 'logs1'))
            
            # Local path to trained weights file
            self.ROBOTVQA_WEIGHTS_PATH =  rospy.get_param('weight_path',os.path.join(self.ROOT_DIR,"mask_rcnn_coco.h5"))

            self.config = ExtendedRobotVQAConfig()#Model config
            if not os.path.exists(self.ROBOTVQA_WEIGHTS_PATH):
               rospy.loginfo('\nThe weight path '+str(self.ROBOTVQA_WEIGHTS_PATH)+' does not exists!!!\n')
            rospy.loginfo('Root directory:'+str(self.ROOT_DIR) ) 
            rospy.loginfo('Model directory:'+str(self.MODEL_DIR) ) 
            rospy.loginfo('Weight path:'+str(self.ROBOTVQA_WEIGHTS_PATH)) 
            self.config.display()

            #load the training set's cartography
	    rospy.loginfo('Getting Dataset ...')
            binary_dataset_path=rospy.get_param('binary_dataset_path',os.path.join(self.ROOT_DIR,'dataset/virtual_training_dataset(51000_Images).data'))
	    self.train_set=self.getDataset(binary_dataset=binary_dataset_path)
            #result_path: Where should the output of RobotVQA be saved?
	    self.result_path=rospy.get_param('result_path',self.ROOT_DIR+'/result')

            rospy.loginfo('Starting RobotVQA core ...')
	    #Start Inference
	    self.inference(self.train_set,result_path=self.result_path)
            #service
            self.getSceneGraph=rospy.Service('/get_scene_graph', GetSceneGraph, self.syncImageProcessing)
	    #subscribers
            topic=rospy.get_param('input_topic','/RoboSherlock/input_image')
	    self.sub = rospy.Subscriber(topic,self.color_hints[self.color_hint],self.asyncImageProcessing)
            rospy.loginfo('\nTaskManager started successfully\n')
            rospy.logwarn('\nWaiting for images ... '+str(topic)+'\n')
        except Exception as e:
            rospy.loginfo('\n Starting TaskManager failed: ',e.args[0],'\n')
            sys.exit(-1)
    
    ###################################################################
    def resize(self,images,meanSize1,meanSize2):
		normImages=[]
		try:		
			for i in range(len(images)):
				if(images[i].shape[0]*images[i].shape[1]<meanSize1*meanSize2):
					normImages.append(np.array(cv2.resize(images[i].copy(),(meanSize1,meanSize2),
		                                          interpolation=cv2.INTER_LINEAR),dtype='uint8'))#enlarge
				else:
					normImages.append(np.array(cv2.resize(images[i].copy(),(meanSize1,meanSize2),
		                                          interpolation=cv2.INTER_AREA),dtype='uint8'))#shrink
			rospy.loginfo('Resizing of images successful')
		except:
			rospy.logwarn('Failed to normalize/resize dataset')
		return normImages

  
    ###################################################################
    def cleanup(self):
	rospy.logwarn('Shutting down RobotVQA node ...')	

	
    ###################################################################
    def showImages(self):
        k=0
        while self.mutex.testandset():
              pass
	if len(self.currentImage)>0:
		cv2.imshow("Streaming-World",self.currentImage)
                while True:
                        while self.mutex2.testandset():
                               k = cv2.waitKey(1) & 0xFF
                               if k==27:
		                    cv2.destroyWindow("Streaming-World")
                        if not self.wait:
                           break
                        self.mutex2.unlock()
		       
        self.mutex.unlock()
              
                

    ###################################################################
    
    def getDataset(self,folder=[DatasetClasses.DATASET_FOLDER], imgNameRoot=DatasetClasses.LIT_IMAGE_NAME_ROOT, annotNameRoot=DatasetClasses.ANNOTATION_IMAGE_NAME_ROOT,depthNameRoot=DatasetClasses.DEPTH_IMAGE_NAME_ROOT,binary_dataset=os.path.join(DatasetClasses.DATASET_FOLDER,DatasetClasses.DATASET_BINARY_FILE), with_depth=True,high_depth=True):
        try:
            with open(binary_dataset,'rb') as f:
                return pickle.load(f)
        except Exception as exc:
                try:
                    dataset=ExtendedDatasetLoader()
                    dataset.register_images(folder,imgNameRoot,annotNameRoot,depthNameRoot,self.config,with_depth=with_depth,high_depth=high_depth)
                    dataset.prepare()
                    return dataset
                except Exception as e:
                    rospy.logerr('Dataset creation failed: '+str(e))
                    return None
    ###################################################################        
    
    def visualize_dataset(self,dataset,nbImages):
        try:
            image_ids = np.random.choice(dataset.image_ids, nbImages)
            for image_id in image_ids:
                image = dataset.load_image(image_id,0)[:,:,:3]
                mask, class_ids = dataset.load_mask(image_id)
                visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
        except Exception as e:
            rospy.logerr('Error-Could not visualize dataset: '+str(e))

    ###################################################################
    
    def train(self,train_set,val_set,init_with='last',depth='float32',op_type='training'):
        #config= should be adequately set for training
	#op_type= training or validation.
        model = modellib.RobotVQA(mode="training", config=self.config,
                          model_dir=self.MODEL_DIR)
	self.model=model
                          
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
                            exclude=ExtendedRobotVQAConfig.EXCLUDE)
        elif init_with == "last":
            # Load the last model you trained and continue training
            model_path=model.find_last()[1]
            model.load_weights(model_path, by_name=True)
        rospy.loginfo('Weights loaded successfully from '+str(model_path))
                          
        #Train progressively all the segments of the networks

        #Training loop
	model.train(train_set, val_set,learning_rate=self.config.LEARNING_RATE, epochs=self.config.NUM_EPOCHS,layers='all',depth=depth,op_type=op_type)
        #save weights after training
        model_path = os.path.join(self.MODEL_DIR, "robotVQA.h5")
        model.keras_model.save_weights(model_path)
        rospy.loginfo('Training terminated successfully!')

    ###################################################################    
    def inference(self,dst,init_with='last',result_path=None):
        #set config for inference properly
        self.config.GPU_COUNT = 1
        self.config.IMAGES_PER_GPU = 1
        self.model = modellib.RobotVQA(mode="inference",config=self.config,model_dir=self.MODEL_DIR)
        #Weights initialization imagenet, coco, or last
        if init_with == "imagenet":
            model_path=self.model.get_imagenet_weights()
            self.model.load_weights(model_path, by_name=True,exclude=ExtendedRobotVQAConfig.EXCLUDE)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model_path=self.ROBOTVQA_WEIGHTS_PATH
            self.model.load_weights(model_path, by_name=True,
                            exclude=ExtendedRobotVQAConfig.EXCLUDE)
        elif init_with == "last":
            # Load the last model you trained and continue training
            model_path=self.model.find_last()[1]
            self.model.load_weights(model_path, by_name=True)
	self.model.keras_model._make_predict_function()
        rospy.loginfo('Weights loaded successfully from '+str(model_path))
   
   


  ###################################################################
	"""
	    def asyncImageProcessing(self,image):
		
		try:    
		        if self.counter<self.frequency:
		                self.counter+=1
		        else:
		                self.counter=0
				while self.mutex2.testandset():
				      pass   
				self.wait=False
				self.mutex2.unlock()
				while self.mutex.testandset():
				      pass  
				dst=self.train_set
				self.currentImage = self.bridge.imgmsg_to_cv2(image, self.cvMode)
				while self.mutex3.testandset():
				      pass 
				self.currentImage1=self.currentImage[:]
				self.mutex3.unlock() 
				b=self.currentImage[:,:,0].copy()
				self.currentImage[:,:,0]=self.currentImage[:,:,2].copy()
				self.currentImage[:,:,2]=b.copy()
				self.currentImage = self.resize([self.currentImage],self.iwidth,self.iheight)[0]
				cv2.imwrite(self.TempImageFile,self.currentImage)
				rospy.loginfo('Buffering of current image successful')
				image = utils.load_image(self.TempImageFile,None,self.config.MAX_CAMERA_CENTER_TO_PIXEL_DISTANCE)
				#predict
				rospy.logwarn(image.shape)
			       
			        results = self.model.detect([image], verbose=0)
			        
			        r=results[0]
				class_ids=[r['class_cat_ids'],r['class_col_ids'],r['class_sha_ids'],r['class_mat_ids'],r['class_opn_ids'],r['class_rel_ids']]
				scores=[r['scores_cat'],r['scores_col'],r['scores_sha'],r['scores_mat'],r['scores_opn'],r['scores_rel']]
			       
				visualize.display_instances(image[:,:,:3], r['rois'], r['masks'], class_ids, dst.class_names,r['poses'],[],[],get_ax(cols=2), scores=scores,\
					title='Object description',title1='Object relationships',result_path=self.result_path+'/'+self.TempImageFile)
			       
				resImage=cv2.imread(self.result_path+'/'+self.TempImageFile)    
				if(len(resImage)>0):
					self.currentImage =resImage.copy()
				#self.currentImage = self.resize([self.currentImage],self.width,self.height)[0]
				#self.showImages()
				rospy.loginfo('Inference terminated!!!')
				self.wait=True
				self.mutex.unlock()
		except Exception as e:
		    rospy.logwarn(' Failed to buffer image '+str(e))
	""" 
  ###################################################################
	    
    def asyncImageProcessing(self,image):
	
		try:    
			if self.counter<self.frequency:
			        self.counter+=1
			else:
			        self.counter=0
				while self.mutex2.testandset():
				      pass   
				self.wait=False
				self.mutex2.unlock()
				while self.mutex.testandset():
				      pass  
				dst=self.train_set
				self.currentImage = self.bridge.imgmsg_to_cv2(image, self.cvMode)
				while self.mutex3.testandset():
				      pass 
				self.currentImage1=self.currentImage[:]
				self.mutex3.unlock() 
				b=self.currentImage[:,:,0].copy()
				self.currentImage[:,:,0]=self.currentImage[:,:,2].copy()
				self.currentImage[:,:,2]=b.copy()
				self.currentImage = self.resize([self.currentImage],self.iwidth,self.iheight)[0]
				cv2.imwrite(self.TempImageFile,self.currentImage)
				rospy.loginfo('Buffering of current image successful')
				image = utils.load_image(self.TempImageFile,None,self.config.MAX_CAMERA_CENTER_TO_PIXEL_DISTANCE)
				#predict
				rospy.logwarn(image.shape)
				R=image[:,:,0].copy()
				G=image[:,:,1].copy()
				B=image[:,:,2].copy()
				image0=np.stack((R.copy()*0,G.copy(),B.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image0=np.flip(image,0)
				image1=np.stack((R.copy(),B.copy(),G.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image1=np.flip(image,1)
				image2=np.stack((B.copy(),G.copy(),R.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image2=np.flip(image1,0)
				image3=np.stack((B.copy(),R.copy(),G.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image3=np.flip(image0,1)
				image4=np.stack((G.copy(),R.copy(),B.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image4=self.resize([np.rot90(image,1)],self.iwidth,self.iheight)[0]
				image5=np.stack((G.copy(),B.copy(),R.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image5=self.resize([np.rot90(image,3)],self.iwidth,self.iheight)[0]
				image6=np.stack((R.copy(),G.copy()*0,B.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image6=image-30
				image7=np.stack((R.copy(),G.copy(),B.copy()*0,image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image7=image+30
				#images=[image0.copy(),image1.copy(),image2.copy(),image3.copy(),image.copy(),image4.copy(),image5.copy(),image6.copy(),image7.copy()]
				images=[image]
				rImages=[]
				main_ax=get_ax(cols=2)
				main_mask=[]
				main_back=[]
				list_results=[]
				merge_results={"class_ids":[],"scores":[],"boxes":[],"poses":[],"masks":[]}
				while self.mutex4.testandset():
				      pass  
                                for image in images:
					results = self.model.detect([image], verbose=0)
					list_results.append(results)
					r=results[0]
					class_ids=[r['class_cat_ids'],r['class_col_ids'],r['class_sha_ids'],r['class_mat_ids'],r['class_opn_ids'],r['class_rel_ids']]
					scores=[r['scores_cat'],r['scores_col'],r['scores_sha'],r['scores_mat'],r['scores_opn'],r['scores_rel']]
					merge_results["class_ids"].append(class_ids)
					merge_results["scores"].append(scores) 
					merge_results["boxes"].append(r['rois'])
					merge_results["poses"].append(r['poses'])
					merge_results["masks"].append(r['masks'])
				#        visualize.display_instances(image[:,:,:3], r['rois'], r['masks'], class_ids, dst.class_names,r['poses'],[],[],get_ax(cols=2), scores=scores,\
				#        title='Object description',title1='Object relationships',result_path=self.result_path+'/'+self.TempImageFile)
				#	title='Object description',title1='Object relationships',result_path=self.result_path+'/'+self.TempImageFile)
                                self.mutex4.unlock()
				rospy.loginfo("****************************** BEGIN MERGING **************************************************")
				listOfObjects,spatialRelations=self.merge_results_fct(merge_results,dst.class_names)
				print(listOfObjects,spatialRelations)
				rospy.loginfo("****************************** END MERGING **************************************************")
				visualize.display_instances_v2(images[0][:,:,:3], listOfObjects, spatialRelations,main_ax,score=True,title="",title1='',
					figsize=(16, 16),result_path=self.result_path+'/'+self.mainTempImageFile)
				
				#for i in range(len(list_results)):
				#        results=list_results[i]
				#        image=images[i]
				#	r=results[0]
				#	class_ids=[r['class_cat_ids'],r['class_col_ids'],r['class_sha_ids'],r['class_mat_ids'],r['class_opn_ids'],r['class_rel_ids']]
				#	scores=[r['scores_cat'],r['scores_col'],r['scores_sha'],r['scores_mat'],r['scores_opn'],r['scores_rel']]
				#	visualize.display_instances(image[:,:,:3], r['rois'], r['masks'], class_ids, dst.class_names,r['poses'],[],[],get_ax(cols=2), scores=scores,\
				#	title='Object description',title1='Object relationships',result_path=self.result_path+'/'+self.TempImageFile)
				#	rImages.append(self.resize([cv2.imread(self.result_path+'/'+self.TempImageFile)],self.width/3,self.height/3)[0])
				#rImages.append(self.resize([cv2.imread(self.result_path+'/'+self.mainTempImageFile)],3*(self.width/3),3*(self.height/3))[0])
				#resImage=np.concatenate(( np.concatenate((rImages[0],rImages[1],rImages[4]),axis=0) , np.concatenate((rImages[2],rImages[3],rImages[7]),axis=0),np.concatenate((rImages[5],rImages[6],rImages[8]),axis=0) ),axis=1) 
				#resImage= rImages[0].copy() 
				resImage=cv2.imread(self.result_path+'/'+self.mainTempImageFile)    
				if(len(resImage)>0):
					self.currentImage =resImage.copy()
				#self.currentImage = self.resize([self.currentImage],self.width,self.height)[0]
				#self.showImages()
				rospy.loginfo('Inference terminated!!!')
				self.wait=True
				self.mutex.unlock()
		except Exception as e:
			rospy.logwarn(' Failed to buffer image '+str(e))
	

 ################################################################################################
    def IOU(self,r1,r2):
        y1,x1,y2,x2=map(float,r1)
        yp1,xp1,yp2,xp2=map(float,r2)
        
        if abs(x1-x2)==0 or abs(y1-y2)==0 or abs(xp1-xp2)==0 or abs(yp1-yp2)==0:
           return 0
        if yp1>=y2 or y1>=yp2 or xp1>=x2 or x1>=xp2:
             return 0.0 #no intersection
        if y1>=yp1:
             if x1<=xp1:
             	return (abs(y1-min([y2,yp2]))*abs(xp1-min([x2,xp2])))/(abs(x1-x2)*abs(y1-y2)+abs(xp1-xp2)*abs(yp1-yp2)-abs(y1-min([y2,yp2]))*abs(xp1-min([x2,xp2])))
             else: 
                return (abs(y1-min([y2,yp2]))*abs(x1-min([x2,xp2])))/(abs(x1-x2)*abs(y1-y2)+abs(xp1-xp2)*abs(yp1-yp2)-abs(y1-min([y2,yp2]))*abs(x1-min([x2,xp2])))
        if yp1>=y1:
             if xp1<=x1:
             	return (abs(yp1-min([yp2,y2]))*abs(x1-min([xp2,x2])))/(abs(xp1-xp2)*abs(yp1-yp2)+abs(x1-x2)*abs(y1-y2)-abs(yp1-min([yp2,y2]))*abs(x1-min([xp2,x2])))
             else: 
                return (abs(yp1-min([yp2,y2]))*abs(xp1-min([xp2,x2])))/(abs(xp1-xp2)*abs(yp1-yp2)+abs(x1-x2)*abs(y1-y2)-abs(yp1-min([yp2,y2]))*abs(xp1-min([xp2,x2])))


 ################################################################################################
    def merge_results_fct(self,results,class_names,mainSource=0):
         listOfObjects=[]
         #get all objects
         n=len(results["class_ids"])
         m=max([-1]+map((lambda x:x.shape[0]),results["poses"]))
         if m<0:
            m=0
         mapObjectTocluster=np.zeros([n,m],dtype="int")
         for i in range(len(results["class_ids"])):
            for j in range(results["poses"][i].shape[0]):
            	listOfObjects.append({"cat":(results["class_ids"][i][0][j],results["scores"][i][0][j]),
                                      "col":(results["class_ids"][i][1][j],results["scores"][i][1][j]),
                                      "sha":(results["class_ids"][i][2][j],results["scores"][i][2][j]),
                                      "mat":(results["class_ids"][i][3][j],results["scores"][i][3][j]),
                                      "opn":(results["class_ids"][i][4][j],results["scores"][i][4][j]),
                                      "poses":results["poses"][i][j],
                                      "boxes":results["boxes"][i][j],
                                      "masks":results["masks"][i][:, :, j],
                                      "source":i,
                                      "position":j
                                     }
                                    )
         clusters=[]
         listOfclusters=[]
         #cluster the set of objects
         while listOfObjects!=[]:
             cluster=listOfObjects[0]
             del clusters[:]
             for elem in listOfObjects:
                 distance= self.IOU(cluster["boxes"],elem["boxes"])
                 rospy.loginfo(str(cluster["boxes"])+" "+str(elem["boxes"])+" distance:"+str(distance))
                 if (distance >= self.config.CLUSTER_RADIUS and cluster["cat"][0]!=elem["cat"][0]) or (distance >= self.config.CLUSTER_RADIUS1 and cluster["cat"][0]==elem["cat"][0]):
                    mapObjectTocluster[elem["source"]][elem["position"]]=len(listOfclusters)
                    clusters.append(elem)
             for elem in clusters:
                 listOfObjects.remove(elem)
             listOfclusters.append(clusters[:])
         #merging clusters into objects
         del listOfObjects[:]
         for cluster in listOfclusters:

             catVal=map((lambda x: x["cat"][0]),cluster)
             catProb=map((lambda x: x["cat"][1]),cluster)
             argmaxCat=catVal[np.argmax(catProb)]
              
             colorList=np.concatenate(map((lambda x: filter((lambda y: DatasetClasses.CVTCOLOR[1][x["source"]][y]==class_names[1][x["col"][0]]), DatasetClasses.CVTCOLOR[1][x["source"]].keys())),cluster))             

             colVal=map((lambda x: x["col"][0]),cluster)
             colProb=map((lambda x: x["col"][1]),cluster)
             if list(colorList)!=[]:
             	argmaxCol=max(set(list(colorList)),key=list(colorList).count)
             else:
                 argmaxCol=class_names[1][colVal[np.argmax(colProb)]]
      
             shaVal=map((lambda x: x["sha"][0]),cluster)
             shaProb=map((lambda x: x["sha"][1]),cluster)
             argmaxSha=shaVal[np.argmax(shaProb)]
             
             matVal=map((lambda x: x["mat"][0]),cluster)
             matProb=map((lambda x: x["mat"][1]),cluster)
             argmaxMat=matVal[np.argmax(matProb)]
         
             opnVal=map((lambda x: x["opn"][0]),cluster)
             opnProb=map((lambda x: x["opn"][1]),cluster)
             argmaxOpn=opnVal[np.argmax(opnProb)]

             poseVal=map((lambda x: x["poses"]),cluster)
             poseVal=sum(poseVal)/len(poseVal)
             
             boxVal=map((lambda x: x["boxes"]),cluster)
             boxVal=sum(boxVal)*1.0/len(boxVal)

             maskVal=map((lambda x: x["masks"]),cluster)
             maskVal=np.array(np.ceil(sum(maskVal)*1.0/len(boxVal)),dtype="uint8")

             listOfObjects.append({   "cat":(class_names[0][argmaxCat],np.max(catProb)),
                                      "col":(argmaxCol,np.max(colProb)),
                                      "sha":(class_names[2][argmaxSha],np.max(shaProb)),
                                      "mat":(class_names[3][argmaxMat],np.max(matProb)),
                                      "opn":(class_names[4][argmaxOpn],np.max(opnProb)),
                                      "poses":(poseVal,1.),
                                      "boxes":(boxVal,1.),
                                      "masks":(maskVal,1.),
                                      "source":(mainSource,1.)
                                   }
                                  )
         rospy.loginfo("Return merged objects: "+str(len(listOfObjects))+" objects")

         #spatial relationship resolution
         spatialRelations=(np.array([["BG"]*len(listOfObjects) for i in range(len(listOfObjects))],dtype="|S5"),np.ones([len(listOfObjects),len(listOfObjects)],dtype="float"))
         
         for i in range(len(results["class_ids"])):
            for j in range(results["poses"][i].shape[0]):
		for k in range(results["poses"][i].shape[0]):
                	spatialRelations[0][mapObjectTocluster[i][j]][mapObjectTocluster[i][k]]=class_names[5][results["class_ids"][i][5][j][k]]
                        spatialRelations[1][mapObjectTocluster[i][j]][mapObjectTocluster[i][k]]=results["scores"][i][5][j][k]

         del listOfclusters[:]
         return listOfObjects,spatialRelations
  
 ################################################################################################
    def syncImageProcessing(self,request):
		
	try:    
                image = self.bridge.imgmsg_to_cv2(request.query, self.cvMode)
	        b=image[:,:,0].copy()
		image[:,:,0]=image[:,:,2].copy()
		image[:,:,2]=b.copy()
		image = self.resize([image],self.iwidth,self.iheight)[0]
	        cv2.imwrite(self.TempImageFile1,image)
		rospy.loginfo('Buffering of current image successful')
		image = utils.load_image(self.TempImageFile1,None,self.config.MAX_CAMERA_CENTER_TO_PIXEL_DISTANCE)
                dst=self.train_set
	        #predict
                rospy.logwarn(image.shape)
                R=image[:,:,0].copy()
                G=image[:,:,1].copy()
                B=image[:,:,2].copy()
                image0=np.stack((R.copy()*0,G.copy(),B.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image0=np.flip(image,0)
                image1=np.stack((R.copy(),B.copy(),G.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image1=np.flip(image,1)
                image2=np.stack((B.copy(),G.copy(),R.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image2=np.flip(image1,0)
                image3=np.stack((B.copy(),R.copy(),G.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image3=np.flip(image0,1)
                image4=np.stack((G.copy(),R.copy(),B.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image4=self.resize([np.rot90(image,1)],self.iwidth,self.iheight)[0]
                image5=np.stack((G.copy(),B.copy(),R.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image5=self.resize([np.rot90(image,3)],self.iwidth,self.iheight)[0]
                image6=np.stack((R.copy(),G.copy()*0,B.copy(),image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image6=image-30
                image7=np.stack((R.copy(),G.copy(),B.copy()*0,image[:,:,3].copy(),image[:,:,4].copy(),image[:,:,5].copy(),image[:,:,6]),axis=2)#image7=image+30
                #images=[image0.copy(),image1.copy(),image2.copy(),image3.copy(),image.copy(),image4.copy(),image5.copy(),image6.copy(),image7.copy()]
                images=[image.copy()]
                rImages=[]
                main_ax=get_ax(cols=2)
                main_mask=[]
                main_back=[]
                list_results=[]
                merge_results={"class_ids":[],"scores":[],"boxes":[],"poses":[],"masks":[]}
                while self.mutex4.testandset():
		      pass 
                for image in images:
                        results = self.model.detect([image], verbose=0)
                        list_results.append(results)
                        r=results[0]
                        class_ids=[r['class_cat_ids'],r['class_col_ids'],r['class_sha_ids'],r['class_mat_ids'],r['class_opn_ids'],r['class_rel_ids']]
			scores=[r['scores_cat'],r['scores_col'],r['scores_sha'],r['scores_mat'],r['scores_opn'],r['scores_rel']]
                        merge_results["class_ids"].append(class_ids)
                        merge_results["scores"].append(scores) 
			merge_results["boxes"].append(r['rois'])
                        merge_results["poses"].append(r['poses'])
                        merge_results["masks"].append(r['masks'])
                self.mutex4.unlock()
                rospy.loginfo("****************************** BEGIN MERGING **************************************************")
                listOfObjects,spatialRelations=self.merge_results_fct(merge_results,dst.class_names)
                print(listOfObjects,spatialRelations)
                rospy.loginfo("****************************** END MERGING **************************************************")
		scenegraph=visualize.display_instances_v3(images[0][:,:,:3], listOfObjects, spatialRelations,main_ax,score=True,title="",title1='',
                         figsize=(16, 16),result_path=self.result_path+'/'+self.TempImageFile1)	
                
		rospy.loginfo('Inference terminated!!!')
                return scenegraph
	except Exception as e:
		rospy.logwarn(' Failed to buffer image '+str(e))
		return GetSceneGraphResponse()

###################################################################################################



if __name__=="__main__":
    
    try:
        #start the model loader
	tkm=TaskManager()
	
	#Infinite Loop
	#rospy.spin()
        while not rospy.is_shutdown():
           tkm.showImages()
       
    except Exception as e:
	rospy.logwarn('Shutting down RobotVQA node ...'+str(e))







