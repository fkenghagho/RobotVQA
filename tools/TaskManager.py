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



from DatasetClasses import DatasetClasses
import pickle
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

#setting python paths
sys.path.append('../models')
import robotVQA as modellib



        

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
		print('\nLoading classes from dataset ...\n')
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
		                print('Data '+str(anot)+': An object could not be processed:'+str(e))
		                nbfails+=1    
		    except Exception as e:
		        print('Data '+str(anot)+' could not be processed:'+str(e))
		        nbfails+=1
        print('\n',nbsuccess,' Objects successfully found and ',nbfails,' Objects failed!', '\n')
        print('\nClasses found:',classes, '\n')
        print('\nRegistering classes ...\n')
        for feature_id in range(config.NUM_FEATURES-2):
            for i in range(len(classes[feature_id])):
                self.add_class(feature_id,"robotVQA",i+1,classes[feature_id][i])
        
        print('\nAdding object relationships ...\n')        
        #Add relationships
        feature_id=5
        for i in range(config.NUM_CLASSES[feature_id]-1):
                 self.add_class(feature_id,"robotVQA",i+1,self.normalize(list(config.OBJECT_RELATION_DICO.keys())[i]))
                 
        print('\nAdding relationship categories ...\n')        
        #Add relationship categories
        feature_id=6
        for i in range(config.NUM_CLASSES[feature_id]-1):
                 self.add_class(feature_id,"robotVQA",i+1,self.normalize(list(config.RELATION_CATEGORY_DICO.values())[i]))
        
        print('\nAdding images ...\n')
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
		        print('Image '+str(images[i])+' could not be registered:'+str(e))
		print('\nImages found in'+folder+':',len(images), '\n')
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
            print('\n\n',nbsuccess,'/',nbsuccess+nbfail,' Object(s) found!')
            nbInstances=len(mask)
            shape.append(nbInstances)
            print('\nShape:\n',shape)
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
                    print('An object relationship could not be processed: '+str(e))
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
            print('\n\n Data '+str(annotationPath)+' could not be processed:'+str(e))
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
    DETECTION_NMS_THRESHOLD = 0.1

    
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
    NUMBER_THREADS=32
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
    def __init__(self,modeldir=None,rootdir=None,weightpath=None,pythonpath=None):

        try:
            
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
                    print('Dataset creation failed: '+str(e))
                    return None
            
    
    def visualize_dataset(self,dataset,nbImages):
        try:
            image_ids = np.random.choice(dataset.image_ids, nbImages)
            for image_id in image_ids:
                image = dataset.load_image(image_id,0)[:,:,:3]
                mask, class_ids = dataset.load_mask(image_id)
                visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
        except Exception as e:
            print('Error-Could not visualize dataset: '+str(e))
    
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
        print('Weights loaded successfully from '+str(model_path))
                          
        #Train progressively all the segments of the networks

        #Training loop
	model.train(train_set, val_set,learning_rate=self.config.LEARNING_RATE, epochs=self.config.NUM_EPOCHS,layers='all',depth=depth,op_type=op_type)
        





        #save weights after training
        model_path = os.path.join(self.MODEL_DIR, "robotVQA.h5")
        model.keras_model.save_weights(model_path)
        print('Training terminated successfully!')
    
    def inference(self,dst,input_image_paths,init_with='last',result_path=None):
        #set config for inference properly
        self.config.GPU_COUNT = 1
        self.config.IMAGES_PER_GPU = 1
        model = modellib.RobotVQA(mode="inference",config=self.config,model_dir=self.MODEL_DIR)
	self.model=model
        #Weights initialization imagenet, coco, or last
        if init_with == "imagenet":
            model_path=model.get_imagenet_weights()
            model.load_weights(model_path, by_name=True,exclude=ExtendedRobotVQAConfig.EXCLUDE)
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
        print('Weights loaded successfully from '+str(model_path))
        #load image
	for input_image_path in input_image_paths:
		image = utils.load_image(input_image_path[0],input_image_path[1],self.config.MAX_CAMERA_CENTER_TO_PIXEL_DISTANCE)
                #predict
                results = model.detect([image], verbose=0)
	        r=results[0]
		class_ids=[r['class_cat_ids'],r['class_col_ids'],r['class_sha_ids'],r['class_mat_ids'],r['class_opn_ids'],r['class_rel_ids']]
		scores=[r['scores_cat'],r['scores_col'],r['scores_sha'],r['scores_mat'],r['scores_opn'],r['scores_rel']]
		visualize.display_instances(image[:,:,:3], r['rois'], r['masks'], class_ids, dst.class_names,r['poses'], scores=scores, axs=get_ax(cols=2),\
		title='Object description',title1='Object relationships',result_path=result_path+'/'+input_image_path[0].split('/').pop())

	print('Inference terminated!!!')


