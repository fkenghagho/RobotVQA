"""
#@Author:   Frankln Kenghagho
#@Date:     04.04.2019
#@Project:  RobotVA
"""


import numpy as np

class DatasetClasses(object):
	    """This class is platform independent and holds 
	       information about the dataset
	    """
	    #Field of view
	    FIELD_OF_VIEW=70.0
	    #Object Orientation Precision
	    OBJECT_ORIENTATION_PRECISION=1e-13
	    #Number of target feature(classes)
	    #Object category, color, shape, material, openability, poses, relationships
	    NUM_FEATURES=7
	    
	    #Target features
	    FEATURES_INDEX={'CATEGORY':0,'COLOR':1,'SHAPE':2,'MATERIAL':3,'OPENABILITY':4,'RELATION':5,'RELATION_CATEGORY':6}

	    #categories
	    OBJECT_NAME_DICO=['Tea','Juice','Plate','Mug','Bowl','Tray','Tomato','Ketchup','Salt','Muesli','Spoon','Spatula','Milk','Coffee','Knife','Cornflakes','Eggholder','Pancake','Cereal','Rice']#Any other is part of background
	    
	    #colors
	    OBJECT_COLOR_DICO=['Red', 'Orange', 'Brown', 'Yellow', 'Green', 'Blue', 'White', 'Gray', 'Black', 'Violet','Pink']
	    
	    #shape
	    OBJECT_SHAPE_DICO=['Cubic','Conical', 'Cylindrical', 'Filiform', 'Flat']
	    
	    #material
	    OBJECT_MATERIAL_DICO=['Plastic', 'Wood', 'Glass', 'Steel', 'Cartoon', 'Ceramic']
	    
	    #openability
	    OBJECT_OPENABILITY_DICO={'True':'Openable','False':'Non-Openable'}
	    
	    #object relationships
	    OBJECT_RELATION_DICO={'Left':'LeftRight','Front':'FrontBehind','In':'OnIn','On':'OnIn'}
	    
	    #relationship categories
	    RELATION_CATEGORY_DICO={0+1:'LeftRight',1+1:'FrontBehind',2+1:'OnIn'}
	    
	    #relation to color mapping
	    RELATION_COLOR_DICO={'in':(125,0,255),'on':(255,0,0),'left':(0,255,0),'front':(0,0,255)}

	    # Number of classes per features(object's category/name, color, shape, material, openability) (including background)
	    NUM_CLASSES =[1+len(OBJECT_NAME_DICO),
		          1+len(OBJECT_COLOR_DICO),
		          1+len(OBJECT_SHAPE_DICO),
		          1+len(OBJECT_MATERIAL_DICO),
		          1+len(OBJECT_OPENABILITY_DICO),
		          1+len(OBJECT_RELATION_DICO),
		          1+len(RELATION_CATEGORY_DICO)]  # background + 3 shapes

	    #Max Object Coordinate in cm
	    MAX_OBJECT_COORDINATE=420
	    
	    #Max CAMERA_CENTER_TO_PIXEL_DISTANCE in m
	    MAX_CAMERA_CENTER_TO_PIXEL_DISTANCE=np.sqrt(3.*(MAX_OBJECT_COORDINATE**2))/100.
	    
	    #Image Shape
	    IMAGE_MIN_DIM = 512
	    IMAGE_MAX_DIM = 640
	    IMAGE_MAX_CHANNEL=7
	    
	    
	    # Image mean (RGB)
	    MEAN_PIXEL = np.array([127.5, 127.5, 127.5,127.5, 127.5, 127.5,127.5])
	    
	    #Camera Intrinsic Matrix
	    CAMERA_INTRINSIC_MATRIX=np.array([[457,   0,     320],[0,     457,   240],[0 ,      0  ,   1]],dtype='float')
	    
	    #PIXELS SIZE in cm
	    PIXEL_SIZE=0.1555
	    
	    
	    #Object Orientation Normalization Factor. Angles belong to [0,2pi[
	    #Amgles are normalized to [0,1[
	    ANGLE_NORMALIZED_FACTOR=2*np.pi
	    
	    #Object Poses' Boundaries  for normalizing objects'poses
	    #poses are normalized to [0,1[
	    MAX_OBJECT_POSES=np.array([ANGLE_NORMALIZED_FACTOR,ANGLE_NORMALIZED_FACTOR,ANGLE_NORMALIZED_FACTOR,IMAGE_MAX_DIM,IMAGE_MAX_DIM,MAX_OBJECT_COORDINATE/PIXEL_SIZE])
	    
	    
	    
	    #Properties of image files 
	    DATASET_BINARY_FILE='dataset.data'
	    DATASET_FOLDER='/home/franklin/franklin/test/dataset35'
	    SCREENSHOT_FOLDER='C:\\Users\\Franklin\\Desktop\\masterthesis\\P12-VisionScanning-UR16\\VisionScanning\\Saved\\Screenshots\\Windows'
	    LIT_IMAGE_NAME_ROOT='litImage'
	    NORMAL_IMAGE_NAME_ROOT='normalImage'
	    DEPTH_IMAGE_NAME_ROOT='depthImage'
	    MASK_IMAGE_NAME_ROOT='maskImage'
	    ANNOTATION_IMAGE_NAME_ROOT='annotation'
	    ANNOTATION_FILE_EXTENSION='json'
	    IMAGE_FILE_EXTENSION='jpg'
	    DEPTH_FILE_EXTENSION='exr'
	    NUMBER_IMAGES=0
	    INDEX=32821
	    CAMERA_ID=0
	    ORIENTATION_DELTA=360
	    ALTER_CHOICE_MAX_ITERATION=50
	    #with or without connection to UE4: values={'offline','online'}
	    MODE='offline'
	    #resume from actual scene or restart for a new scene:values={'continue','restart'}
	    STATE='continue' 
	    
	    #Canonical relationships between object in the scene
	    ACTOR_IDS={0:'A_Musli_1',1:'A_Mug_11',2:'A_Spoon_1',3:'A_Cereal_Nesquik_2',4:'A_Milch_Voll_3'}
	    # Only use relations={ 'left','front','under','on','in','has','valign'} for optimization.
	    # The other relationships will be deductively generated. if A is left B,then B is right A.
	    RELATIONSHIP_MAP=[[1,'on',0],[2,'in',1],[4,'on',3],[0,'left',3]]
	    #Actors' stacking graph: shows how actors are stacked in the scene
	    ACTOR_STACKING_GRAPH={0:0,1:0,2:1,3:3,4:3}#i:j means actor i is directly contained(on/in) by actor j. j=i implies no contenance.
	    
	    CB_OBJECT_FILTER={"Muesli":{"col":["Green","Orange", "Yellow","Brown","Blue"]},"Spoon":{"col":["Blue", "Orange", "Red", "Violet"]}, "Milk":{"col":["Blue", "White"]}, "Mug":{"col":["Brown", "Orange", "Red", "White", "Black","Gray","Yellow"]},  "Bowl":{"col":["Orange", "Red", "White"]}}
	    #Contenance relationships
	    CONTENANCE_RELATIONSHIPS=['on','in']
	    #Actor common temporary pose
	    ACTOR_COMMON_TEMP_LOCATION='-651 -701 30'
	    ACTOR_COMMON_TEMP_ROTATION='0 0 0'
            #Objects that are poorly featured
            POOR_FEATURED_OBJECTS=["Tray"]
            #Semantic Color Mappings from RGB to ["0GB","RBG","BGR","BRG","RGB","GRB","GBR","R0B","RG0"]
            CVTCOLOR=[
          ["0GB","RBG","BGR","BRG","RGB","GRB","GBR","R0B","RG0"],
        
          [  
            {"Red":"Red" ,"Orange":"Orange","Brown":"Brown","Yellow":"Yellow","Green":"Green","Blue":"Blue","White":"White","Gray":"Gray","Black":"Black",'Violet':"Violet",'Pink':"Pink"},
            {"Red":"Black","Orange":"Green","Brown":"Green","Yellow":"Green","Green":"Green","Blue":"Blue","White":"Blue","Gray":"Green","Black":"Black","Violet":"Blue","Pink":"Blue"},         
            {"Red":"Red","Orange":"Red","Brown":"Violet","Yellow":"Violet","Green":"Blue","Blue":"Green","White":"White","Gray":"Gray","Black":"Black","Violet":"Green",'Pink':"Yellow"},
            {"Red":"Blue" ,"Orange":"Blue","Brown":"Blue","Yellow":"Green","Green":"Green","Blue":"Red","White":"White","Gray":"Gray","Black":"Black","Violet":"Pink",'Pink':"Pink"},
            {"Red":"Green","Orange":"Green","Brown":"Green","Yellow":"Green","Green":"Blue","Blue":"Red","White":"White","Gray":"Gray","Black":"Black",'Violet':"Brown",'Pink':"Yellow"},
            {"Red":"Green","Orange":"Green","Brown":"Green","Yellow":"Yellow","Green":"Red","Blue":"Blue","White":"White","Gray":"Gray","Black":"Black",'Violet':"Green",'Pink':"Green"},
            {"Red":"Blue","Orange":"Violet","Brown":"Violet","Yellow":"Pink","Green":"Red","Blue":"Green","White":"White","Gray":"Gray","Black":"Black",'Violet':"Green",'Pink':"Green"},
            {"Red":"Red" ,"Orange":"Red","Brown":"Red","Yellow":"Red","Green":"Black","Blue":"Blue","White":"Violet","Gray":"Violet","Black":"Black",'Violet':"Violet",'Pink':"Violet"},
            {"Red":"Red" ,"Orange":"Orange","Brown":"Brown","Yellow":"Yellow","Green":"Green","Blue":"Black","White":"Yellow","Gray":"Yellow","Black":"Black",'Violet':"Red",'Pink':"Orange"}
          ]
         ]
            
	    
	    
	    def __init__(self):
		pass
