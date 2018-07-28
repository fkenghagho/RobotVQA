import numpy as np
class DatasetClasses(object):
    """This class is platform independent and holds 
       information about classes found in the dataset
    """
    #Field of view
    FIELD_OF_VIEW=70.0
    
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
    OBJECT_RELATION_DICO={'Left':'LeftRight','Right':'LeftRight','Front':'FrontBehind','Behind':'FrontBehind','Over':'OverUnder','Under':'OverUnder','Valign':'OverUnder','In':'OnIn','On':'OnIn'}
    
    #relationship categories
    RELATION_CATEGORY_DICO={0+1:'LeftRight',1+1:'FrontBehind',2+1:'OverUnder',3+1:'OnIn'}

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
    IMAGE_MAX_CHANNEL=6
    
    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9,173., 122., 127.])
    
    #Properties of image files 
    
    DATASET_FOLDER='D:/dataset16'
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
    MODE='online'
    #resume from actual scene or restart for a new scene:values={'continue','restart'}
    STATE='continue' 
    
    #Canonical relationships between object in the scene
    #ACTOR_IDS={0:'A_Ice_Tea_1',1:'A_EggHolder_1',2:'A_Tray_1',3:'A_Pancake_1',4:'A_Coffee_ElBryg_1',5:'A_Plate_2',6:'A_Ketchup_1',7:'A_Salz_12',8:'A_Albi_Juice_1',9:'A_Knife_1',10:'A_Spatula_16',11:'A_CornFlakes_1',12:'A_Reis_Spitzen_1',13:'A_Musli_1',14:'A_Tomato_1',15:'A_Mug_11',16:'A_Spoon_1',17:'A_Cereal_Nesquik_2',18:'A_Bowl_13',19:'A_Plate_4',20:'A_Milch_Voll_3'}
    # Only use relations={ 'left','front','under','on','in','has','valign'} for optimization.
    # The other relationships will be deductively generated. if A is left B,then B is right A.
    #RELATIONSHIP_MAP=[[1,'on',0],[3,'in',2],[4,'in',2],[3,'left',4],[5,'on',4],[6,'in',5],[7,'in',5],[6,'left',7],[9,'on',8],[10,'on',9],[12,'on',11],[13,'on',12],[14,'on',13],[15,'on',14],[16,'in',15],[18,'on',17],[19,'in',18],[20,'in',19],[0,'left',2],[8,'left',11],[11,'left',17],[0,'front',8],[0,'front',11],[0,'front',17],[2,'front',8],[2,'front',11],[2,'front',17]]
    #Actors' stacking graph: shows how actors are stacked in the scene
    #ACTOR_STACKING_GRAPH={0:0,1:0,2:2,3:2,4:2,5:4,6:5,7:5,8:8,9:8,10:9,11:11,12:11,13:12,14:13,15:14,16:15,17:17,18:17,19:18,20:19}#i:j means actor i is directly contained(on/in) by actor j. j=i implies no contenance.
    ACTOR_IDS={0:'A_Musli_1',1:'A_Mug_11',2:'A_Spoon_1',3:'A_Cereal_Nesquik_2',4:'A_Milch_Voll_3'}
    
    # Only use relations={ 'left','front','under','on','in','has','valign'} for optimization.
    # The other relationships will be deductively generated. if A is left B,then B is right A.
    RELATIONSHIP_MAP=[[1,'on',0],[2,'in',1],[4,'on',3],[0,'left',3]]
    #Actors' stacking graph: shows how actors are stacked in the scene
    ACTOR_STACKING_GRAPH={0:0,1:0,2:1,3:3,4:3}#i:j means actor i is directly contained(on/in) by actor j. j=i implies no contenance.
    
    #Contenance relationships
    CONTENANCE_RELATIONSHIPS=['on','in']
    #Actor common temporary pose
    ACTOR_COMMON_TEMP_LOCATION='-651 -701 30'
    ACTOR_COMMON_TEMP_ROTATION='0 0 0'
    
    
   
    
    def __init__(self):
        pass