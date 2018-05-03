import numpy as np
import os 
import glob
import sys
import time
import StringIO, PIL.Image
from unrealcv import client
import json
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shutil import copyfile
 
R=[]


objectColor=['pink','red','orange','brown','yellow','olive','green','blue','purple','white','gray','black']
class Dataset(object):
    #mode=offline(without connection to image server. Used when processing existing data)/online(With connection to image server)
    #state=continue/restart
    def __init__(self,folder,nberOfImages,offset,cameraId,mode="offline", state="continue"):
        self.T=[]
        self.folder=folder
        self.litImage='litImage'
        self.normalImage='normalImage'
        self.depthImage='depthImage'
        self.maskImage='maskImage'
        self.annotation='annotation'
        self.annotExtension='json'
        self.index=offset
        self.extension='jpg'
        self.depthExtension='exr'
        self.nberOfImages=nberOfImages
        self.cameraId=cameraId
        self.objectColor={}
        self.listObjects={}
        self.objectIndex={}
        self.objectColorMatch=np.ones([256,256,256],dtype='int')*(-1)
        if mode=="online":
                #make dataset directory
                try:
                    if state=="restart":
                        if not os.path.exists(self.folder):
                            os.makedirs(self.folder)
                        else:
                            for the_file in os.listdir(self.folder):
                                file_path = os.path.join(self.folder, the_file)
                                if os.path.isfile(file_path):
                                    os.unlink(file_path)
                except Exception,e:
                    print('Error: Problem with dataset directory. '+str(e))
                
                try:
                    client.connect()
                    print('Status: \n'+client.request('vget /unrealcv/status'))
                    #set field of view
                    client.request('vset /camera/'+str(self.cameraId)+'/horizontal_fieldofview '+str(70.0))
                    objects=client.request('vget /objects').split(' ')
                    print(objects)
                    print(str(len(objects))+' objects found in the scene.')
                    #map object to color
                    for i in range(len(objects)):
                        #convert '(R=127,G=191,B=127,A=255)' to [127, 191, 127, 255]
                        self.objectIndex[i]=objects[i]
                        e=client.request('vget /object/'+objects[i]+'/color')
                        t=e.split('(')[1].split(')')[0].split(',')
                        t=[int(t[0].split('=')[1]),int(t[1].split('=')[1]),int(t[2].split('=')[1]),int(t[3].split('=')[1])]
                        #t=np.array(t,dtype='uint8')
                        self.objectColor[objects[i]]=t
                        [j,k,l,s]=t
                        self.objectColorMatch[j][k][l]=i
                        for u in range(j-3,j+4):
                                for v in range(k-3,k+4):
                                    for w in range(l-3,l+4):
                                        if u>=0 and v>=0 and w>=0 and u<=255 and v<=255 and w<=255:
                                            if self.objectColorMatch[u][v][w]<0:
                                                self.objectColorMatch[u][v][w]=i 
                    print(self.objectColor)
                    print(self.objectColorMatch)
                                
                except Exception,e:
                    print('Error: Problem with unrealcv .'+str(e))
        else:
            pass        
    #convert from raw to RGB image matrix
    def read_png(self,res):
        img = PIL.Image.open(StringIO.StringIO(res))
        return np.asarray(img)
        
    #convert from raw to  float depth image matrix
    def read_npy(self,res):
        import StringIO
        return np.load(StringIO.StringIO(res))
        
    #get key from dictionnary given value
    def getKey(self,val):
        res=None
        for key in self.objectColor.keys():
            if(self.objectColor[key][0]<=val[0]+3 and self.objectColor[key][0]>=val[0]-3 and
                self.objectColor[key][1]<=val[1]+3 and self.objectColor[key][1]>=val[1]-3 and
                self.objectColor[key][2]<=val[2]+3 and self.objectColor[key][2]>=val[2]-3):
                return key
        print ('key failed',val)
        return None
    #get objects     
    def getCurrentObjects(self,img):
        self.listObjects={}
        sh=img.shape
        n=sh[0]
        m=sh[1]
        for i in range(n):
            for j in range(m):
                color=list(img[i][j])
                key=self.objectIndex[self.objectColorMatch[color[0]][color[1]][color[2]]]
                if key in self.listObjects.keys():
                    self.listObjects[key].append([i,j])
                else:
                    self.listObjects[key]=[[i,j]]
        
    #object name from id
    def getName(self,objectId):
        return objectId
        
    #annotate images
    def annotate(self):
        #build json object
        jsonArray=[]
        #image and question ids. Question id is not used for now
        imageId=str(self.index)
        questionId=""
        try:
            #get camera Position x,y,z
            camPosition=client.request('vget /camera/'+str(self.cameraId)+'/location').split(' ')
            camPosition=[float(camPosition[0]),float(camPosition[1]),float(camPosition[2])]
            #get camera orientation teta,beta,phi
            camOrientation=client.request('vget /camera/'+str(self.cameraId)+'/rotation').split(' ')
            camOrientation=[float(camOrientation[2]),float(camOrientation[0]),float(camOrientation[1])]
        except Exception,e:
            print('Error occured when requesting camera properties. '+str(e))
        #objId is the object Id
        print self.listObjects.keys()
        for objId in self.listObjects.keys():
            #get object tags template
            objTagTemp={"objectShape":"","objectExternalMaterial":"","objectInternalMaterial":"","objectHardness":"",
                        "objectPickability":"","objectGraspability":"","objectStackability":"","objectOpenability":"","objectColor":"","objectName":""}
           
            #get object Location
            objLocation=""
            #get object segmentation color
            objSegColor=self.objectColor[objId]
            #get object segmentation pixels
            objSegPixels=self.listObjects[objId]
            if len(objSegPixels)<=0:
                raise ValueError()
            #get object cuboid
            objCuboid=[]
            #get object local orientation
            objLocalOrientation=[]
            #get object local Position: with camera as reference
            objLocalPosition=[]
            #get object tags,global Position and orientation
            try:
                objTags=client.request('vget /object/'+objId+'/tags')
                #split tags
                objTags=objTags.split(';')
                for elt in objTags:
                    try:
                        elt=elt.split(':')
                        objTagTemp[elt[0]]=elt[1]
                    except Exception,e:
                         print('Error occured when parsing object tags. '+str(e))
                #get object Position x,y,z
                objPosition=client.request('vget /object/'+objId+'/location').split(' ')
                objPosition=[float(objPosition[0]),float(objPosition[1]),float(objPosition[2])]
                #get object orientation teta,beta,phi
                objOrientation=client.request('vget /object/'+objId+'/rotation').split(' ')
                objOrientation=[float(objOrientation[2]),float(objOrientation[0]),float(objOrientation[1])]
                #compute object pose in camera coordinate system
                [objLocalPosition,objLocalOrientation]=self.getCameraObjectPose(np.array(camPosition),self.dToR(np.array(camOrientation)),np.array(objPosition),self.dToR(np.array(objOrientation)))
                objLocalPosition=list(objLocalPosition)
                objLocalOrientation=list(self.rToD(objLocalOrientation))
            except Exception,e:
                print('Error occured when requesting object properties. '+str(e))
                
             
            jsonArray.append(
                              '{"objectId":"'+objId+'",'+
                                '"objectName":"'+str(objTagTemp["objectName"])+'",'+
                                '"objectShape":"'+str(objTagTemp["objectShape"])+'",'+
                                '"objectColor":"'+str(objTagTemp["objectColor"])+'",'+
                                '"objectExternMaterial":"'+str(objTagTemp["objectExternalMaterial"])+'",'+
                                '"objectInternMaterial":"'+str(objTagTemp["objectInternalMaterial"])+'",'+
                                '"objectHardness":"'+str(objTagTemp["objectHardness"])+'",'+
                                '"objectLocation":"'+str(objLocation)+'",'+
                                '"objectPickability":"'+str(objTagTemp["objectPickability"])+'",'+
                                '"objectGraspability":"'+str(objTagTemp["objectGraspability"])+'",'+
                                '"objectStackability":"'+str(objTagTemp["objectStackability"])+'",'+
                                '"objectOpenability":"'+str(objTagTemp["objectOpenability"])+'",'+
                                '"objectGlobalOrientation":'+str(objOrientation)+','+
                                '"objectGlobalPosition":'+str(objPosition)+','+
                                '"objectLocalPosition":'+str(objLocalPosition)+','+
                                '"objectLocalOrientation":'+str(objLocalOrientation)+','+
                                '"objectCuboid":'+str(objCuboid)+','+
                                '"objectSegmentationColor":'+str(objSegColor)+','+
                                '"objectSegmentationPixels":'+str(objSegPixels)+''+
                                '}'
                            )
        
        
        listObj='['
        for i in range(len(jsonArray)):
            listObj=listObj+jsonArray[i]
            if i==len(jsonArray)-1:
                listObj=listObj+']'
            else:
                listObj=listObj+','
        jsonImage='{'+'"imageId":"'+str(imageId)+'",'+'"questionId":"'+str(questionId)+'",'+'"cameraGlobalOrientation":'+str(camOrientation)+','+'"cameraGlobalPosition":'+str(camPosition)+','+'"objectRelationship":[],'+'"objects":'+str(listObj)+''+'}'
        try:
            #convert from plain to json
            jsonImage=json.loads(jsonImage)
            #write json annotation to file
            with open(os.path.join(self.folder,self.annotation+str(self.index)+'.'+self.annotExtension),'w') as outfile:
                json.dump(jsonImage,outfile,indent=5)
            print('Annotation saved successfully.')
            del jsonArray[:]
            return True
        except Exception,e:
            print('Annotation failed. '+str(e))
            return False
    
    def cleanup(self):
        client.disconnect()
        
    #get object pixels
    def getObjectColor(self, jsonFile,objName,imageName):
        try:
            with open(jsonFile,'r') as infile:
                jsonImage=json.load(infile)
            for a in jsonImage['objects']:
                if a['objectName']==objName:
                    e=a
                    break
            if e==None:
                raise ValueError('Unknown object with name: '+objName)
            lign=[]
            col=[]
            obj=e
            img=cv2.imread(imageName)
            histo=np.zeros([256,256,256],dtype='uint')
            for elt in obj['objectSegmentationPixels']:
                histo[img[elt[0]][elt[1]][0]][img[elt[0]][elt[1]][1]][img[elt[0]][elt[1]][2]]+=1
            imax=0
            jmax=0
            kmax=0
            max=0
            for i in range(255):
                for j in range(255):
                    for k in range(255):
                        if  histo[i][j][k]>=max:
                            max= histo[i][j][k]
                            [imax,jmax,kmax]=[i,j,k]
            
            print('Color computed successfully: '+str(objName))
            return [imax,jmax,kmax,max]
        except Exception,e:
            print('Failed to compute object color. '+str(e))
            return []
    
    def saveObject(self, jsonFile,objName,imageName,outImageName):
        try:
            with open(jsonFile,'r') as infile:
                jsonImage=json.load(infile)
            for a in jsonImage['objects']:
                if a['objectName']==objName:
                    e=a
                    break
            if e==None:
                raise ValueError('Unknown object with name: '+objName)
            lign=[]
            col=[]
            obj=e
            
            print  jsonImage['cameraGlobalOrientation']
            print  jsonImage['cameraGlobalPosition']
            print  obj['objectGlobalOrientation']
            print  obj['objectGlobalPosition']
            print  obj['objectLocalOrientation']
            print  obj['objectLocalPosition']
            
            for e in obj['objectSegmentationPixels']:
                lign.append(e[0])
                col.append(e[1])
            
            xmin=min(lign)
            ymin=min(col)
            xmax=max(lign)
            ymax=max(col)
            
            #img=np.zeros([xmax-xmin+1,ymax-ymin+1,3],dtype='uint8')
            img=np.ones([xmax-xmin+1,ymax-ymin+1,3],dtype='uint8')*255
            im=cv2.imread(imageName)
            for e in obj['objectSegmentationPixels']:
                img[e[0]-xmin][e[1]-ymin][0]=im[e[0]][e[1]][0]
                img[e[0]-xmin][e[1]-ymin][1]=im[e[0]][e[1]][1]
                img[e[0]-xmin][e[1]-ymin][2]=im[e[0]][e[1]][2]
            cv2.imwrite(outImageName,img)
            print('Object saved successfully.')
        except Exception,e:
            print('Failed to save object. '+str(e))
     
    #parse image
    #mode direct to display output image as graphic
    #mode indirect to save output image into file 
    def ImageParser(self, jsonFile,imageName,outImageName,size,mode="direct"):
        try:
            #open annotation-json file
            with open(jsonFile,'r') as infile:
                jsonImage=json.load(infile)
            listObjectId=[]
            #load image to be overwritten with objects ids
            img=cv2.imread(imageName)
            #explore all objects
            for obj in jsonImage['objects']:
                #add object id in the list
                print obj["objectId"]#debug
                listObjectId.append(obj["objectId"])
                lign=[]
                col=[]
                #get object pixels coordinates
                for e in obj['objectSegmentationPixels']:
                    lign.append(e[0])
                    col.append(e[1])
                #object's central point
                Xc=int(np.average(lign))
                Yc=int(np.average(col))
                #overwrite image with this object id at point (Xc,Yc)
                cv2.putText(img, str(len(listObjectId)-1), (Yc,Xc),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, size, (0, 255, 0), 1, cv2.LINE_AA)
            if mode=="indirect":
                cv2.imwrite(outImageName,img)
            else:
                cv2.imshow('Annotation',img)
                k=cv2.waitKey(0) & 0xFF
                if k==27:
                    cv2.destroyAllWindows()
            print('Image Parsed successfully.')
            return listObjectId,str(jsonImage["cameraGlobalOrientation"][0])
        except Exception,e:
            print('Failed to parse image. '+str(e))
            return [],''
    
    
    
    
    #compute  RGB-color of an image
    def computeRGBColor(self, imageName):
        #weighted sum of pixels' RGB color
        length=0
        try:
            sum=np.array([256,256,256],dtype='uint')
            img=cv2.imread(imageName)
            histo=np.zeros([256,256,256],dtype='uint')
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                   histo[img[i][j][0]][img[i][j][1]][img[i][j][2]]+=1
            imax=0
            jmax=0
            kmax=0
            max=0
            for i in range(255):
                for j in range(255):
                    for k in range(255):
                        if  histo[i][j][k]>=max:
                            max= histo[i][j][k]
                            [imax,jmax,kmax]=[i,j,k]
            
            print('Color computed successfully: '+str(imageName))
            return [imax,jmax,kmax,max]
        except Exception,e:
            print('Failed to compute color: '+str(imageName)+' .'+str(e))
            return None
            
            
            
        
    #save nberOfImages images
    def scan(self):
        X=np.arange(-267,-208,23)
        Y=np.arange(182,308,21)
        Z=np.arange(140,216,11)
        TETAX=np.arange(-90,91,45)
        TETAY=np.arange(-60,-33,13)
        TETAZ=np.arange(-35,48,17)
        i=self.index
        for x in X:
            for y in Y:
                for z in Z:
                    for tetax in TETAX:
                        for tetay in TETAY:
                            for tetaz in TETAZ:
                                try:
                                    if (x,y,z,tetax,tetay,tetaz)>(-221,224,140,45,-47,-18): 
                                        i=i+1
                                        self.index=i
                                        #set camera position
                                        client.request('vset /camera/'+str(self.cameraId)+'/location '+str(x)+' '+str(y)+' '+str(z))                              
                                        #set camera orientation
                                        client.request('vset /camera/'+str(self.cameraId)+'/rotation '+str(tetay)+' '+str(tetaz)+' '+str(tetax))
                                        #take lit image
                                        client.request('vset /viewmode lit')
                                        img=client.request('vget /camera/'+str(self.cameraId)+'/lit png')
                                        #convert image properly . remove unused A channel  
                                        img=cv2.cvtColor(self.read_png(img),cv2.COLOR_RGBA2BGR)
                                        #save image
                                        if not(cv2.imwrite(os.path.join(self.folder,self.litImage)+str(i)+'.'+self.extension,img)):
                                            raise Exception('Failed to save lit image!!!')
                                        
                                        #take depth Float32
                                        img=client.request('vget /camera/'+str(self.cameraId)+'/depth npy')
                                        #convert image properly .  
                                        img=self.read_npy(img)
                                        #save image
                                        if not(cv2.imwrite(os.path.join(self.folder,self.depthImage)+str(i)+'.'+self.depthExtension,img)):
                                            raise Exception('Failed to save depth float32!!!')    
                                        
                                        #take depth image    
                                        img=255.*(1.-pow(np.exp(-img),0.2))
                                        imgc=np.zeros([img.shape[0],img.shape[1],3])
                                        imgc[:,:,0]=img
                                        imgc[:,:,1]=img
                                        imgc[:,:,2]=img
                                        imgc=np.array(imgc,dtype='uint8')
                                        #take depth image
                                        #client.request('vset /viewmode depth') 
                                        #img=client.request('vget /camera/'+str(self.cameraId)+'/depth png')
                                        #img=self.read_png(img)
                                        #convert image properly . remove unused A channel  
                                        #print img
                                        #img=cv2.cvtColor(img,cv2.COLOR_RGBA2BGR)
                                        #save image
                                        if not(cv2.imwrite(os.path.join(self.folder,self.depthImage)+str(i)+'.'+self.extension,imgc)):
                                            raise Exception('Failed to save depth image!!!')
                                        
                                    
                                        #take normal image 
                                        client.request('vset /viewmode normal')
                                        img=client.request('vget /camera/'+str(self.cameraId)+'/normal png')
                                        #convert image properly . remove unused A channel  
                                        img=cv2.cvtColor(self.read_png(img),cv2.COLOR_RGBA2BGR)
                                        #save image
                                        if not(cv2.imwrite(os.path.join(self.folder,self.normalImage)+str(i)+'.'+self.extension,img)):
                                            raise Exception('Failed to save normal image!!!')
                                            
                                            
                                        #take mask image 
                                        client.request('vset /viewmode object_mask')
                                        img=client.request('vget /camera/'+str(self.cameraId)+'/object_mask png')
                                        img=self.read_png(img)
                                        imgc=img.copy()
                                        #convert image properly . remove unused A channel  
                                        img=cv2.cvtColor(img,cv2.COLOR_RGBA2BGR)
                                        #save image
                                        if not(cv2.imwrite(os.path.join(self.folder,self.maskImage)+str(i)+'.'+self.extension,img)):
                                            raise Exception('Failed to save mask image!!!') 
                                        
                                        #get current objects
                                        self.getCurrentObjects(imgc)
                                        self.T=imgc.copy()
                                        #create annotation
                                        #not
                                        if not self.annotate():
                                            raise Exception('Annotation failed!!!')
                                        print('Image saved with success.')
                                except Exception,e:
                                        if  os.path.exists(os.path.join(self.folder,self.litImage)+str(i)+'.'+self.extension):
                                            os.unlink(os.path.join(self.folder,self.litImage)+str(i)+'.'+self.extension)
                                        
                                            
                                        if os.path.exists(os.path.join(self.folder,self.maskImage)+str(i)+'.'+self.extension):
                                            os.unlink(os.path.join(self.folder,self.maskImage)+str(i)+'.'+self.extension)
                                            
                                            
                                        if os.path.exists(os.path.join(self.folder,self.depthImage)+str(i)+'.'+self.extension):
                                            os.unlink(os.path.join(self.folder,self.depthImage)+str(i)+'.'+self.extension)
                                        
                                        if os.path.exists(os.path.join(self.folder,self.depthImage)+str(i)+'.'+self.depthExtension):
                                            os.unlink(os.path.join(self.folder,self.depthImage)+str(i)+'.'+self.depthExtension)
                                        
                                            
                                        if os.path.exists(os.path.join(self.folder,self.normalImage)+str(i)+'.'+self.extension):
                                            os.unlink(os.path.join(self.folder,self.normalImage)+str(i)+'.'+self.extension)
                                        
                                        print('Image could not be saved not saved: error occured .'+str(e))
                                    #make a pause
                                
                                
        

    # Calculates translation Matrix given translation vector.
    def vectorToTranslationMatrix(self,vector) :
        
        T = np.array([[1,         0,                  0,               vector[0]    ],
                        [0,         1,                  0,               vector[1]    ],
                        [0,         0,                  1,               vector[2]    ],
                        [0,         0,                  0,               1    ]
                        ])
        return T
        
    # Calculates Rotation Matrix given euler angles.
    #mode='inv' for inverse rotation 
    def eulerAnglesToRotationMatrix(self,theta,mode='normal') :
        if mode=='inv':
            theta=-theta
        R_x = np.array([[(1.),         (0.0),                  (0.0)                   ],
                        [(0.0),        ( np.cos(-theta[0])), (-np.sin(-theta[0])) ],
                        [(0.0),         (np.sin(-theta[0])), (np.cos(-theta[0]))  ]
                        ])
            
            
                        
        R_y = np.array([[(np.cos(-theta[1])),    (0.0),     ( np.sin(-theta[1]))  ],
                        [(0.0),                     (1.0),      (0.0)                   ],
                        [(-np.sin(-theta[1])),   (0.0),     (np.cos(-theta[1]))  ]
                        ])
                    
        R_z = np.array([[np.cos(theta[2]),    (-np.sin(theta[2])),    (0.0)],
                        [(np.sin(theta[2])),   ( np.cos(theta[2])),     (0.0)],
                        [(0.0),                     (0.0),   (1.0)]
                        ])
                        
                        
        
        if mode=='inv':
            R=np.dot(R_x,np.dot(R_y,R_z))
        else:
            R=np.dot(R_z,np.dot(R_y,R_x))
        return R
    
    # Checks if a matrix is a valid translation matrix.
    def isTranslationMatrix(self,R) :
        if R.shape[0]!=R.shape[1]:
            return False
        for i in range(R.shape[0]):
            if R[i][i]!=1:
                return False
            for j in range(R.shape[1]-1):
                if j!=i and R[i][j]!=0:
                    return False
        return True
    
        
    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self,R) :
        if R.shape[0]!=R.shape[1]:
            return False
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

     
    # Calculates translation matrix to translation vector
    def translationMatrixToVector(self,R) :
        assert(self.isTranslationMatrix(R))
        vector=[]
        for i in range(R.shape[0]):
            vector.append(R[i][R.shape[1]-1])
        return vector
        
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self,R) :
    
        assert(self.isRotationMatrix(R))
        
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        
        singular = sy < 1e-6
    
        if  not singular :
            x = np.arctan2(R[2,1] , R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else :
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
    
        return np.array([-x, -y, z])    
    
    #degree to radian
    def dToR(self,degree):
        return np.pi*degree/180.0
    #radian to degree
    def rToD(self,radian):
        
        return radian*180.0/np.pi
        
    #normal surface from depth
    #typ='raw' for matrice images
    def normalSurface(self, depth,outputfile,divisor,typ='file'):
        global R
        if typ=='file':
            depth=cv2.imread(depth)
        #we assume depth is of type CV_8UC3
        #depthImg=((np.array(depth[:,:,0],dtype='float32')/divisor)-1.0)
        depthImg=((depth/divisor)-1.0)
        #sobel x
        sobelx=cv2.Sobel(depthImg,cv2.CV_32F,1,0,ksize=15)
        sobely=cv2.Sobel(depthImg,cv2.CV_32F,0,1,ksize=15)
        normalImg=np.zeros([depthImg.shape[0],depthImg.shape[1],3],dtype='float32')
        cols=depthImg.shape[1]
        rows=depthImg.shape[0]
        for y in range(rows):
            for x in range(cols):
                normalImg[y][x][2]=-sobely[y][x]
                normalImg[y][x][1]=-sobelx[y][x]
                normalImg[y][x][0]=1
                normalImg[y][x]=normalImg[y][x]/np.linalg.norm(normalImg[y][x])
                normalImg[y][x]=(0.5*normalImg[y][x]+0.5)*255.0
                
             
       
        R=normalImg 
        normalImg=np.array(normalImg,dtype='uint8')     
        cv2.imshow('normal',normalImg)
        k=cv2.waitKey(0) & 0xFF
        if k==27:
            cv2.destroyAllWindows()
        cv2.imwrite(outputfile,np.array(normalImg,dtype='uint8'))
        
    #return object pose in camera coordinate system   
    def getCameraObjectPose(self, camPos, camOri,objPos,objOri):
        Rin1=self.eulerAnglesToRotationMatrix(camOri,mode='inv')
        R2=self.eulerAnglesToRotationMatrix(objOri,mode='normal')
        T2=objPos
        Ti1=-camPos
        newPos=np.dot(Rin1,T2+Ti1)
        newOri=self.rotationMatrixToEulerAngles(np.dot(Rin1,R2))
        return [newPos[0:3],newOri[0:3]]
    
    def depthConversion(self,PointDepth,f,pixelsToCm):
        H = PointDepth.shape[0]
        W = PointDepth.shape[1]
        i_c = pixelsToCm*(np.float(H) / 2 - 1)
        j_c = pixelsToCm*(np.float(W) / 2 - 1)
        columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
        columns=pixelsToCm*columns
        rows=pixelsToCm*rows
        DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
        DistanceCameraImage=(DistanceFromCenter**2 + f**2)**0.5
        DistanceObjectImage=(PointDepth+DistanceCameraImage)
        PlaneDepth = DistanceObjectImage / (1 + (DistanceFromCenter / f)**2)**(0.5)
        return PlaneDepth-f
    
    #return the binary of an object given its id and the image's annotation file .json
    def binaryMaskObject(self, jsonFile,objName,shape, outImageName):
        try:
            with open(jsonFile,'r') as infile:
                jsonImage=json.load(infile)
            for a in jsonImage['objects']:
                if a['objectName']==objName:
                    e=a
                    break
            if e==None:
                raise ValueError('Unknown object with name: '+objName)
            img=np.zeros(shape,dtype='uint8')
            obj=e
            print  obj['objectSegmentationColor']
            for e in obj['objectSegmentationPixels']:
               img[e[0]][e[1]]=255
                        
            cv2.imwrite(outImageName,img)
            print('Object mask saved successfully.')
        except Exception,e:
            print('Failed to save object mask. '+str(e))
    
    def copyIndexedTo(self,indices,destination):
        try:
            if not os.path.exists(destination):
                os.makedirs(destination)
            for i in range(len(indices)):
                try:
                    #copy annotation
                    copyfile(self.folder+'/'+self.annotation+str(indices[i])+'.'+self.annotExtension,destination+'/'+self.annotation+str(indices[i])+'.'+self.annotExtension)
                    #copy depth
                    copyfile(self.folder+'/'+self.depthImage+str(indices[i])+'.'+self.depthExtension,destination+'/'+self.depthImage+str(indices[i])+'.'+self.depthExtension)
                    #copy litImage
                    copyfile(self.folder+'/'+self.litImage+str(indices[i])+'.'+self.extension,destination+'/'+self.litImage+str(indices[i])+'.'+self.extension)
                    #copy mask image
                    copyfile(self.folder+'/'+self.maskImage+str(indices[i])+'.'+self.extension,destination+'/'+self.maskImage+str(indices[i])+'.'+self.extension)



                except Exception as e:
                    print('A failure occured: '+str(e)+' at index '+str(indices[i]))
            print('Copying successfully terminated!')
        except Exception as e:
               print('A failure occured: '+str(e)+' by creating '+str(destination))      
    
    def getAllAnnotWithRelation(self):
        """return all indices of annotations from dataset in self.folder
           which contain object relations.Since relation annotation is very sparse, this allow
           a special training of the portion of the dataset containing annotations of relations
        """
        listIndex=[]
        try:
            annotations=glob.glob(self.folder+'/*.json')
            total=len(annotations)
            completed=0
            for jsonFile in annotations:
                print(completed,'/',total,' completed')
                try:
                    with open(jsonFile,'r') as infile:
                        jsonImage=json.load(infile)
                    if len(jsonImage['objectRelationship'])>0:
                        listIndex.append(int(jsonFile.split(self.folder)[1].split(self.annotation)[1].split('.')[0]))
                except Exception as e:
                   print('A failure occured: '+str(e)+' in '+jsonFile)
                completed+=1
            return np.array(listIndex,dtype='int32')
        except Exception as e:
            print('A failure occured: '+str(e))
            return np.array(listIndex,dtype='int32')
            
    def cleanRelation(self):
        """make the relations consistent """
        try:
            #load annotation files
            annotation_files=glob.glob(self.folder+'/*.json')
            print annotation_files
            #Initialize dictionary for objId-category mapping
            self.dict_id_cat={}
            
            #Initialize counter for number of relations group by relation types
            self.rel_count={'left':0,'right':0,'front':0,'behind':0,'over':0,'under':0,'valign':0,'in':0,'on':0}
            
            #Initialize set of valid categories
            self.valid_cat==['Cooktop','Tea','Juice','Plate','Mug','Bowl','Tray','Tomato','Ketchup','Salz','Milch','Spoon','Spatula','Milk','Coffee','Cookie','Knife','Cornflakes'
    ,'EggHolder', 'Cube','Mayonnaise','Cereal','Reis']
            
            #process each file
            
            for jsonFile in annotation_files:
                try:
                    #open json file
                    with open(jsonFile,'r') as infile:
                        jsonImage=json.load(infile)
                    infile.close()
                    
                    #Modify json file
                    if jsonImage['objectRelationship']!=[]:
                        #Complete the id-cat dictionary
                        print('Updating id-cat dictionary ...')
                        for obj in jsonImage['objects']:
                            try:
                                catName=obj['objectName'][0].upper()+obj['objectName'][1:]
                                idName=obj['objectId']
                                if idName not in list(self.dict_id_cat.keys()):
                                    self.dict_id_cat[idName]=catName
                            except Exception,e:
                                print('An object processing failed: '+str(e))
                        #clean up relations
                        relations1=[]
                        print('Cleaning up relations ...')
                        #Remove redundance
                        print('1. Eliminating duplications ...')
                        for rel in jsonImage['objectRelationship']:
                            if rel not in relations1:
                                relations1.append(rel.copy())
                        #Remove relations involving invalid object categories
                        print('2. Eliminating relations involving invalid object categories ...')
                        relations2=[]
                        for rel in relations1:
                            if (self.dict_id_cat[rel['object1']] in self.valid_cat and self.dict_id_cat[rel['object2']] in self.valid_cat):
                                relations2.append(rel.copy())
                        del relations1[:]
                        #Remove identities
                        print('3. Eliminating identity relations ...')
                        relations3=[]
                        for rel in relations2:
                            if rel['object1']!=rel['object2']:
                                relations3.append(rel.copy())
                        del relations2[:]
                        #Remove  inconsistencies
                        print('4. Eliminating inconstent relations ...')
                        relations4=[]
                        for rel in relations3:
                            rel1=rel.copy()
                            rel1['object1']=rel['object2']
                            rel1['object2']=rel['object1']
                            if rel1 not in relations3:
                                relations4.append(rel.copy())
                        del relations3[:]
                        #Remove  unknown relations
                        print('5. Eliminating unknown relations ...')
                        relations5=[]
                        for rel in relations4:
                            if rel['relation'] in list(self.rel_count.keys()):
                                relations5.append(rel.copy())
                        del relations4[:]
                        #Augmenting relations
                        print('6. Augmenting relations ...')
                        relations6=[]
                        for rel in relations5:
                            rel1=rel.copy()
                            if rel['relation']=='on':
                                rel1['relation']='under'
                                rel1['object1']=rel['object2']
                                rel1['object2']=rel['object1']
                            else:
                                if rel['relation']=='under':
                                    rel1['relation']='over'
                                    rel1['object1']=rel['object2']
                                    rel1['object2']=rel['object1']
                                else:
                                    if rel['relation']=='front':
                                        rel1['relation']='behind'
                                        rel1['object1']=rel['object2']
                                        rel1['object2']=rel['object1']
                                    else:
                                        if rel['relation']=='left':
                                            rel1['relation']='right'
                                            rel1['object1']=rel['object2']
                                            rel1['object2']=rel['object1']
                            if rel not in relations6:
                                relations6.append(rel.copy())
                            if rel1 not in relations6:
                                relations6.append(rel1.copy())
                        del relations5[:]
                        print('7. Updating instance counter ...')
                        for rel in relations6:
                            self.rel_count[rel['relation']]+=1
                    
                    #Save json file
                    del jsonImage['objectRelationship'][:]
                    jsonImage['objectRelationship']=jsonImage['objectRelationship']+relations6
                    print('Saving changes ...')
                    with open(jsonFile,'w') as infile:
                        json.dump(jsonImage,infile)
                    infile.close()
                    del relations6[:]
                except Exception,e:
                    print('Failed to clean up '+jsonFile+': '+str(e))
            print('Relation clean-up successfully terminated!')
        except Exception,e:
            print('\n\n Failed to clean up relations among objects: '+str(e))
    
    def BigNum(self,x):
        return (x)