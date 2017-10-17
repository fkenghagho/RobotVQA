import numpy as np
import os 
import sys
import time
import StringIO, PIL.Image
from unrealcv import client
import json
import cv2
class Dataset(object):
    def __init__(self,folder,nberOfImages,cameraId):
        self.folder=folder
        self.litImage='litImage'
        self.normalImage='normalImage'
        self.depthImage='depthImage'
        self.maskImage='maskImage'
        self.annotation='annotation'
        self.annotExtension='json'
        self.index=0
        self.extension='jpg'
        self.nberOfImages=nberOfImages
        self.cameraId=cameraId
        self.objectColor={}
        self.listObjects={}
        #make dataset directory
        try:
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
            objects=client.request('vget /objects').split(' ')
            print(objects)
            print(str(len(objects))+' objects found in the scene.')
            #map object to color
            for i in range(len(objects)):
                #convert '(R=127,G=191,B=127,A=255)' to [127, 191, 127, 255]
                e=client.request('vget /object/'+objects[i]+'/color')
                t=e.split('(')[1].split(')')[0].split(',')
                t=[int(t[0].split('=')[1]),int(t[1].split('=')[1]),int(t[2].split('=')[1]),int(t[3].split('=')[1])]
                #t=np.array(t,dtype='uint8')
                self.objectColor[objects[i]]=t
            print(self.objectColor)
        except Exception,e:
            print('Error: Problem with unrealcv .'+str(e))
        
    #convert from raw to RGB image matrice
    def read_png(self,res):
        img = PIL.Image.open(StringIO.StringIO(res))
        return np.asarray(img)
    #get key from dictionnary given value
    def getKey(self,val):
        res=None
        for key in self.objectColor.keys():
            if(self.objectColor[key][0]<=val[0]+3 and self.objectColor[key][0]>=val[0]-3 and
                self.objectColor[key][1]<=val[1]+3 and self.objectColor[key][1]>=val[1]-3 and
                self.objectColor[key][2]<=val[2]+3 and self.objectColor[key][2]>=val[2]-3 and
                self.objectColor[key][3]<=val[3]+3 and self.objectColor[key][3]>=val[3]-3 ):
                return key
        return None
    #get objects     
    def getCurrentObjects(self,img):
        self.listObjects={}
        sh=img.shape
        n=sh[0]
        m=sh[1]
        for i in range(n):
            for j in range(m):
                key=self.getKey(list(img[i][j]))
                if key in self.listObjects.keys():
                    self.listObjects[key].append((i,j))
                else:
                    self.listObjects[key]=[(i,j)]
        
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
            camPosition=client.request('vget /camera/'+self.cameraId+'/location').split(' ')
            camPosition=[float(camPosition[0]),float(camPosition[1]),float(camPosition[2])]
            #get camera orientation teta,beta,phi
            camOrientation=client.request('vget /camera/'+self.cameraId+'/rotation').split(' ')
            camOrientation=[float(camOrientation[0]),float(camOrientation[1]),float(camOrientation[2])]
        except Exception,e:
            print('Error occured when requesting camera properties. '+str(e))
        
        for objId in self.listObjects.keys():
            #get object tags template
            objTagTemp={"objectShape":"","objectExternMaterial":"","objectInternMaterial":"","objectHardness":"",
                        "objectPickability":"","objectGraspability":"","objectStackability":"","objectOpenability":""}
            #get object color
            objColor=""
            #get object Location
            objLocation=""
            #get object segmentation color
            objSegColor=self.objectColor[objId]
            #get object segmentation pixels
            objSegPixels=self.listObjects[objId]
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
                objOrientation=[float(objOrientation[0]),float(objOrientation[1]),float(objOrientation[2])]
            except Exception,e:
                print('Error occured when requesting object properties. '+str(e))
                
             
            jsonArray.append(
                              '{"objectId":"'+objId+'",'+
                                '"objectName":"'+self.getName(objId)+'",'+
                                '"objectShape":"'+objTagTemp["objectShape"]+'",'+
                                '"objectColor":"'+objColor+'",'+
                                '"objectExternMaterial":"'+str(objTagTemp["objectExternMaterial"])+'",'+
                                '"objectInternMaterial":"'+str(objTagTemp["objectInternMaterial"])+'",'+
                                '"objectHardness":"'+str(objTagTemp["objectHardness"])+'",'+
                                '"objectLocation":"'+str(objLocation+)'",'+
                                '"objectPickability":"'+str(objTagTemp["objectPickability"]+)'",'+
                                '"objectGraspability":"'+str(objTagTemp["objectGraspability"])+'",'+
                                '"objectStackability":"'+str(objTagTemp["objectStackability"])+'",'+
                                '"objectOpenability":"'+str(objTagTemp["objectOpenability"])+'",'+
                                '"objectGlobalOrientation":"'+str(objOrientation)+'",'+
                                '"objectGlobalPosition":"'+str(objPosition)+'",'+
                                '"objectLocalPosition":"'+str(objLocalPosition)+'",'+
                                '"objectLocalOrientation":"'+str(objLocalOrientation)+'",'+
                                '"objectCuboid":"'+str(objCuboid)+'",'+
                                '"objectSegmentationColor":"'+str(objSegColor)+'",'+
                                '"objectSegmentationPixels":"'+str(objSegPixels)+'",'+
                                '}'
                            )
        
        
        listObj='['
        for i in range(len(jsonArray)):
            listObj=listObj+jsonArray[i]
            if i==len(jsonArray)-1:
                listObj=listObj+']'
            else:
                listObj=listObj+','
        jsonImage='{'+
                    '"imageId":"'+str(imageId)+'",'+
                    '"questionId":"'+str(questionId)+'",'+
                    '"cameraGlobalOrientation":"'+str(camOrientation)+'",'+
                    '"cameraGlobalPosition":"'+str(camPosition)+'",'+
                    '"objects":"'+str(listObj)+'"'+
                  '}'
        try:
            #convert from plain to json
            jsonImage=json.loads(jsonImage)
            #write json annotation to file
            with open(os.path.join(self.folder,self.annotation+str(self.index)+'.'+self.annotExtension),'w') as outfile:
                json.dump(jsonImage,outfile,indent=5)
            print('Annotation saved successfully.')
        except Exception,e:
            print('Annotation failed. '+str(e))
            
    #save nberOfImages images
    def scan(self):
        for i in range(self.nberOfImages):
                try:
                    img=client.request('vget /camera/'+str(self.cameraId)+'/lit png')
                    img=self.read_png(img)
                    img=cv2.cvtColor(img,cv2.COLOR_RGBA2BGRA)
                    if not(cv2.imwrite(os.path.join(self.folder,self.litImage)+str(i)+'.'+self.extension,img)):
                        print('Failed to save lit image!!!')
                        
                        
                    img=client.request('vget /camera/'+str(self.cameraId)+'/normal png')
                    img=self.read_png(img)
                    img=cv2.cvtColor(img,cv2.COLOR_RGBA2BGRA)
                    if not(cv2.imwrite(os.path.join(self.folder,self.normalImage)+str(i)+'.'+self.extension,img)):
                        print('Failed to save normal image!!!')
                    
                   
                    img=client.request('vget /camera/'+str(self.cameraId)+'/depth png')
                    img=self.read_png(img)
                    img=cv2.cvtColor(img,cv2.COLOR_RGBA2BGRA)
                    if not(cv2.imwrite(os.path.join(self.folder,self.depthImage)+str(i)+'.'+self.extension,img)):
                        print('Failed to save depth image!!!')
                        
                    img=client.request('vget /camera/'+str(self.cameraId)+'/object_mask png')
                    img=self.read_png(img)
                    self.getCurrentObjects(img)
                    #build annotation
                    self.index=i
                    self.annotate()
                    print(self.listObjects)  
                    img=cv2.cvtColor(img,cv2.COLOR_RGBA2BGRA)
                    if not(cv2.imwrite(os.path.join(self.folder,self.maskImage)+str(i)+'.'+self.extension,img)):
                        print('Failed to save maskimage!!!')
                        
                   
                 
                except Exception,e:
                    print('Image could not be saved not saved: error occured .'+str(e))
            
        print('Scan terminated with success.')