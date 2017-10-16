import numpy as np
import os 
import sys
import time
import StringIO, PIL.Image
from unrealcv import client
import cv2
class Dataset(object):
    def __init__(self,folder,nberOfImages,cameraId):
        self.folder=folder
        self.litImage='litImage'
        self.normalImage='normalImage'
        self.depthImage='depthImage'
        self.maskImage='maskImage'
        self.annotation='annotation'
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
                    self.listObjects[key].append(list(img[i][j]))
                else:
                    self.listObjects[key]=[list(img[i][j])]
        
    #annotate images
    def annotate(self):
        pass
        
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
                    print(self.listObjects)  
                    img=cv2.cvtColor(img,cv2.COLOR_RGBA2BGRA)
                    if not(cv2.imwrite(os.path.join(self.folder,self.maskImage)+str(i)+'.'+self.extension,img)):
                        print('Failed to save maskimage!!!')
                        
                   
                 
                except Exception,e:
                    print('Image could not be saved not saved: error occured .'+str(e))
            
        print('Scan terminated with success.')