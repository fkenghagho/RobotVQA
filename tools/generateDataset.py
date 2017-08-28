import numpy as np
import os 
import sys
import time
from unrealcv import client

class Dataset(object):
    def __init__(self,folder,nberOfImages):
        self.folder=folder
        self.nberOfImages=nberOfImages
        self.client.connect()
 


    def scan():
        try:
            p=self.client.request('vget /camera/0/lit')
            a=p.split('/').pop()
            p=self.client.request('vget /camera/0/object_mask '+a)
            print p
        except Exception,e:
            print 'Image not saved: error occured, '+str(e)
    
