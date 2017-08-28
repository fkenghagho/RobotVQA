import numpy as np
import os 
import sys
import time


class Dataset(object):
    def __init__(self,folder,nberOfImages):
        self.folder=folder
        self.nberOfImages=nberOfImages
