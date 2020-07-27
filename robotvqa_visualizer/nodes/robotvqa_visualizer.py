#!/usr/bin/env python
import rospy
import sys
import os
from PIL import Image
from std_msgs.msg import( String )


import cv2
import numpy as np
import cv_bridge
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import roslib

 

class RobotVQAVisualizer(object):

	def __init__(self):
		try:

			#ros node
			rospy.init_node('robotvqa_visualizer')
			rospy.on_shutdown(self.cleanup)
                       
			#subscribers
                        topic=rospy.get_param('topic','/camera/color/image_raw')
			self.sub = rospy.Subscriber(topic,Image,self.imageBuffering)
 
			#attributes
			self.cvImageBuffer=[]
			self.INDEX=-1
			self.INSTANCEINDEX=0
			if rospy.get_param('VIDEOMODE','local')=='local':
				self.cvMode='rgb8'
			else:
				self.cvMode='bgr8'
                     	self.total=0
			self.success=0
			self.currentImage=[]#current image
			self.height=1000
			self.width=1000
			self.bridge = CvBridge()
			rospy.logwarn('visualizer initialized!!!'+str(e))
		except Exception as e:
			rospy.logwarn('Failed to properly initialize the visualizer '+str(e))
	
	
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
	def imageBuffering(self,image):
		
		try:
			self.currentImage = self.bridge.imgmsg_to_cv2(image, self.cvMode)
                        b=self.currentImage[:,:,0].copy()
			self.currentImage[:,:,0]=self.currentImage[:,:,2].copy()
			self.currentImage[:,:,2]=b.copy()
			self.currentImage = self.resize([self.currentImage],self.height,self.width)[0]
			
			rospy.loginfo('Buffering of current image successful')
			self.showImages()
		except Exception as e:
			rospy.logwarn(' Failed to buffer image '+str(e))

		
        ###################################################################
	def cleanup(self):
		rospy.logwarn('Shutting down RobotVQA Visualizer Node ...')	

	
	###################################################################
	def showImages(self):
		if len(self.currentImage)>0:
			cv2.imshow("Streaming-World",self.currentImage)
			k = cv2.waitKey(1) & 0xFF
      

if __name__=="__main__":
    
    try:
        vis=RobotVQAVisualizer()
	rospy.spin()
       
    except Exception as e:
	rospy.logwarn('Shutting down RobotVQA Visualizer node ...'+str(e))
