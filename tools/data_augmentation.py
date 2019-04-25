"""
#@Author:   Frankln Kenghagho
#@Date:     04.04.2019
#@Project:  RobotVA
"""


import os
import numpy as np


def data_augmentation(image,aug_img_probs=0.6,aug_pix_probs=0.6,pixel_shift=2, image_shift=2):
    """Online Pixelwise Dataset Augmentation
    """
    try:
	    final_image=None
	    final_image=image.copy()
	    rows,cols,depth=image.shape
	    #decide whether to augment the image
	    if((int(os.urandom(5).encode('hex'),16)%100)/100.>=aug_img_probs):
		    for i in range(rows):
			for j in range(cols):
			    for k in range(depth):
				    #choose whether to shift the pixel and how if
				    if((int(os.urandom(5).encode('hex'),16)%100)/100.>=aug_pix_probs):
					#choose whether to shift down or up the pixel and how if
					if(int(os.urandom(5).encode('hex'),16)%2==0):
					   val=np.minimum(255,image[i][j][k]+(int(os.urandom(5).encode('hex'),16)%pixel_shift))
					else:
					   val=np.maximum(0,image[i][j][k]-(int(os.urandom(5).encode('hex'),16)%pixel_shift))
					image[i][j][k]+=val
                    for i in range(rows):
			    #choose whether to shift the rows and how if
			    if((int(os.urandom(5).encode('hex'),16)%100)/100.>=aug_pix_probs):
				#choose whether to shift left or right the row and how if
				if(int(os.urandom(5).encode('hex'),16)%2==0):
				   image[i]=np.roll(image[i],int(os.urandom(5).encode('hex'),16)%image_shift,axis=0)
				else:
				   image[i]=np.roll(image[i],-int(os.urandom(5).encode('hex'),16)%image_shift,axis=0)

                    for j in range(cols):
			    #choose whether to shift the cols and how if
			    if((int(os.urandom(5).encode('hex'),16)%100)/100.>=aug_pix_probs):
				#choose whether to shift left or right the cols and how if
				if(int(os.urandom(5).encode('hex'),16)%2==0):
				   image[:,j,:]=np.roll(image[:,j,:],int(os.urandom(5).encode('hex'),16)%image_shift,axis=0)
				else:
				   image[:,j,:]=np.roll(image[:,j,:],-int(os.urandom(5).encode('hex'),16)%image_shift,axis=0)
	    print('Augmentation of the image/inputs successful!!!')
	    return image
    except Exception,e:
	   print('Failed to augment the image/inputs: '+str(e))
	   if final_image==None:
		return image
	   else:
                return final_image


