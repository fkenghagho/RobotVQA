"""
#@Author:   Frankln Kenghagho
#@Date:     04.04.2019
#@Project:  RobotVA
#Extends the work below Mask R-CNN
"""



"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import random
import itertools
import colorsys
import skimage.io
import numpy as np
from skimage.measure import find_contours
#import IPython.display
import utils
from generateDataset import Dataset
from  DatasetClasses import DatasetClasses
from  robotVQAConfig import  RobotVQAConfig
#Scene Graph Server
from rs_robotvqa_msgs.srv import *
from rs_robotvqa_msgs.msg import *
FIGURE=None

############################################################
#  Visualization
############################################################
def get_ax(rows=1, cols=1, size=6,size1=6):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """

    fig, ax = plt.subplots(rows, cols,gridspec_kw = {'width_ratios':[3, 1]}, figsize=(size1*cols, size*rows))
    FIGURE=fig
    return ax
    
    




def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i*1.0 / N, 1.0, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
   
    return colors


def apply_mask(image, mask, color, alpha=0.2):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,poses,masked_image,back_img,axs,
                      scores=None, title="",title1='',
                      figsize=(16, 16),result_path=None, name="single", results=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    poses: [num_instance, (tetax,tetay,tetaz,x,y,z)] in radians and cm.
    masks: [height, width, num_instances]
    class_ids: [NUM_FEATURES,num_instances] ids/feature
    class_names: [NUM_FEATURES], list of class names of the dataset/feature
    scores: [NUM_FEATURES,num_instances],(optional) confidence scores for each box/feature
    figsize: (optional) the size of the image.
    """
    
    
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
	if result_path:
	    skimage.io.imsave(result_path,image.astype(np.uint8))
        return [masked_image,back_img]
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids[0].shape[0]==class_ids[1].shape[0]==class_ids[2].shape[0]==\
        class_ids[3].shape[0]==class_ids[4].shape[0]==class_ids[5].shape[0]==scores[0].shape[0]==scores[1].shape[0]==scores[2].shape[0]==\
        scores[3].shape[0]==scores[4].shape[0]==scores[5].shape[0]==poses.shape[0]

    if not np.any(axs):
        _, axs = plt.subplots(1,2, figsize=figsize)
    ax=axs[0]
    ax1=axs[1]

    dataset=Dataset()

    # Generate random colors
    colors = random_colors(N)
    titles=[title,title1]
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    if masked_image==[] or back_img==[]:
	    for i in range(axs.size):
		axs[i].set_ylim(height + 10, -10)
		axs[i].set_xlim(-10, width + 10)
		axs[i].axis('off')
		axs[i].set_title(titles[i])
    if masked_image==[]:
    	masked_image = image.astype(np.uint32).copy()
    if back_img==[]:
    	back_img=image.astype(np.uint32).copy()*0+255
    x3=10
    y3=10
    #class_ids[len(class_ids)-1]=class_ids[len(class_ids)-1]+1
    #best_relations=np.argmax(scores[len(class_ids)-1],axis=1)
    #best_relations=utils.relation_graph(boxes)
    object_filter=[]
    config=RobotVQAConfig()
    for i in range(N):
         for j in range(len(class_ids)-1):
             if (scores[j][i]<config.OBJECT_PROP_THRESHOLD) or (class_names[0][class_ids[0][i]] in DatasetClasses.POOR_FEATURED_OBJECTS):
			object_filter.append(i)
                        break
    for i in range(N):

        if i in object_filter:
		continue
             
        # Bounding box##################################################################
        color = colors[i]
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)
	
	#Draw poses
	AXIS_LENGTH=20	
	
	orientation=np.array([poses[i][0],poses[i][1],poses[i][2]],dtype='float32')
	position=np.array([poses[i][3],poses[i][4],poses[i][5]],dtype='float32')
	#position=np.array([1.7107088e+02,9.0871674e+01, 6.2195825e+02],dtype='float32')
	#orientation=np.array([5.4416003e+00, 3.3464733e-01, 4.9978323e+00],dtype='float32')

	#Get rotation matrix for the corresponding Euler's angles orientation
	rotation=dataset.eulerAnglesToRotationMatrix(orientation)

	#Rotated axes
	VZ=np.dot(rotation,np.array([0.,0.,1.]))[1:]
	VY=np.dot(rotation,np.array([0.,1.,0.]))[1:]
	VX=np.dot(rotation,np.array([1.,0.,0.]))[1:]
	if  np.linalg.norm(VZ)!=0:
		VZ/=np.linalg.norm(VZ)
	if  np.linalg.norm(VY)!=0:
		VY/=np.linalg.norm(VY)
	if  np.linalg.norm(VX)!=0:
			VX/=np.linalg.norm(VX)

	#exact object position
	X=x1+(x2 - x1)/2
	Y=y1+(y2 - y1)/2
	
	#Draw axis Y in green
	Xc=np.maximum(np.minimum(X+VY[0]*(AXIS_LENGTH),image.shape[1]),0)
	Yc=np.maximum(np.minimum(Y+VY[1]*(-AXIS_LENGTH),image.shape[0]),0)
	QY=patches.Arrow(X, Y, Xc-X, Yc-Y, width=18.0, color='green')
	ax.add_patch(QY)

	#Draw axis X in red
	Xc=np.maximum(np.minimum(X+VX[0]*(AXIS_LENGTH),image.shape[1]),0)
	Yc=np.maximum(np.minimum(Y+VX[1]*(-AXIS_LENGTH),image.shape[0]),0)
	QX=patches.Arrow(X, Y, Xc-X, Yc-Y, width=18.0, color='red')
	ax.add_patch(QX)

	#Draw axis Z in blue
	Xc=np.maximum(np.minimum(X+VZ[0]*(AXIS_LENGTH),image.shape[1]),0)
	Yc=np.maximum(np.minimum(Y+VZ[1]*(-AXIS_LENGTH),image.shape[0]),0)
	QZ=patches.Arrow(X, Y, Xc-X, Yc-Y, width=18.0, color='blue')
	ax.add_patch(QZ)
	
	#Identify poses
	Center=patches.Circle((X,Y), radius=5, fill=True,edgecolor=color, facecolor=color)
	ax.add_patch(Center)

        # Label#######################################################################
        caption=str(i)+'. '
        if scores!=None or True:
            for j in range(len(class_ids)-1):
                caption=caption+class_names[j][class_ids[j][i]]+' '+str(scores[j][i])+'\n'
        else:
            for j in range(len(class_ids)-1):
                caption=caption+class_names[j][class_ids[j][i]]+'\n'
        caption=caption+'Orientation: ('+str(poses[i][0])+','+str(poses[i][1])+','+str(poses[i][2])+')\n'
        caption=caption+'Position: ('+str(X)+','+str(Y)+','+str(poses[i][5])+')'
        x = random.randint(x1, (x1 + x2) // 2)
	
        #adding labels
        ax.text(x1, y1 + 8, caption,
                color=colors[i], size=9, backgroundcolor="none")
                
        #object relationship#############################################################################
	
	#left
        kleft=0
        sleft=-1

	#left
        kfront=0
        sfront=-1
        for k in range(N):
            #k=best_relations[i]
            if k in object_filter:
		continue
            if(class_ids[len(class_ids)-1][i][k]!=0 and i!=k and (class_ids[len(class_ids)-1][k][i]==0 or (class_ids[len(class_ids)-1][k][i]!=0  and scores[len(class_ids)-1][i][k]>scores[len(class_ids)-1][k][i]))):
		rel_name=class_names[5][class_ids[len(class_ids)-1][i][k]]
                if rel_name not in ['Front', 'Left']:
		        caption=str(i)+'.'+class_names[0][class_ids[0][i]]+' is '+class_names[5][class_ids[len(class_ids)-1][i][k]]+' '+str(k)+\
		        '.'+class_names[0][class_ids[0][k]]+': '+str((scores[len(class_ids)-1][i][k]))+'.'
		        #print('RELATION:'+caption)
		        ax1.text(x3, y3, caption,color='black', size=10, backgroundcolor="none")
		        y3+=20
		if class_names[5][class_ids[len(class_ids)-1][i][k]]=='Front':
			if scores[len(class_ids)-1][i][k]>sfront:
				sfront=scores[len(class_ids)-1][i][k]
				kfront=k
		if class_names[5][class_ids[len(class_ids)-1][i][k]]=='Left':
			if scores[len(class_ids)-1][i][k]>sleft:
				sleft=scores[len(class_ids)-1][i][k]
				kleft=k
	if sleft!=-1:
		k=kleft
 		caption=str(i)+'.'+class_names[0][class_ids[0][i]]+' is '+class_names[5][class_ids[len(class_ids)-1][i][k]]+' '+str(k)+\
		        '.'+class_names[0][class_ids[0][k]]+': '+str((scores[len(class_ids)-1][i][k]))+'.'
		#print('RELATION:'+caption)
	        ax1.text(x3, y3, caption,color='black', size=10, backgroundcolor="none")
	        y3+=20	

	if sfront!=-1:
		k=kfront
 		caption=str(i)+'.'+class_names[0][class_ids[0][i]]+' is '+class_names[5][class_ids[len(class_ids)-1][i][k]]+' '+str(k)+\
		        '.'+class_names[0][class_ids[0][k]]+': '+str((scores[len(class_ids)-1][i][k]))+'.'
		#print('RELATION:'+caption)
	        ax1.text(x3, y3, caption,color='black', size=10, backgroundcolor="none")
	        y3+=30	

		
     
        # Mask
        mask = masks[:, :, i]
	#print('******************** ERROR TRACKER *****************')
	#print(mask.shape,mask.sum())
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8),aspect='auto')
    ax1.imshow(back_img.astype(np.uint8),aspect='auto')
    
    if result_path:
	    plt.savefig(result_path)
    print('ok')
    return [masked_image,back_img]
   
##############################################################################################################################################
def display_instances_v2(image, listOfObjects, spatialRelations,axs,score=True,title="",title1='',
                         figsize=(16, 16),result_path=None):
    """
    image: RGB Image(WxH)
    listOfObjects: [num_instance, object dictionary], list of object descriptions + belief probabilities 
    spatialRelations: [num_instance, num_instance, spatial relation], squarre grid of spatial relations + belief probabilities
    score:       whether or not we should show the belief probabilities
    axs: the graphics objects
    result_path: where to save the graphical result
    figsize: (optional) the size of the image.
    """
    FINAL_RESULT=GetSceneGraphResponse()
    # Number of instances
    N =len(listOfObjects)
    if not N:
        print("\n*** No instances to display *** \n")
	if result_path:
	    skimage.io.imsave(result_path,image.astype(np.uint8))
        return FINAL_RESULT
   
    # if the graphics objects are null
    if not np.any(axs):
        _, axs = plt.subplots(1,2, figsize=figsize)
    ax=axs[0]
    ax1=axs[1]
     
    # dataset
    dataset=Dataset()

    # Generate random colors
    colors = random_colors(N)
    titles=[title,title1]
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    for i in range(axs.size):
	axs[i].set_ylim(height + 10, -10)
	axs[i].set_xlim(-10, width + 10)
	axs[i].axis('off')
	axs[i].set_title(titles[i])
    	masked_image = image.astype(np.uint32).copy()
    	back_img=image.astype(np.uint32).copy()*0+255
    x3=10
    y3=10
    
    object_filter=[]
    config=RobotVQAConfig()
    for i in range(N):
         for value in listOfObjects[i].values():
             if (value[1]<config.OBJECT_PROP_THRESHOLD) or (value[0] in DatasetClasses.POOR_FEATURED_OBJECTS):
			object_filter.append(i)
                        break
         if (not (listOfObjects[i]["cat"][0] in CB_OBJECT_FILTER.keys()) or not(listOfObjects[i]["col"][0] in CB_OBJECT_FILTER[listOfObjects[i]["cat"][0]]["col"])):
            object_filter.append(i)
                      
    for i in range(N):

        if i in object_filter:
		continue
        objdescr=ObjectDescription()
        # Bounding box##################################################################
        color = colors[i]
        if not np.any(listOfObjects[i]["boxes"][0]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = np.array(listOfObjects[i]["boxes"][0],dtype="int")
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)
	
	#Draw poses
	AXIS_LENGTH=20	
	
	orientation=np.array(listOfObjects[i]["poses"][0][:3],dtype='float32')
	position=np.array(listOfObjects[i]["poses"][0][3:],dtype='float32')
	
	#Get rotation matrix for the corresponding Euler's angles orientation
	rotation=dataset.eulerAnglesToRotationMatrix(orientation)

	#Rotated axes
	VZ=np.dot(rotation,np.array([0.,0.,1.]))[1:]
	VY=np.dot(rotation,np.array([0.,1.,0.]))[1:]
	VX=np.dot(rotation,np.array([1.,0.,0.]))[1:]
	if  np.linalg.norm(VZ)!=0:
		VZ/=np.linalg.norm(VZ)
	if  np.linalg.norm(VY)!=0:
		VY/=np.linalg.norm(VY)
	if  np.linalg.norm(VX)!=0:
			VX/=np.linalg.norm(VX)

	#exact object position
	X=x1+(x2 - x1)/2
	Y=y1+(y2 - y1)/2
	
	#Draw axis Y in green
	Xc=np.maximum(np.minimum(X+VY[0]*(AXIS_LENGTH),image.shape[1]),0)
	Yc=np.maximum(np.minimum(Y+VY[1]*(-AXIS_LENGTH),image.shape[0]),0)
	QY=patches.Arrow(X, Y, Xc-X, Yc-Y, width=18.0, color='green')
	ax.add_patch(QY)

	#Draw axis X in red
	Xc=np.maximum(np.minimum(X+VX[0]*(AXIS_LENGTH),image.shape[1]),0)
	Yc=np.maximum(np.minimum(Y+VX[1]*(-AXIS_LENGTH),image.shape[0]),0)
	QX=patches.Arrow(X, Y, Xc-X, Yc-Y, width=18.0, color='red')
	ax.add_patch(QX)

	#Draw axis Z in blue
	Xc=np.maximum(np.minimum(X+VZ[0]*(AXIS_LENGTH),image.shape[1]),0)
	Yc=np.maximum(np.minimum(Y+VZ[1]*(-AXIS_LENGTH),image.shape[0]),0)
	QZ=patches.Arrow(X, Y, Xc-X, Yc-Y, width=18.0, color='blue')
	ax.add_patch(QZ)
	
	#Identify poses
	Center=patches.Circle((X,Y), radius=5, fill=True,edgecolor=color, facecolor=color)
	ax.add_patch(Center)

        # Label#######################################################################
        caption=str(i)+'. '
          
        if score:
                caption=caption+listOfObjects[i]["cat"][0]+' '+str(listOfObjects[i]["cat"][1])+'\n'
                caption=caption+listOfObjects[i]["col"][0]+' '+str(listOfObjects[i]["col"][1])+'\n'
                caption=caption+listOfObjects[i]["sha"][0]+' '+str(listOfObjects[i]["sha"][1])+'\n'
                caption=caption+listOfObjects[i]["mat"][0]+' '+str(listOfObjects[i]["mat"][1])+'\n'
                caption=caption+listOfObjects[i]["opn"][0]+' '+str(listOfObjects[i]["opn"][1])+'\n'
        else:
                caption=caption+listOfObjects[i]["cat"][0]+'\n'
                caption=caption+listOfObjects[i]["col"][0]+'\n'
                caption=caption+listOfObjects[i]["sha"][0]+'\n'
                caption=caption+listOfObjects[i]["mat"][0]+'\n'
                caption=caption+listOfObjects[i]["opn"][0]+'\n'
        caption=caption+'Orientation: ('+str(listOfObjects[i]["poses"][0][0])+','+str(listOfObjects[i]["poses"][0][1])+','+str(listOfObjects[i]["poses"][0][2])+')\n'
        caption=caption+'Position: ('+str(X)+','+str(Y)+','+str(listOfObjects[i]["poses"][0][5])+')'
        x = random.randint(x1, (x1 + x2) // 2)
	
        #adding labels
        ax.text(x1, y1 + 8, caption,
                color=colors[i], size=9, backgroundcolor="none")


        # writing object into server response
        objprop= ObjectProperty()
        objprop.name="Category"
        objprop.num_value.append(0.0)
        objprop.text_value.append(str(i)+'.'+listOfObjects[i]["cat"][0])
        objprop.confidence.append(listOfObjects[i]["cat"][1])
        objdescr.properties.append(objprop)

        objprop= ObjectProperty()
        objprop.name="Color"
        objprop.num_value.append(0.0)
        objprop.text_value.append(listOfObjects[i]["col"][0])
        objprop.confidence.append(listOfObjects[i]["col"][1])
        objdescr.properties.append(objprop)

        objprop= ObjectProperty()
        objprop.name="Shape"
        objprop.num_value.append(0.0)
        objprop.text_value.append(listOfObjects[i]["sha"][0])
        objprop.confidence.append(listOfObjects[i]["sha"][1])
        objdescr.properties.append(objprop)
        
        objprop= ObjectProperty()
        objprop.name="Material"
        objprop.num_value.append(0.0)
        objprop.text_value.append(listOfObjects[i]["mat"][0])
        objprop.confidence.append(listOfObjects[i]["mat"][1])
        objdescr.properties.append(objprop)

        objprop= ObjectProperty()
        objprop.name="Openability"
        objprop.num_value.append(0.0)
        objprop.text_value.append(listOfObjects[i]["opn"][0])
        objprop.confidence.append(listOfObjects[i]["opn"][1])
        objdescr.properties.append(objprop)

        objprop= ObjectProperty()
        objprop.name="Position"
        objprop.num_value=[X,Y,listOfObjects[i]["poses"][0][5]]
        objprop.text_value=[str(X),str(Y),str(listOfObjects[i]["poses"][0][5])]
        objprop.confidence=[0.017,0.017,0.017]
        objdescr.properties.append(objprop)

        objprop= ObjectProperty()
        objprop.name="Orientation"
        objprop.num_value=[listOfObjects[i]["poses"][0][0],listOfObjects[i]["poses"][0][1],listOfObjects[i]["poses"][0][2]]
        objprop.text_value=[str(listOfObjects[i]["poses"][0][0]),str(listOfObjects[i]["poses"][0][1]),str(listOfObjects[i]["poses"][0][2])]
        objprop.confidence=[0.87,0.87,0.87]
        objdescr.properties.append(objprop)


        objprop= ObjectProperty()
        objprop.name="BBox"
        objprop.num_value=[x1, y1, x2 - x1, y2 - y1]
        objprop.text_value=[str(x1), str(y1),str(x2 - x1),str(y2 - y1)]
        objprop.confidence=[0.983,0.983,0.983,0.983]
        objdescr.properties.append(objprop)
        
        print("\n************************  2   ************************\n")        
        #object relationship#############################################################################
	
	#left
        kleft=0
        sleft=-1

	#left0
        kfront=0
        sfront=-1
        for k in range(N):
            #k=best_relations[i]
            if k in object_filter or (spatialRelations[1][i][k]<=config.OBJECT_PROP_THRESHOLD):
		continue
            if(spatialRelations[0][i][k]!="BG" and i!=k and (spatialRelations[0][k][i]=="BG" or (spatialRelations[0][k][i]!="BG" and spatialRelations[1][i][k]>spatialRelations[1][k][i]))):
		rel_name=spatialRelations[0][i][k]
                if rel_name not in ['Front', 'Left']:
		        caption=str(i)+'.'+listOfObjects[i]["cat"][0]+' is '+spatialRelations[0][i][k]+' '+str(k)+\
		        '.'+listOfObjects[k]["cat"][0]+': '+str(spatialRelations[1][i][k])+'.'
		        #print('RELATION:'+caption)
		        ax1.text(x3, y3, caption,color='black', size=10, backgroundcolor="none")
		        y3+=20
                        #adding relations to server's response
                        relation=ObjectRelation()
                        relation.object_id1=str(i)+'.'+listOfObjects[i]["cat"][0]
                        relation.object_id2=str(k)+'.'+listOfObjects[k]["cat"][0]
                        relation.num_value.append(0.0)
			relation.text_value.append(spatialRelations[0][i][k])
                        relation.confidence.append(spatialRelations[1][i][k])
                        FINAL_RESULT.answer.relations.append(relation)
		if spatialRelations[0][i][k]=='Front':
			if spatialRelations[1][i][k]>sfront:
				sfront=spatialRelations[1][i][k]
				kfront=k
		if spatialRelations[0][i][k]=='Left':
			if spatialRelations[1][i][k]>sleft:
				sleft=spatialRelations[1][i][k]
				kleft=k
            else:
                 spatialRelations[0][i][k]="BG"
                 spatialRelations[1][i][k]=1.0  
	if sleft!=-1:
		k=kleft
 		caption=str(i)+'.'+listOfObjects[i]["cat"][0]+' is '+spatialRelations[0][i][k]+' '+str(k)+\
		        '.'+listOfObjects[k]["cat"][0]+': '+str(spatialRelations[1][i][k])+'.'
		#print('RELATION:'+caption)
	        ax1.text(x3, y3, caption,color='black', size=10, backgroundcolor="none")
	        y3+=20	
	        #adding relations to server's response
                relation=ObjectRelation()
                relation.object_id1=str(i)+'.'+listOfObjects[i]["cat"][0]
                relation.object_id2=str(k)+'.'+listOfObjects[k]["cat"][0]
                relation.num_value.append(0.0)
		relation.text_value.append(spatialRelations[0][i][k])
                relation.confidence.append(spatialRelations[1][i][k])
                FINAL_RESULT.answer.relations.append(relation)              
	if sfront!=-1:
		k=kfront
 		caption=str(i)+'.'+listOfObjects[i]["cat"][0]+' is '+spatialRelations[0][i][k]+' '+str(k)+\
		        '.'+listOfObjects[k]["cat"][0]+': '+str(spatialRelations[1][i][k])+'.'
		#print('RELATION:'+caption)
	        ax1.text(x3, y3, caption,color='black', size=10, backgroundcolor="none")
	        y3+=30
                #adding relations to server's response
                relation=ObjectRelation()
                relation.object_id1=str(i)+'.'+listOfObjects[i]["cat"][0]
                relation.object_id2=str(k)+'.'+listOfObjects[k]["cat"][0]
                relation.num_value.append(0.0)
		relation.text_value.append(spatialRelations[0][i][k])
                relation.confidence.append(spatialRelations[1][i][k])
                FINAL_RESULT.answer.relations.append(relation) 	

		
        print("\n************************  3   ************************\n")
        # Mask
        mask = listOfObjects[i]["masks"][0]
	#print('******************** ERROR TRACKER *****************')
	#print(mask.shape,mask.sum())
        masked_image = apply_mask(masked_image, mask, color)
        print(masked_image.shape,mask.shape)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
      
        objprop= ObjectProperty()
        objprop.name="Mask"
        objprop.num_value=mask.reshape([mask.shape[0]*mask.shape[1]])
        objprop.text_value=[str(objprop.num_value[i]) for i in range(mask.shape[0]*mask.shape[1])]
        objprop.confidence=[0.968 for i in range(mask.shape[0]*mask.shape[1])]
        objdescr.properties.append(objprop)
        #adding objects to server's response
        FINAL_RESULT.answer.objects.append(objdescr)
    ax.imshow(masked_image.astype(np.uint8),aspect='auto')
    ax1.imshow(back_img.astype(np.uint8),aspect='auto')
    
    if result_path:
	    plt.savefig(result_path)
    print('ok')
    return FINAL_RESULT
##############################################################################################################################################   





##############################################################################################################################################
def display_instances_v3(image, listOfObjects, spatialRelations,axs,score=True,title="",title1='',
                         figsize=(16, 16),result_path=None):
    """
    image: RGB Image(WxH)
    listOfObjects: [num_instance, object dictionary], list of object descriptions + belief probabilities 
    spatialRelations: [num_instance, num_instance, spatial relation], squarre grid of spatial relations + belief probabilities
    score:       whether or not we should show the belief probabilities
    axs: the graphics objects
    result_path: where to save the graphical result
    figsize: (optional) the size of the image.
    """
    print("\n************************  1.Initialization   ************************\n")
    FINAL_RESULT=GetSceneGraphResponse()
    # Number of instances
    N =len(listOfObjects)
    if not N:
        print("\n*** No instances to display *** \n")
	if result_path:
	    skimage.io.imsave(result_path,image.astype(np.uint8))
        return FINAL_RESULT
     
    # dataset
    dataset=Dataset()
    
    object_filter=[]
    config=RobotVQAConfig()
    for i in range(N):
        for value in listOfObjects[i].values():
             if (value[1]<config.OBJECT_PROP_THRESHOLD) or (value[0] in DatasetClasses.POOR_FEATURED_OBJECTS):
			object_filter.append(i)
                        break
        if (not (listOfObjects[i]["cat"][0] in DatasetClasses.CB_OBJECT_FILTER.keys()) or not(listOfObjects[i]["col"][0] in DatasetClasses.CB_OBJECT_FILTER[listOfObjects[i]["cat"][0]]["col"])):
            object_filter.append(i)

    for i in range(N):

        if i in object_filter:
		continue
        objdescr=ObjectDescription()
        # Bounding box##################################################################
        if not np.any(listOfObjects[i]["boxes"][0]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = np.array(listOfObjects[i]["boxes"][0],dtype="int")
        
	#Draw poses
	orientation=np.array(listOfObjects[i]["poses"][0][:3],dtype='float32')
	position=np.array(listOfObjects[i]["poses"][0][3:],dtype='float32')
	
	#exact object position
	X=x1+(x2 - x1)/2
	Y=y1+(y2 - y1)/2

        # writing object into server response
        print("\n************************  1.Properties   ************************\n") 
        objprop= ObjectProperty()
        objprop.name="Category"
        objprop.num_value.append(0.0)
        objprop.text_value.append(str(i)+'.'+listOfObjects[i]["cat"][0])
        objprop.confidence.append(listOfObjects[i]["cat"][1])
        objdescr.properties.append(objprop)

        objprop= ObjectProperty()
        objprop.name="Color"
        objprop.num_value.append(0.0)
        objprop.text_value.append(listOfObjects[i]["col"][0])
        objprop.confidence.append(listOfObjects[i]["col"][1])
        objdescr.properties.append(objprop)

        objprop= ObjectProperty()
        objprop.name="Shape"
        objprop.num_value.append(0.0)
        objprop.text_value.append(listOfObjects[i]["sha"][0])
        objprop.confidence.append(listOfObjects[i]["sha"][1])
        objdescr.properties.append(objprop)
        
        objprop= ObjectProperty()
        objprop.name="Material"
        objprop.num_value.append(0.0)
        objprop.text_value.append(listOfObjects[i]["mat"][0])
        objprop.confidence.append(listOfObjects[i]["mat"][1])
        objdescr.properties.append(objprop)

        objprop= ObjectProperty()
        objprop.name="Openability"
        objprop.num_value.append(0.0)
        objprop.text_value.append(listOfObjects[i]["opn"][0])
        objprop.confidence.append(listOfObjects[i]["opn"][1])
        objdescr.properties.append(objprop)

        objprop= ObjectProperty()
        objprop.name="Position"
        objprop.num_value=[X,Y,listOfObjects[i]["poses"][0][5]]
        objprop.text_value=[]
        objprop.confidence=[0.017]
        objdescr.properties.append(objprop)

        objprop= ObjectProperty()
        objprop.name="Orientation"
        objprop.num_value=[listOfObjects[i]["poses"][0][0],listOfObjects[i]["poses"][0][1],listOfObjects[i]["poses"][0][2]]
        objprop.text_value=[]
        objprop.confidence=[0.87]
        objdescr.properties.append(objprop)


        objprop= ObjectProperty()
        objprop.name="BBox"
        objprop.num_value=[x1, y1, x2 - x1, y2 - y1]
        objprop.text_value=[]
        objprop.confidence=[0.983]
        objdescr.properties.append(objprop)
        
        print("\n************************  2.Relations   ************************\n")        
        #object relationship#############################################################################
	
	#left
        kleft=0
        sleft=-1

	#left0
        kfront=0
        sfront=-1
        for k in range(N):
            #k=best_relations[i]
            if k in object_filter or (spatialRelations[1][i][k]<=config.OBJECT_PROP_THRESHOLD):
		continue
            if(spatialRelations[0][i][k]!="BG" and i!=k and (spatialRelations[0][k][i]=="BG" or (spatialRelations[0][k][i]!="BG" and spatialRelations[1][i][k]>spatialRelations[1][k][i]))):
		rel_name=spatialRelations[0][i][k]
                if rel_name not in ['Front', 'Left']:
                        #adding relations to server's response
                        relation=ObjectRelation()
                        relation.object_id1=str(i)+'.'+listOfObjects[i]["cat"][0]
                        relation.object_id2=str(k)+'.'+listOfObjects[k]["cat"][0]
                        relation.num_value.append(0.0)
			relation.text_value.append(spatialRelations[0][i][k])
                        relation.confidence.append(spatialRelations[1][i][k])
                        FINAL_RESULT.answer.relations.append(relation)
		if spatialRelations[0][i][k]=='Front':
			if spatialRelations[1][i][k]>sfront:
				sfront=spatialRelations[1][i][k]
				kfront=k
		if spatialRelations[0][i][k]=='Left':
			if spatialRelations[1][i][k]>sleft:
				sleft=spatialRelations[1][i][k]
				kleft=k
            else:
                 spatialRelations[0][i][k]="BG"
                 spatialRelations[1][i][k]=1.0  
	if sleft!=-1:
		k=kleft
	        #adding relations to server's response
                relation=ObjectRelation()
                relation.object_id1=str(i)+'.'+listOfObjects[i]["cat"][0]
                relation.object_id2=str(k)+'.'+listOfObjects[k]["cat"][0]
                relation.num_value.append(0.0)
		relation.text_value.append(spatialRelations[0][i][k])
                relation.confidence.append(spatialRelations[1][i][k])
                FINAL_RESULT.answer.relations.append(relation)              
	if sfront!=-1:
		k=kfront
                #adding relations to server's response
                relation=ObjectRelation()
                relation.object_id1=str(i)+'.'+listOfObjects[i]["cat"][0]
                relation.object_id2=str(k)+'.'+listOfObjects[k]["cat"][0]
                relation.num_value.append(0.0)
		relation.text_value.append(spatialRelations[0][i][k])
                relation.confidence.append(spatialRelations[1][i][k])
                FINAL_RESULT.answer.relations.append(relation) 	

		
        print("\n************************  3.Mask   ************************\n")
        # Mask
        mask = listOfObjects[i]["masks"][0]
	objprop= ObjectProperty()
        objprop.name="Mask"
        objprop.num_value=mask.reshape([mask.shape[0]*mask.shape[1]])
        objprop.text_value=[]
        objprop.confidence=[0.968]
        objdescr.properties.append(objprop)
        #adding objects to server's response
        FINAL_RESULT.answer.objects.append(objdescr)
    return FINAL_RESULT
##############################################################################################################################################   





def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
                                  [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def plot_precision_recall(AP, precisions, recalls):
    """Draw the precision-recall curve.

    AP: Average precision at IoU >= 0.5
    precisions: list of precision values
    recalls: list of recall values
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictins and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    """Draw bounding boxes and segmentation masks with differnt
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominant each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            x = random.randint(x1, (x1 + x2) // 2)
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))


def display_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for l in layers:
        weight_values = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights  # list of TF tensors
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.4f}".format(w.min()),
                "{:+10.4f}".format(w.max()),
                "{:+9.4f}".format(w.std()),
            ])
    display_table(table)
