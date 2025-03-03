"""
#@Author:   Frankln Kenghagho
#@Date:     04.04.2019
#@Project:  RobotVA
#Extends the work below Mask R-CNN
"""

"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

from DatasetClasses import DatasetClasses
import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.color
import skimage.io
import urllib
import shutil
import pickle
import cv2
# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

#############################################################
#Data augmentation function


def data_augmentation(image,aug_img_probs=0.93,aug_pix_probs=0.5,pixel_shift=2, image_shift=2):
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



#############################################################
#Pickle load function
#############################################################

def loadFile(filename='../validation.data'):
	with open(filename,'rb') as f:
		res=pickle.load(f)
		f.close()
	print(res)
	return res

#############################################################
#Random functions
#############################################################
def randomIndex(N):
    assert N>0
    summ=0
    for i in range(N):
        summ+=int(os.urandom(5).encode('hex'),16)
        summ=summ%N
    return summ%N
############################################################
#  Bounding Boxes
############################################################

def bbox_distance(r1,r2,flags):
    """Compute spatial relation between two rectangles r1,r2
    """
    y1, x1, y2, x2=r1
    y3, x3, y4, x4=r2
    if not (((x3>=x1 and x3<=x2) or (x1>=x3 and x1<=x4)) and ((y3>=y1 and y3<=y2) or (y1>=y3 and y1<=y4))):
        # no intersection between r1 and r2. distance between r1 and r2's centers
        return np.array([-np.inf,np.sqrt(0.25*(x1+x2-x3-x4)**2+0.25*(y1+y2-y3-y4)**2)])
    else:
         #amount ratio of r1 in r2
         if((x3>=x1 and x3<=x2)):
             width=abs(x3-np.minimum(x2,x4))
         else:
             width=abs(x1-np.minimum(x2,x4))
             
         if((y3>=y1 and y3<=y2)):
             height=abs(y3-np.minimum(y2,y4))
         else:
             height=abs(y1-np.minimum(y2,y4))
         if (width*height)<1e-3:
            return np.array([-np.inf,np.sqrt(0.25*(x1+x2-x3-x4)**2+0.25*(y1+y2-y3-y4)**2)]) 
            
         if(((x2-x1)*(y2-y1)==(x4-x3)*(y4-y3)) and flags) or ((x2-x1)*(y2-y1)>(x4-x3)*(y4-y3)) :
            return np.array([-np.inf,np.inf]) 
         return np.array([(width*height)/((x1-x2)*(y1-y2)),np.inf])
         
def map_bbox_distance(bbox_list):
    """Compute the image of the bbox_distance relation over set bbox_list x bbox_list
    """
    N,M=bbox_list.shape
    result=np.zeros([N,N,2],dtype='float32')
    for i in range(N):
        for j in range(N):
            if i!=j:
                
                result[i,j]=bbox_distance(bbox_list[i],bbox_list[j],(i>j)*1)
            else:
                result[i,j,0]=-np.inf
                result[i,j,1]=np.inf
    return result
    
def relation_graph(bbox_list):
    """Summarize the above computed graph through hierarchical clustering
    """
    result=map_bbox_distance(bbox_list)
    res_area=np.argmax(result[:,:,0],axis=1)
    i=np.where(np.less_equal(np.max(result[:,:,0],axis=1),0))[0]
    res_dist=np.argmin(result[:,:,1],axis=1)
    res_area[i]=res_dist[i]
    return res_area


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    '''
    # flatten masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


def normalSurface(depthImg,max_distance,depth="float32"):
    #we assume depth is of type raw float
    if depth=="float32":
        depthImg=cv2.imread(depthImg,cv2.IMREAD_ANYDEPTH)
        x,y=np.where(depthImg>max_distance)
        depthImg[x,y]=max_distance
        if max_distance!=0.:
            depthImg=depthImg/max_distance
    else:
        depthImg=np.array(cv2.imread(depthImg)[:,:,0],dtype="float32")
        depthImg=depthImg/255.0
    depthImage=depthImg*255.0
    depthImg=depthImg-1.0
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
    normalImg=np.array(normalImg,dtype='uint8')    
    return cv2.cvtColor(normalImg,cv2.COLOR_BGR2RGB),depthImage

def load_image(litImg,depthImg,max_distance,depth="float32"):
        # Load image
        image = skimage.io.imread(litImg)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image =np.array(skimage.color.gray2rgb(image),dtype='float32')
        try:
            depthImage1,depthImage2=normalSurface(depthImg,max_distance,depth=depth)
	    shapes=list(image.shape)[:2]
	    shapes.append(4)
            depthImage=np.zeros(shapes,dtype='float32')
	    depthImage[:,:,:3]=depthImage1
	    depthImage[:,:,3]=depthImage2
	    print('Load depth successfully!')
        except Exception as e:
	    shapes=list(image.shape)[:2]
	    shapes.append(4)
	    print('Failed to load depth:'+str(e))
            depthImage=np.zeros(shapes,dtype='float32')
        shape=list(image.shape)
        shape[len(shape)-1]=shape[len(shape)-1]+4
        finalImage=np.zeros(shape,dtype='float32')
        #finalImage[:,:,:3]=data_augmentation(image)
	finalImage[:,:,:3]=image
        finalImage[:,:,3:]=depthImage
        return finalImage

############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [ [{"source": "", "id": 0, "name": "BG"}],#category
                            [{"source": "", "id": 0, "name": "BG"}],#color
                            [{"source": "", "id": 0, "name": "BG"}],#shape
                            [{"source": "", "id": 0, "name": "BG"}],#material
                            [{"source": "", "id": 0, "name": "BG"}],#openability
                            [{"source": "", "id": 0, "name": "BG"}],#relation
                            [{"source": "", "id": 0, "name": "BG"}]#relation category
                          ]
        self.source_class_ids = {}

    def add_class(self,feature_id, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info[feature_id]:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info[feature_id].append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes=[]
        self.class_ids=[]
        self.class_names=[]
        self.class_from_source_map=[]
        for i in range(len(self.class_info)):
            self.num_classes.append(len(self.class_info[i]))
            self.class_ids.append(np.arange(self.num_classes[i]))
            self.class_names.append([clean_name(c["name"]) for c in self.class_info[i]])
            self.class_from_source_map.append({"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info[i], self.class_ids[i])})

        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        
        # Map sources to class_ids they support
        self.sources=[]
        self.source_class_ids=[]
        for i in range(len(self.class_info)):
            self.sources.append(list(set([info['source'] for info in self.class_info[i]])))
            self.source_class_ids.append({})
            # Loop over datasets
            for source in self.sources[i]:
                self.source_class_ids[i][source] = []
                # Find classes that belong to this dataset
                for j, info in enumerate(self.class_info[i]):
                    # Include BG class in all datasets
                    if j == 0 or source == info['source']:
                        self.source_class_ids[i][source].append(j)

    def map_source_class_id(self,feature_id, source_class_id):
        """Takes a source class ID and target target, then returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("category","coco.12") -> 23
        """
        return self.class_from_source_map[feature_id][source_class_id]

    def get_source_class_id(self,feature_id, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[feature_id][class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = []
        for j in range(len(self.class_info)):
            self.external_to_class_id.append({})
            for i, c in enumerate(self.class_info[j]):
                for ds, id in c["map"]:
                    self.external_to_class_id[j][ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]
        
   

    def load_image(self, image_id,max_distance,depth="float32"):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        litImg = self.image_info[image_id]['path']
        depthImg = self.image_info[image_id]['depthPath']
        return load_image(litImg,depthImg,max_distance,depth=depth)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids= np.empty([0], np.int32)
        poses= np.empty([0], np.float32)
        relations=np.empty([0,0,0], np.int32)
        return mask, class_ids, poses, relations


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = cv2.resize(
            image, (int(round(w * scale)), int(round(h * scale))))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask
    
def resize_poses(poses, scale, padding):
    """Resizes a poses using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the poses(in image coordinates) are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the postions in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    
    poses[:,3:5]=poses[:,3:5]*scale
    poses[:,3] = poses[:,3]+padding[1][0]#the x coordinate is only affected by the left padding
    poses[:,4] = poses[:,4]+padding[0][0]#the y coordinate is only affected by the top padding
    #the z coordinate is kept unchanged
    return poses



def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
        mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding and sort predictions by score from high to low
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through ground truth boxes and find matching predictions
    match_count = 0
    pred_match = np.zeros([pred_boxes.shape[0]])
    gt_match = np.zeros([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] == 1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                break

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None,parallel_processing=False):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    
    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        if parallel_processing:
            output_slice = graph_fn(inputs_slice)
        else:
            output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def getPositionFromCamToImg(objPosition):
    """Convert objPosition from camera system(cm) to image system(pixels)
    """
    
    objPosition=np.array([objPosition[1],objPosition[2],objPosition[0]],dtype='float32')/DatasetClasses.PIXEL_SIZE
    objPosition=np.dot(DatasetClasses.CAMERA_INTRINSIC_MATRIX,objPosition)
    objPosition[:2]/=objPosition[2]
    objPosition[1]=2*DatasetClasses.CAMERA_INTRINSIC_MATRIX[1,2]-objPosition[1]
    return objPosition

#principal measure in radians of  an  angle[0,2pi[:
def principal_angle(angle):
    angle=np.pi*angle/180.
    if angle<0.:
        inc=np.pi*2
    else:
        inc=-np.pi*2
    if(abs(angle)<DatasetClasses.OBJECT_ORIENTATION_PRECISION):
	angle=0.0
    while(not(angle<2*np.pi and angle>=0.)):
        angle+=inc
	if(abs(angle)<DatasetClasses.OBJECT_ORIENTATION_PRECISION):
		angle=0.0
    return angle

def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")
