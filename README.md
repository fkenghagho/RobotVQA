
# RobotVQA

# 1. Introduction 


   Robohow is the European Research Project that aims at enabling robots to competently perform human-scale daily manipulation activities such as cooking in an ordinary kitchen. However, to complete such tasks, robots should not only exhibit standard visual perception capabilities such as captioning, detection, localization or recognition, but also demonstrate some cognitive vision, in which all these capabilities including considerable reasoning are integrated together, so that semantically deep understanding of the unstructured and dynamic scene can be reached.
   
   In this thesis, we formulate the underlying perception problem of scene understanding as two subproblems:
- **Objects description:** we design and train a deep convo-neural network to provide an end-to-end dense description of objects in the scene. Since context-independent annotation of objects in the scene can be done almost automatically, we can easily generate a big dataset and take advantage of deep learning.
- **Relationships description:** we design and train a relational neural network for computing relationships among objects in the scene. Relationships essentially encode scale-invariant relative positioning (on-the, in-the, left-of, under-the, ...) as well as composition (has-a). The network takes as inputs the input image code(Referential) and the codes(Mask(local infos)+Pose) of two objects in the image, then outputs the most likely relationship between both objects. Note that this can be regarded as a soft or more semantic variant of the hard object pose estimation performed by the object descriptor mentioned above. The annotation of images with relationships is done hybridly ie. automatically along with the description of objects and fine-tuned manually with a specialized software.

As far as the system training and inference are concerned, both networks are integrated together into a single multi-task deep convo-neural network where training and inference take place end-to-end. 

As output, the system returns a scene graph. A scene graph is a directed graph, whose nodes and edges respectively encode objects description and relationships among objects in the scene.  



# 2. Typical Scene 

  
   The following figure briefly illustrates the concept of scene graph:

![Objects and Relationships description](images/scenegraphFinal.png "Objects and Relationships description")


# 3. Multi-Task Deep Convo-Neural Network


   Our model works as follows:

![Objects and Relationships description](images/architecture2.png "Model description")



# 5. Frameworks

   We make use of the following Frameworks:

- **PythonLibs, Unreal Engine and UnrealCV:** to build the dataset 
- **PythonLibs, TensorFlow and Caffe:** to build the deep convo-neural and relational networks, train them and make inferences



# 6. Dataset 

   The structure of the visual dataset can be found at [dataset's structure](https://github.com/fkenghagho/RobotVQA/blob/master/dataset/datasetStructure.txt). This file deeply specifies the structure of objects in the scene and the image and information types needed to learn the structure of objects. For building a huge dataset, we model the environment of the target robot in Unreal Engine4.16, which is a photorealistic game engine allowing therefore efficient transfer learning. Then,  we write a software for an end-to-end construction of the dataset starting from the automatic camera navigation in the virtual environment till the saving of scene images and annotations. To enable a training of the relational network, we augment the annotation of the collected images by loosely specifying relationships among objects. Our annotation software can be found at [dataset generator](https://github.com/fkenghagho/RobotVQA/blob/master/tools/generateDataset.py). An example of annotation can be downloaded from [example of annotation](https://github.com/fkenghagho/RobotVQA/blob/master/dataset/datasetExample.zip)(**zip file**).
   
   The following images depict the annotation of a typical scene from objects to relationships description:
   
![Typical scene](images/dataset1.png "Typical scene")   

![Objects in the scene](images/objects.png "Objects in the scene")   

![Relationships among Objects in the scene](images/relationAnnotation.PNG "Relationships among Objects in the scene") 

# 7. Object Color Estimation: Learning vs Engineering

   In this thesis, we semantically select as many color classes as needed, which are the basic/standard colors set up by the **ISCC–NBS System**, to colourfully label the robot environment. These colors are **red**, **blue**, **green**, **yellow** ...
   
**Engineering approach:** we assume that the pixels's colors are normally distributed as it usually seems to be the case for natural images(Gaussian mixtures). We find the most frequent pixel color(RGB) and compute the nearest color class to the most frequent pixel color. The color class is then considered as the color of the object. By modeling colors as 3D-vectors/points in the **Lab-Color space**, we define a consistent distance between two colors, which captures human visual ability for color similarity and dissimilarity. 
- Quantitatively more precise than human vision, however fails to capture quality: **too few qualitative classes(only 12) and very low features(Gaussian mean of pixel values)**
- Very sensitive to noise: **shadow, very strong mixture of colors**
- Very simple and fast.

**Learning approach:** By building the dataset for learning, we objectively assign color classes to objects and after the training, the network is able to qualitatively appreciate the color of objects in the scene.
- Very powerful: end-to-end rational color estimation only based on objective description of data
- Qualitatively matches human vision **(color constancy)**
- More complicated and costlier: objective description of data, network, training and estimation.


# 8. Object Pose Estimation: Position, Orientation and Cuboid
                 
Refering to the best of our knowledge, we are the first to design an **end-to-end object pose estimator** based on Deep-Learning. From Unreal Engine, we sample images annotated with objets and cameras' poses within the virtual world coordinate system. Then we refine the annotation by mapping the objects' poses  from the world's coordinate system into the camera's coordinate system. The end-to-end object pose estimator takes as input an image and outputs the poses of objects on the image. Futhermore, the estimator also outputs the minimal cuboid enclosing an object.
   
A brief comparison of our learning approach to other traditional approaches on pose estimation follows:

**Traditional approach:** state of the art as PCL requires a lot of engineering
- A 3D-Model for each objects, explicit features extraction: almost manually
- Calibration(camera)
- Clustering-based scene segmentation: imposes some restrictions on the scene configuration(colorfulness, sparseness, ...)

**Learning approach:**
- very natural: only based on data
- No engineering effort: end to end
- Almost no restriction on the scene configuration(colorfulness, sparseness)
- Huge dataset: no problem, we do it almost automatically

The following figure illustrates how we compute the objects' poses for annotation/learning:

![Pose estimation](images/poseEstimation3.png "Pose estimation")

Given that the dataset sampling camera's intrinsic matrix might be different from the ones of the robot camera, we evaluate the
intrisic matrices *R1,R2* respectively of the sampling camera and the robot camera once and properly map the outputted position *(X,Y,Z)* by the network as *inverse(R2)xR1x(X,Y,Z)*. The following figures depicts a demonstration of the camera projection.

![Camera Projection](images/TestCamProj.png "Camera Projection")


# 9. Object Shape Recognition

As far as 3D-shape recognition in Computer Vision is concerned, our main contribution in this thesis is the setup of a pure learning-based or data-driven 3D-shape recognizer, which was not at least deeply conducted so far. In contrast to most existing methods, which are subjects to multiple restrictions such as object orientations, lighting and colors due to engineered feature extraction such as shape descriptors, our method naturally establishes a context-free correlation between objects and 3D-shapes. 

However, to reach an efficient(speed and accuracy) deep learning, it is recommended to supervise it as much as possible. That is by feeding the network with as much information as possible, by pretraining its submodules or by adjusting the general structure of the network to avoid useless computation.

Given an RGB-D image in our context, we design a Sobel/Scharr-like convolution layer at the entry of the network to compute from the Depth(D)-map a normal map, which seems to convey more information about the shape to the Human visual system. The following figure illustrates the idea.

![Shape recognition](images/shaperecognition.png "Shape recognition")

# 10. Materials, Graspability, Pickability, Stackability and Openability

As cleary stated in the introduction, we aim at developing a cognitive vision module for human-like manipulation activities. Therefore, one of our goal is to design a mechanism being also able to track non-standard visual properties namely the material, the graspability, the stackability and the openability of objects in the scene, which actually play a crucial role for reasoning about how to handle the objects during the manipulation phase.

- **Materials:**  we further distinguish external material from internal material of an object. For instance a bottle of juice can be externally seen as plastic but internally as liquid. While the external material may inform about how much pressure the object should undergo and what can be put inside the object, the internal material may inform about how to carry, empty or fill the object(container). As far as the computation of this feature is concerned, the object surface's behavior faced to light can be determinant.

- **Graspability:** another important thing to know about objects in the manipulation environment is  whether or not they can be grasped: we call this feature graspability. It is the quality of an object to be held purposely by the robot arm. It can be inferred but not systematically from other object' properties such as *pickability, stackability or openability*. The system can also find deep correlation factors such as shape, location or function of objects to track this feature.

- **Pickability:** this feature is very similar to *graspability* however different. Pickability of an object refers to how easy the object can be grasped, lifted up and moved. This feature is very common to small objects and can be inferred from the object's shape, 
size or material.

- **Stackability:**  an object is stackable if and only if another similar object can be placed on top of it. This is very important when tryng to free the place occupied by objects. Determinants for this feature can be the object's shape, flatness or openness.

- **Openability:** this feature informs us about whether or not an object can be opened. This is very important for getting into objects such as a bottle or fridge 

# 11. Objects Categories

In our virtual environment, we try to categorize objects as abstractly as possible to capture generality. That is, instead of categorizing an object as **FridgeDrawer**, it would be much better to  simply categorize it as **Drawer**, because there are only few instances of the category **FridgeDrawer** in the environment but so many instances of particular drawers. As far as the complementarity of **Fridge** is concerned, this is ensured by the relationship **has(Fridge, Drawer)**. This is also frequently observed with packaged food items: since the overall visual configuration of packages of a category(Coffee) can drastically vary, what would be the most common sufficient features to most packages? Observations show that the most common sufficient features to most packages are neither the package's shape, nor its color, nor the words written on it but rather the image depicting the content of the package. This discovery merely reinforces the idea, that objects should be carefully categorized to enable an efficient machine learning.


# 12. Relationships among Objects

Some information, that we would like to get from the scene such as the semantic location *(in the pot , on the table, ...)* of objects or the possible compositional  relationships *( has(Fridge,Door), has(Door, Handle) ...)* between them could also be computed by our deep convo-neural network, however a very big set of manually annotated images would be required. A manual annotation because those properties in contrast to the ones mentioned earlier cannot be dynamically estimated during the sampling of data from the simulated virtual world of the robot. In this thesis, we address the problem of relationships determination with a relation-specific network. As described in the introduction, this seperation of modules(networks) allows us to capture dynamic properties of the scene such as objects relationships with a loosely annotated set of images.
