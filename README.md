
# RobotVQA

# 1. Introduction 


   Robohow is the European Research Project that aims at enabling robots to competently perform human-scale daily manipulation activities such as cooking in an ordinary kitchen. However, to complete such tasks, robots should not only exhibit standard visual perception capabilities such as captioning, detection, localization or recognition, but also demonstrate some cognitive vision, in which all these capabilities including considerable reasoning are integrated together, so that semantically deep understanding of the unstructured and dynamic scene can be reached.
   
   In this thesis, we formulate the underlying perception problem of scene understanding as two subproblems:
- **Objects description:** we design and train a deep convo-neural network to provide an end-to-end dense description of objects in the scene. Since context-independent annotation of objects in the scene can be done almost automatically, we can easily generate a big dataset and take advantage of deep learning.
- **Relationships description:** we design and train a markov logic network to provide a relevant and consistent description of relationships among objects in the scene. Markov logic networks are suitable for reasoning about relationships among objects, very flexible(few rules for so many behaviors) and the separation of this module from the object description module enables modularity(changes in the one module does not affect the other module).


# 2. Typical Scene 

  
   The following figure briefly illustrates the concept:

![Objects and Relationships description](images/illustration.png "Objects and Relationships description")


# 3. Deep Convo-Neural Network


   Our model works as follows:

![Objects and Relationships description](images/architecture.png "Model description")



# 4. Markov Logic Network 


   An illustrative markov logic network for consistent description of objects as ell as relationships among them follows:

**Types declaration**

*object={BOWL_1,SPOON_2,MUG_3,...}*

*name={SPATULA, BOWL, SPOON, KNIFE, MUG,...}*

*pickability={TRUE, FALSE}*

*inMaterial={SOLID, LIQUID, GAZ, POWDER}*

*outMaterial={CERAMIC, WOOD, GLASS, STEEL, PLASTIC, CARTOON}*

*shape={CUBIC, CYLINDRICAL, CONICAL, FLAT, FILIFORM, SPHERICAL, PYRAMIDAL}*

**Predicates declaration**

*objectName(object,name)*

*objectShape(object,shape)*

*objectPickability(object, pickability)*

*objectOutMaterial(object, outMaterial)*

*objectInMaterial(object, inMaterial)*

*object(object)*

*container(object)*

*throw(object,object)

**Rules declaration**

*Vx(objectName(x, SPATULA) => (objectOutMaterial(x, WOOD) v objectOutMaterial(x, STEEL) v objectOutMaterial(x, PLASTIC))* , **weight=?**

*Vx(objectName(x, SPATULA) => objectInMaterial(x, SOLID))* , **weight=?**

*Vx,y,m,n(objectName(x, m) ^ objectName(x, n) ^ (n=m) => (x=y))* , **weight=+infinity (hard constraint)**

*Vx((objectName(x, MUG) v objectName(x, BOWL) v objectName(x, GLASS)) => container(x))* , **weight=?**

*Vx,y(container(x) ^ objectInMaterial(y, LIQUID) ^ (x=/=y) => throw(y,x))* ,  **weight=?**



# 5. Frameworks

   We make use of the following Frameworks:

- **Unreal Engine and UnrealCV:** to partially build the visual datasets
- **TensorFlow and Caffe:** to build the deep convo-neural network, train it and make inferences
- **Pracmln and Alchemy:** to build the markov logic network, train it and make inferences 


# 6. Dataset 

   The structure of the visual dataset can be found at [dataset's structure](https://github.com/fkenghagho/RobotVQA/blob/master/dataset/datasetStructure.txt). This file deeply specifies the structure of objects in the scene and the image and information types needed to learn the structure of objects. Then, we modeled the environment of the target robot in Unreal Engine4.16 and wrote a software for an end-to-end construction of the dataset starting from the automatic camera navigation in the virtual environment till the saving of scene images and annotations. The software can be found at [dataset generator](https://github.com/fkenghagho/RobotVQA/blob/master/tools/generateDataset.py). An example of annotation can be downloaded from [example of annotation](https://github.com/fkenghagho/RobotVQA/blob/master/dataset/datasetExample.zip)(**zip file**).
   
   The following images depict a typical scene:
   
![Typical scene](images/dataset1.png "Typical scene")   

![Objects in the scene](images/objects.png "Objects in the scene")   



# 7. Object Color Estimation: Learning vs Engineering

   In this thesis, we identify 12 classes of colors which are the basic/standard colors set up by the **ISCC–NBS System**. These colors are **red**, **blue**, **green**, **yellow** ...
   
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


