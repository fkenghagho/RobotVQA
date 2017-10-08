
# RobotVQA

   Robohow is the European Research Project that aims at enabling robots to competently perform human-scale daily manipulation activities such as cooking in an ordinary kitchen. However, to complete such tasks, robots should not only exhibit standard visual perception capabilities such as captioning, detection, localization or recognition, but also demonstrate some cognitive vision, in which all these capabilities including considerable reasoning are integrated together, so that semantically deep understanding of the unstructured and dynamic scene can be reached. 
   
   In this thesis, we formulate the underlying perception problem of scene understanding in two subproblems:
- **Objects description:** we design and train a deep convo-neural network to provide an end-to-end dense description of objects in the scene. Since context-independent annotation of objects in the scene can be done almost automatically, we can easily generate a big dataset and take advantage of deep learning.
- **Relationships description:** we design and train a markov logic network to provide a relevant description of relationships among objects in the scene. Markov logic networks are suitable for reasoning about relationships among objects, very flexible(few rules for so many behaviors) and the separation of this module from the object description module enables modularity(changes in the one module does not affect the other module).

The following figure briefly illustrates the concept:

![Objects and Relationships description](images/illustration.png "Objects and Relationships description")
