# Computer Vision Nanodegree
## Projects

### Project 1: Facial KeyPoint Detection
In this project, I built a facial key point detection system using Haar Cascades and CNN.Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. First, I used Haar Cascades to detect and extract faces from an image, then use a CNN to detect the keypoints.
Some of the results include:

![Obama](/CVND_Projects/P1_Facial_Keypoints/images/obamas.jpg)

![Obama](/CVND_Projects/P1_Facial_Keypoints/images/obamasHaar.png)

![Obama](/CVND_Projects/P1_Facial_Keypoints/images/obam.png)

-------------------------------

### Project 2: Image Captioning
In this project I built an image captioning model. When given an image the model outputs a description of that image.
I used an attention model, which is a combination of a CNN and LSTM. I learnt:
- How to combine CNNs and RNNs to build a complex captioning model.
- Implemented an LSTM for caption generation.
- Trained a model to predict captions and understand a visual scene.

Some results are:

![caption](/CVND_Projects/P2_Image_Captioning/images/caption1.png)

![caption](/CVND_Projects/P2_Image_Captioning/images/caption3.png)

![caption](/CVND_Projects/P2_Image_Captioning/images/caption4.png)


And some not so good results

![caption](/CVND_Projects/P2_Image_Captioning/images/CaptionNotSoGood.png)

--------------------------------

### Project 3: Landmark Detection and Tracking


## Exercises

### <span style="color:red">Introduction to Computer Vision<span>

| **Lesson Title**                                                                                                                                             | **What I learned**                                                                                                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **[Image Representation and Analysis](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/Exercises/1_1_Image_Representation)**               | - Saw how images are represented numerically <br/>- Implemented Image Processing techniques like color and geometric transforms<br/>- Programmed my own convolutional kernel for object edge-detection<br/>                                                                               |
| **[Convolution NN Layers](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/Exercises/1_2_Convolutional_Filters_Edge_Detection)**        | - Learned about the layers of a deep CNN: (Convolutional, maxpooling and fully connected layers)<br/>- Build a CNN-based image classifier in PyTorch<br/>- Learned about layer activation and feature visualization techniques<br/>                                                      |
| **[Features And Object Recognition](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/Exercises/1_3_Types_of_Features_Image_Segmentation)** | - Learned why distinguishing features are important in pattern and object recognition tasks<br/>- Wrote code to extract information about an object's color and shape<br/>- Used features to identify areas on a face and to recognize the shape of a car or pedestrian on the road<br/> |
| **[Image Segmenation](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/Exercises/1_4_Feature_Vectors)**                                   | - Implemented k-means clustering to break an image up into parts<br/>- Found the contours and edges of multiple objects in an image<br/>- Learned about background subtraction for video<br/>                                                                                            |


### Advanced Computer Vision and Deep Learning

| **Lesson Title**                                                                                                               | **What I learned**                                                                                                                                                                                                                                                                                           |
|--------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **[Advanced CNN Architectures](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/Exercises/1_5_CNN_Layers)** |- Learned about advances in CNN architectures<br/>- Saw how region-based CNN's like Faster R-CNN, have allowed for fast,localized object recognition in images<br/>- Worked with a [YOLO](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/Exercises/2_2_YOLO) /single shot object detection system<br/> |
| **[Recurrent Neural Networks](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/Exercises/2_4_LSTMs)**        | - Learned how RNNs learn from ordered sequences of data <br/>- Implemented an RNN for sequential text generation<br/>- Explored how memory can be incorporated into a deep learning model<br/>- Understood where RNN's are used in deep learning applications<br/>                                                             |
| **[Attention Mechanisms](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/Exercises/2_6_Attention  )**       | - Learned how attention allows models to focus on a specific piece of input data<br/>- Understood where attention is useful in natural language and computer vision applications<br/>                                                                                                                                 |


### Object Tracking and Localization
| **Lesson Title**                                                                                                                                                  | **What I learned**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **[Object Motion and Tracking](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/CVND_Localization_Exercises/4_2_Robot_Localization)**           | - Learned how to programmatically track a single point over time<br/>- Understood motion models that define object movement over time<br/>- Learned how to analyze videos as a sequence of individual image frames<br/>                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **[Optical Flow and Feature Matching](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/CVND_Localization_Exercises/4_5_State_and_Motion)**      | - Implemented a method for tracking a set of unique features over time<br/>- Learned how to match features from one image frame to another<br/>- Tracked a moving car using optical flow<br/>                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **[Robot Localization](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/CVND_Localization_Exercises/4_6_Matrices_and_Transformation_of_State)** | - Used Bayesian statistics to locate a robot in space<br/>- Learned how sensor measurements can be used to safely navigate an environment an environment<br/>- I learn about the matrix operations that underly multidimensional Kalman Filters<br/>- Understood Gaussian uncertainty and the basics of [Kalman Filters](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/CVND_Localization_Exercises/4_4_Kalman_Filters)<br/>- Implemented a histogram filter for robot localization in Python([2D Histogram Filter](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/CVND_Localization_Exercises/4_3_2D_Histogram_Filter)) |
| **[Graph SLAM](https://github.com/ivyclare/Computer-Vision-Nanodegree/tree/master/CVND_Localization_Exercises/4_7_SLAM)**                                         | - Identified landmarks and built a map of an environment<br/>- Learned how to simultaneously localize an autonomous vehicle and create a map of landmarks<br/>- Implement move and sense functions for a robotic vehicle<br/>                                                                                                                                                                                                                                                                                                                                                                                                                                 |
