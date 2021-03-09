# SocialDistancingDetector
Deep Learning Algorithm written in Python using Keras anf TF to detect Social Distancing violation from CCTV images

![](/images/result.png)

## Methods

The algorithm for the control on Social Distancing is composed of two parts: 
* A deep learning model for object detection that allows to identify all the subjects involved in the image; 
* On the predictions made in the previous point a geometric distance calculation is carried out to quantify the interpersonal distance and therefore to identify the subjects that violate Social Distancing.
This type of approach guarantees satisfactory results since they depend strongly on the quality of the algorithm of object detection which, at the present state of the art, is a task widely debated and perfected, obtaining always better results.

### Object Detection
For the detection algorithm YOLO (You Only Look Once) was used, especially in its latest version, version 4 [2]. YoloV4 is a major improvement of YoloV3: the implementation of a new backbone architecture has improved the mean Average Precision by 10% and the number of FPS (Frames per second) by 12%. In addition, it has become easier to train the network on a single GPU. YOLO is based on a single Convolutional Neural Network (CNN). The CNN divides an image into regions and then predicts the bounding boxes (BB) and probabilities for each region, simultaneously predicting multiple bounding boxes and probabilities for those classes. The peculiarity of YOLO is that it sees the entire image during training and testing, so it implicitly encodes contextual information about the classes as well as their appearance. The architecture of YOLOv4 consists of CSPDarknet53 as the backbone, an additional spatial pyramid pooling module, a PANet (Path-Aggregation neck) and YOLOv3 as the head of the architecture. CSPDarknet53, the new backbone, can improve the learning capacity of the CNN. The spatial pyramid pooling block is added on top of CSPDarknet53 to increase the receptive field and separate the most significant context features. Instead of the feature pyramid networks (FPNs) for object detection used in YOLOv3, YOLOv4 uses PANet as a method for parameter aggregation for different detector layers.

![How YOLO works](https://www.pyimagesearch.com/wp-content/uploads/2018/11/yolo_design.jpg)

### Distance calculation
The distance calculation part, on the other hand, is implemented by means of geometric calculation. It takes in input the classification and bounding box data and calculates the distance between the subjects in the image and then detects the transgressions of Social Distancing. Two algorithms have been proposed: 
* Calculation of the distance from the image with the camera POV (CCTV view) 
* Distance calculation from bird-eye view.

#### CCTV view method
For each bounding box it is necessary to find a point representing the Euclidean position of each person. The choice was made for the lower midpoint of each bounding box in order to have an invariant measurement with respect to the height of the subject. The Euclidean distance between each point in the image is then calculated. For each person, it is then compared the minimum distance measured with a threshold and, for values below this threshold, marking the violation with a red bounding box. In order to take into account the perspective of the image, a grid with different threshold values was designed so that each midpoint was associated with a different "perspective weight" depending on its position in the grid. A position further away from the camera corresponds to a lower weight and vice versa.

#### Bird-eye view method
The problem of perspective is approached in a different way by the Bird-eye view method: a transformation of the perspective of the image is carried out, as depicted in the following figure. The capabilities of OpenCV are exploited in order to transform, by means of a transformation matrix, a part of the image captured by a CCTV into an overhead view of this image. Then the same matrix is used to calculate, for each person, the "GPS" coordinates in the frame, which are accurate for the representation of the position of the subjects and the consequent distance measurement. A limitation of this method is that it only considers a part of the image, as a portion of the image is lost during the transformation into a top view.

![Bird eye transformation](/images/transformation.png)


### Reference
[taipingeric
/
yolo-v4-tf.keras](https://github.com/taipingeric/yolo-v4-tf.keras)

[[2] Dataset](https://exposing.ai/oxford_town_centre/)

