# Real Face Detection
Capture faces and then using different library to recognized these faces thought web cam, images, video with Liveness Detection.

The repository does not contain the dataset of images. You should capture yourself and train it for later recognization.

#### Face detection algorithms using :
+ Haar Cascade Classifiers using OpenCV
+ Histogram of Oriented Gradients using Dlib
+ Convolutional Neural Networks using Dlib

#### Libraries using: 
+ python 3.7
+ anaconda3
+ dlib
+ opencv
+ imutils
+ face_recognition

#### Installation:

```sh
pip install cmake
conda install -c menpo dlib
conda install -c akode face_recognition_models
pip install face_recognition
pip install imutils
```
#### Using:
Capture pictures
```sh
python face_capture.py -n <name>
```
#### Reference:
A full guide to face detection - Maël Fabien
https://maelfabien.github.io/tutorials/face-detection/#

Liveness Detection with OpenCV - Adrian Rosebrock
https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/