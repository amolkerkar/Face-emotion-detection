# Face emotion detection
## Overview
This project focuses on real-time emotion recognition using a webcam and the OpenCV library. The core of the system involves training Convolutional Neural Networks (CNN) for emotion recognition, also implementing emotion recognition using transfer learning with a DenseNet169 and DenseNet201 architecture on a dedicated dataset.

## Confusion matrix comparison
### DenseNet169: <img src= "https://github.com/amolkerkar/Face-emotion-detection/assets/81116875/7b9d43b3-5741-4276-ba25-2bdc4e521b21" width="600" height="400">
### DenseNet201: <img src= "https://github.com/amolkerkar/Face-emotion-detection/assets/81116875/3a56607e-3bde-4327-a9de-caa9aeaa2b01" width="600" height="400">

## Architecture Highlights
The overall architecture adheres to a pattern of convolutional blocks for feature extraction, followed by fully connected layers for classification. Batch normalization and dropout techniques are implemented to enhance training stability and prevent overfitting.
The softmax activation in the output layer enables the model to predict probabilities for each of the seven emotion classes.
Model Features
DenseNet169 Integration: The project leverages the DenseNet169 model pre-trained on ImageNet for efficient feature extraction.
DenseNet201 Integration: Additionally, the DenseNet201 model pre-trained on ImageNet is employed for feature extraction, contributing to the model's overall performance.

## Sample output:



