# Session 9 - Advance concepts in training and Learning rates
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dqPeV8WTlyAzFSDPXt5CoAyVO96tfvcj)

## Assignment Objective

* Implement cutout the transformation using albumentation package, and make a separate transformation for train and test set
  * train set - HorizontalFlip, cutout, Normalize, ToTensor
  * Test set - Normalize and ToTensor
* Implement LR finder with range test to find out the best LR 
* implement ReduceLROnPlatea

Gradcam for visualization of focus of our trained network. and generate a batch gradcam for visulaization of 25 images together.
  
  
## Solution:

* Used cutout with size (8,8) transformation.
* Implemented the LR finder with range test
* implemented ReduceLROnPlatea sheduler


## Results

 * Achived expected validation accuracy

## Accuracy and Loss
<img src="week10/images/acc.png" width="450px"> <img src="week10/images/loss.png" width="450px">

## LRfinder with range test
<img src="week10/images/lrfinder.png" width="450px">

## Gradcam cam result for layers
<img src="week10/images/grad_imgs_layer__1.png" width="450px">
<img src="week10/images/grad_imgs_layer__2.png" width="450px">

## 25 misclassified images
<img src="week10/images/misclass.png" width="720px">
