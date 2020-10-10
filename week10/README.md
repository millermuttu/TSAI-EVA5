# Session 10 - Advance concepts in training and Learning rates
[![Open Jupyter Notebook](images/nbviewer_badge.png)](https://github.com/millermuttu/TSAI-EVA5/blob/master/week10/EVA5_session_10.ipynb)

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
![img1](images/acc.png) ![img2](images/loss.png)

## LRfinder with range test
![img3](images/lrfinder.png)

## Gradcam cam result for layers
![img4](images/grad_imgs_layer__1.png)
![img5](images/grad_imgs_layer__2.png)

## 25 misclassified images
![img6](images/misclass.png)
