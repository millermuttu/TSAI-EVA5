# Session 9 - Data Agumentation
[![Open Jupyter Notebook](images/nbviewer_badge.png)](https://github.com/millermuttu/TSAI-EVA5/blob/master/week8/EVA5_session_9.ipynb)

## Assignment Objective

Implement the transformation using albumentation package, and make a separate transformation for train and test set
  * train set - ToTensor, HorizontalFlip, Normalize, etc
  * Test set - ToTensor and Normalize

Implement Gradcam for visualization of focus of our trained network.
  
  
## Solution:

Implemented both albumentation and gradcam feature.

## Model Hyperparameters

* Batch size: 128
* Epochs = 20
* Learning Rate: step learning rate starting with 0.1 and reducing it after every 6 epochs by 10
  * epochs(1-6) --> LR=0.1
  * epochs(7-12) --> LR=0.01
  * epochs(13-18) --> LR=0.001

## Results

 * Achived grater than 87% validation
