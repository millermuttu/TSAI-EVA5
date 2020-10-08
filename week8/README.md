# Session 8 - Receptive field and network architecture

[![Open Jupyter Notebook](images/nbviewer_badge.png)](https://github.com/millermuttu/TSAI-EVA5/blob/master/week8/EVA5_session_8.ipynb)

## Assignment Objective

Use the Resnet18 architecture and run the modulerized code to get a testing accuracy of >85% with any numbers of epochs
  
  
## Solution:

* used the ResNet18 model architecture and achieved the accuracy of >85%  

## Model Hyperparameters

* Batch size: 128
* Epochs = 20
* Learning Rate: step learning rate starting with 0.1 and reducing it after every 6 epochs by 10
  * epochs(1-6) --> LR=0.1
  * epochs(7-12) --> LR=0.01
  * epochs(13-18) --> LR=0.001

## Results

 * Achived grater than 85% validation accurcy at epochs 12 and onwards
