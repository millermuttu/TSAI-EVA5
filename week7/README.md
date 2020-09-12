# Session 6 - Advanced convolutions

[![Open Jupyter Notebook](nbviewer_badge.png)](https://github.com/millermuttu/TSAI-EVA5/blob/master/week7/EVA5_Session_7.ipynb)

## Assignment Objective

1. Fix the network giveb with below objectives
    * change the code such that it uses GPU
    * change the architecture to C1C2C3C40 (basically 3 MPs)
    * total RF must be more than 44
    * one of the layers must use Depthwise Separable Convolution
    * one of the layers must use Dilated Convolution
    * use GAP (compulsory):- add FC after GAP to target #of classes (optional)
    * achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M.
  
  
## Solution:

1. Model Hyperparameters
    * Dropout: 0.05
    * Batch size: 64
    * Learning Rate: 0.003
    * L1 parameter: 0.0002
    * L2 parameter: 0.0001
    * Ghost Batch Norm Splits: 4
  
2. Model architecture is built including Depthwise seperable convolutions and Dilated Convolutions

3. Total parameters : 838,544

## Results

 * Achived grater than 80% validation accurcy at epochs 21 and onwards
