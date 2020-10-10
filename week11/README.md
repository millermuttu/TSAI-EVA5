# Session 11 - Super Convergence
[![Open Jupyter Notebook](week11/images/nbviewer_badge.png)](https://github.com/millermuttu/TSAI-EVA5/blob/master/week11/EVA5_session_11.ipynb)

## Assignment Objective
- Write a code to generate the plot for Cyclic LR representation.
- Write the network as described below
  uses this new ResNet Architecture for Cifar10:
  * PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
  * Layer1 -
    - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    - Add(X, R1)
  * Layer 2 -
    - Conv 3x3 [256k]
    - MaxPooling2D
    - BN
    - ReLU
  * Layer 3 -
    - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    - Add(X, R2)
  * MaxPooling with Kernel Size 4
  * FC Layer 
  * SoftMax
 - Use the following transformation
    * Padding-->Randomcrop-->flip-->cutout
 - Achive accuracy of **>90%**

## Results
 * Network as described is implemented (DevidNET)
 * Achived expected validation accuracy of **90%**
 * implemented the plot for CLR
 * used Batch size of 512
 * transform used: Padding-->Randomcrop-->flip-->cutout

## Accuracy and Loss
![i](week11/images/acc.png) ![i2](week11/images/loss.png)

## LRfinder with range test
![img3](week11/images/lrfinder.png)

## Cyclic LR plot
![img5](week11/images/clr.png)
