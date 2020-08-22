# [Iteration 1](https://github.com/millermuttu/TSAI-EVA5/blob/master/Week5/EVA5_S5_F1.ipynb)


## Starting point of the assignement: CODE 2 BASIC SKELETON from session 5
## Target:
* Reduce the number of parameters from 194K
* Move model towards achiveing the parameters less 10k and more than 99.4 accuracy
## Result:
* parameters reduced to 12k
* Good scope to reduce the parameters furter in next iters
* Best train accuracy: 99.13
* Best test accuracy: 98.73
## Analysis:
* Definately model is overfitting as there is only 0.87% scope in training accuracy and we have alost same gap to achive in testing accracy.
try to use Regulearization methods like batch norm and dropouts
* also not using GAP and depending on kernels to reduce the size at the end.

# [Iteration 2](https://github.com/millermuttu/TSAI-EVA5/blob/master/Week5/EVA5_S5_F2.ipynb)

## Target:
* Reduce the number of parametes to 10k
* add regulerzation techniques like bacthnorma and dropout

## Result:
* parameter: 10k
* Best training acc: 99.14
* Best testing acc: 99.25

## Analysis:
* good to see train and test acc going along, overfitting is reduced.
* model has still scope to learn in further epoches but we are constrained to 15 epochs, so works needs to be done on model parameters.
* Some of the augmentation techniques can be used to hit the required acc early. 

# [Iteration 3](https://github.com/millermuttu/TSAI-EVA5/blob/master/Week5/EVA5_S5_F3.ipynb)

## Target:
* added Some of the augmentation techniques like rotate, colorjitter to change brightness, contrast, and hue

## Result:
* parameter: 11k
* Best training acc: 99.08
* Best testing acc: 99.50

## Ananlysis:
* our model hits 99.4 acc at 6th epoch and maintains the same till 15th epoch
* reduce the number of parametres to 8k or 10k

# [Iteration 4](https://github.com/millermuttu/TSAI-EVA5/blob/master/Week5/EVA5_S5_F4.ipynb)

## Target:
* Reduce the number of parameters between 8k to 10k
* Use sheduler to decrese the learning late by 0.1 factor after every 6 epochs, startig learning rate being 0.1

## Result:
* parameter: 9.7k
* Best training acc: 99.01
* Best testing acc: 99.46

## Analysis:
* model hits grater than 99.4 acc at epoch 8 and maintains the same till the epochs 15
* test acc is grater than train acc and there is still scope to increse the model accuracy.

# Receptive filed calculation excel
* [File here](https://github.com/millermuttu/TSAI-EVA5/blob/master/Week5/Receptive%20field.xlsx)
