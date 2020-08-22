# Week4 of TSAI-EVA5
## Assignment overview : Need to achive >99.4% test accuracy on MNIST handwritten dataset with following conditions

* 99.4% validation accuracy
* Less than 20k Parameters
* Less than 20 Epochs
* No fully connected layer
* ca use any standard building blonks to achieve the same.

## Approach

* Using convolution layers with 3*3 kernel and maxpooling to reduse the network parameters so that <20k is kept intact. 
  and using GAP(Global average pool) layers before last dense layer
  
## network

```
Net(
  (conv1): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.25, inplace=False)
    (4): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
    (5): ReLU()
    (6): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.25, inplace=False)
    (8): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
    (9): ReLU()
    (10): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (12): Dropout(p=0.25, inplace=False)
  )
  (conv2): Sequential(
    (0): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.25, inplace=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
    (5): ReLU()
    (6): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.25, inplace=False)
    (8): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1))
    (9): ReLU()
    (10): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): Dropout(p=0.25, inplace=False)
  )
  (fc1): Sequential(
    (0): AvgPool2d(kernel_size=7, stride=7, padding=0)
  )
  (fc2): Sequential(
    (0): Linear(in_features=16, out_features=10, bias=True)
  )
)
```

* conv1: 3 convolution are performed with 3x3 kernel so that receptive filed is 7x7 before we hit a maxpool layer, as 7x7 is good enough to capture edges 
and gradients and patterns in MNIST
* conv2 : after conv1, a convolution with 1x1 kernel is added to make the pattern gathering(Ant-man aca DJ), followed by another two convoltions to stop at 
7x7 image before heading into final FC layer
* fc1 : GAP is performed on 2D data before giving it to FC dense layer
* fc2 : layer to coonet from 16 features to number of class i.e 10

## Hyper parameters used:
* LR : 0.01
* Batch size : 64
