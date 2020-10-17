# Session 12 - Object localization YOLO
[![Open Jupyter Notebook](images/nbviewer_badge.png)](https://github.com/millermuttu/TSAI-EVA5/blob/master/week12/EVA5_session_12.ipynb)

## Assignment Objective
* train ResNet18 on the tiny imagenet dataset [download](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
* Target >50% Validation Accuracy with 50 epochs
* Download 50 (min) images each of people wearing hardhat, vest, mask and boots.Use these labels (same spelling and small letters):
  - hardhat
  - vest
  - mask
  - boots
* Describe the contents of this JSON file in FULL details. 
* Find out the best total numbers of clusters

## Results of taring ResNet18 model with tiny imagenet dataset
 * ResNet18 model is trained with tiny imagenet dataset with 50 epochs and achieved the >50% val accuracy.
 * OneCycleLR sheduler is used:
   - min_lr = 0.5
   - max_lr = min_lr/5
   - Epochs = 24
   - Batch Size = 512
 * Achived expected validation accuracy of **50%**
 * transform used: Randomcrop-->flip-->Rotation

## Accuracy and Loss
![i](images/accu.png)

## Annotation of images for PPE detection.
* Downloaded abput 64 images each of people wearing hardhat, vest, mask and boots.
* Images are anootaed with bounding box for classes.
  - hardhat
  - vest
  - mask
  - boot
* tool used for annotation - [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) 
## Kmeans IOU for diffrent k values of clustering
![i2](images/kmeans_iou.png)
* For the above plot, k = 4 or k = 8 seems to be the best choice because for after 8, the curve becomes almost linear.

## Cluster plot for k=4 and k=8
![i3](images/cluster_plot_k4.png) ![i4](images/cluster_plot_k8.png)

## Anchor box for k=4 and k=8
![i5](images/anchor_bbox_k4.png) ![i6](images/anchor_bbox_k8.png)
