import torch

def image_normalization(train):
    train_data = train.train_data
    train_data = train.transform(train_data.numpy())

    mean = torch.mean(train_data)
    std = torch.std(train_data)
    return mean,std

def image_normalization_3(train):
    train_data = train.data
    # train_data = train.transform(train_data)

    mean = train_data.mean(axis=(0,1,2))/255
    std = train_data.std(axis=(0,1,2))/255
    return mean,std


