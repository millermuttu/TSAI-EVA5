import torchvision.datasets as datasets
from torchvision.datasets import VisionDataset
import os
from torch.utils.data import DataLoader, Dataset
from .transformation import *

class AlbumentationsDataset(Dataset):
    def __init__(self, data, targets, transform, class_to_idx):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.targets[idx]
        image = self.data[idx]
        augmented = self.transform(**{'image':image}) #unpacking the dict valus using **
        image = augmented['image']
        return image, label


def data_loader(data, batch_size, num_workers, cuda, shuffle):
    """Create data loader
    Args:
        data: Dataset.
        batch_size: int
            Number of images to considered in each batch.
        num_workers: int
            How many subprocesses to use for data loading.
        cuda: bool
            True is GPU is available.
        shuffle: bool

    Returns:
        DataLoader instance.
    """

    loader_args = {
        'shuffle': shuffle,
        'batch_size': batch_size
    }

    # If GPU exists
    if cuda:
        loader_args['num_workers'] = num_workers
        loader_args['pin_memory'] = True

    return DataLoader(data, **loader_args)


def pytorch_datasets(train=True, download=True, transforms=None, augmentations=None, dataset_name="data"):
    """
    Returns a dataset from torchvision library.
    Downloads the dataset in the folder sapecified by dataset_name variable

    Args
    ----
    train:bool
    download:bool
    transforms:list of transforms
        By default ToTensor is applied.
    dataset_name:str

    Returns
    -------
    data: torch Dataset
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_name)

    transforms = transforma_albumentation(transforms, augmentations)

    if dataset_name.lower() in ['cifar10', 'mnist']:
        if dataset_name.lower() == 'cifar10':
            data = datasets.CIFAR10(
                data_path, train=train, download=download, transform=transforms
            )
        elif dataset_name.lower() == 'mnist':
            data = datasets.MNIST(
                data_path, train=train, download=download, transform=transforms
            )

        return AlbumentationsDataset(data.data, data.targets, transforms, data.class_to_idx)
    else:
        print("Enter a correct dataset name")


