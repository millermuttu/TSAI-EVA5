from torchvision import transforms

def transformations(transformations=None, augmentations=None):
    """Create data transformations

    Args:
       transformations: List of torchvision transforms

    Returns:
        Transform object containing defined data transformations.
    """

    transforms_list = [
        # convert the data to torch.FloatTensor with values within the range [0.0 ,1.0]
        transforms.ToTensor()
    ]

    if transformations is not None:
        transforms_list = transforms_list + transformations

    if augmentations is not None:
        transforms_list = augmentations + transforms_list

    return transforms.Compose(transforms_list)