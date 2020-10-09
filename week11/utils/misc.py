from utils.config import *
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchsummary import summary
import torch.nn as nn



def get_model_summary(model, input_size):
    print(summary(model, input_size))


def cross_entropy_loss_fn():
    return nn.CrossEntropyLoss()


def nll_loss():
    return nn.NLLLoss()


def sgd_optimizer(model, lr=LEARNING_RATE, momentum=MOMENTUM, l2_factor=0):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_factor)


def StepLR_scheduler(optimizer, step_size=STEP_SIZE, gamma=0.1):
    return StepLR(optimizer, step_size=step_size, gamma=gamma)

def ReduseLR_onplateau(optimizer, patience = 2, verbose = False):
    return ReduceLROnPlateau(optimizer, patience=patience, verbose=verbose)

def set_seed(seed, cuda):
    """ Setting the seed makes the results reproducible. """
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def initialize_device(seed):
    """ checking the cuda avalibility and sedding based on it"""

    # Check CUDA availability
    cuda = torch.cuda.is_available()
    print('GPU Available?', cuda)

    # Initialize seed
    set_seed(seed, cuda)

    # Set device
    device = torch.device("cuda" if cuda else "cpu")

    return cuda, device








