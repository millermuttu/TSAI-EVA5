import torch
import dataset
import models
import utils
import numpy as np

cuda, device = utils.initialize_device(utils.SEED)
print(cuda,device)