import torch

DEBUG = False
BATCH_SIZE = 128
EPOCHS = 15
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DOWNLOAD = True
NUM_WORKERS = 4
LEARNING_RATE = 0.01
MOMENTUM = 0.9
STEP_SIZE = 6
