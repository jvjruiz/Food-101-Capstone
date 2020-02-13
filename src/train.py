# PyTorch
import torchvision
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

from torch_lr_finder import LRFinder

# Data science tools
import numpy as np

from utils.model import get_dataloaders, get_image_transforms, get_model, get_optimizer_scheduler_and_criterion, train_model
from utils.dataset import sort_images

# Initialize parameters

# Location of data
data_dir = '../data/'
train_dir = data_dir + 'train/'
valid_dir = data_dir + 'valid/'
test_dir = data_dir + 'test/'

save_file_name = 'resnet50-transfer11.pt'
checkpoint_file_name = 'resnet50-transfer11.pth'

# Change to fit hardware
batch_size = 32
image_size = 224

# Sort images into proper directories
sort_images()

# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

# Number of gpus
if train_on_gpu:
    gpu_count = cuda.device_count()
    print(f'{gpu_count} gpus detected.')
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False

# set random seeds
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

# get model, loss function, optimizer, and scheduler
model = get_model(train_on_gpu, multi_gpu)

criterion, optimizer, scheduler = get_optimizer_scheduler_and_criterion(model)

dataloaders = get_dataloaders()

# start training for first cycle of training
model, history = train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders['train'],
    dataloaders['val'],
    save_file_name=save_file_name,
    checkpoint_file_name=checkpoint_file_name,
    early_stopping_patience=50,
    overfit_patience=10,
    n_epochs=200,
    print_every=1
    )

# unfreeze layers of the model for second cycle of training
for param in model.parameters():
    param.requires_grad = True

model, history = train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders['train'],
    dataloaders['val'],
    save_file_name=save_file_name,
    checkpoint_file_name=checkpoint_file_name,
    early_stopping_patience=50,
    overfit_patience=10,
    n_epochs=200,
    print_every=1
    )