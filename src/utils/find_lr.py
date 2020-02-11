# PyTorch
import torchvision
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

from torch_lr_finder import LRFinder
from utils.model import get_model, get_dataloaders

model = get_model()
dataloaders = get_dataloaders()

# we will be using negative log likelihood as the loss function
criterion = nn.CrossEntropyLoss()
# we will be using the SGD optimizer as our optimizer
optimizer = optim.SGD(model.fc.parameters(), lr=1e-4)
lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
lr_finder.range_test(dataloaders['train'], end_lr=1, num_iter=2500)
lr_finder.plot()
lr_finder.reset()