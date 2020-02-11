# PyTorch
import torchvision
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

from torch_lr_finder import LRFinder

# Data science tools
import pandas as pd
import numpy as np

# Timing utility
from timeit import default_timer as timer
from config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, VALID_DIR, TEST_DIR

# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')


def get_image_transforms():
    image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=IMG_SIZE),
            transforms.RandomRotation(degrees=45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(45),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        # Validation does not use augmentation
        'val':
        transforms.Compose([
            transforms.CenterCrop(size=IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Test does not use augmentation
        'test':
        transforms.Compose([
            transforms.CenterCrop(size=IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return image_transforms

def get_dataloaders():

    image_transforms = get_image_transforms()
    # Datasets from each folder
    data = {
        'train':
        datasets.ImageFolder(root=TRAIN_DIR, transform=image_transforms['train']),
        'val':
        datasets.ImageFolder(root=VALID_DIR, transform=image_transforms['val']),
        'test':
        datasets.ImageFolder(root=TEST_DIR, transform=image_transforms['test'])
    }

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(data['val'], batch_size=BATCH_SIZE, shuffle=True),
        'test': DataLoader(data['test'], batch_size=BATCH_SIZE, shuffle=False)
    }
    return dataloaders

def get_model(train_on_gpu, multi_gpu):
    model = models.resnet50(pretrained=True)
    n_classes = 101
    n_inputs = model.fc.in_features

    # freeze layers in model to stop training for first cycle
    for param in model.parameters():
        param.requires_grad = False

    # replacee pre-trained models last fully connected layer with classifier
    classifier = nn.Sequential(
        nn.Linear(n_inputs,IMG_SIZE),
        nn.LeakyReLU(),
        nn.Linear(IMG_SIZE,n_classes)
    )

    model.fc = classifier

    # move the model to the GPU
    if train_on_gpu:
        model = model.to('cuda')

    # if there are multiple GPU's initialize for parallel processing 
    if multi_gpu:
        model = nn.DataParallel(model)

    return model

def get_optimizer_scheduler_and_criterion(model):
    # we will be using negative log likelihood as the loss function
    criterion = nn.CrossEntropyLoss()

    # we will be using the Adam optimizer as our optimizer
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-1, momentum=0.9)

    # secify learning rate scheduler (if there is no further decrease in loss for next 5 epochs 
    # then lower the learning rate by 0.1)
    # the model utilizes a cyclical training cycle while training
    # LR is found by utlizing the find Learning Rate Util
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=1e-1, base_lr=1e-3, mode='triangular')

    return criterion, optimizer, scheduler

def train_model(model,
                criterion,
                optimizer,
                scheduler,
                train_loader,
                valid_loader,
                save_file_name,
                checkpoint_file_name,
                early_stopping_patience=100,
                overfit_patience=15,
                n_epochs=25,
                print_every=2,
                valid_every=2
               ):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """
    # early stopping initializaiton
    epochs_no_improve = 0
    epochs_overfit = 0
    valid_loss_min = np.Inf
    
    valid_max_acc = 0
    history = []
    
    # number of epochs already trained (if using loaded in model weights)
    try:
        print("Model has been trained for: {} epochs.\n".format(model.epochs))
    except:
        model.epochs = 0
        print("Starting training from scratch.\n")
        
    overall_start = timer()
    
    #Main loop
    for epoch in range(n_epochs):
        
        #keep track of training and validation loss of each epoch
        train_loss = 0.0
        valid_loss = 0.0
        
        train_acc = 0
        valid_acc = 0
        
        #set to training
        model.train()
        start = timer()
        
        # training loop
        for ii, (data, target) in enumerate(train_loader):
            #tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
                
            # clear gradients
            optimizer.zero_grad()
            #predicted outpouts are log probabilities
            output = model(data)
            
            # loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()
            
            # update the parameters
            optimizer.step()
            
            # track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)
            
            # calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)
            
            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')
        # after training loop ends
        else:
            model.epochs += 1
            
            if model.epochs > 1 and (model.epochs % valid_every == 0): 
                # don't need to keep track of gradients
                with torch.no_grad():
                    # set to evaluation mode
                    model.eval()

                    #validation loop
                    for data, target in valid_loader:
                        #tensors to gpu
                        if train_on_gpu:
                            data, target = data.cuda(), target.cuda()

                        # Forward pass
                        output = model(data)

                        # validation loss 
                        loss = criterion(output, target)
                        # multiply average loss times the number of examples in batch
                        valid_loss += loss.item() * data.size(0)

                        # calculate validation accuracy
                        _, pred = torch.max(output, dim=1)
                        correct_tensor = pred.eq(target.data.view_as(pred))
                        accuracy = torch.mean(
                            correct_tensor.type(torch.FloatTensor))
                        # multiply average accuracy times the number of examples
                        valid_acc += accuracy.item() * data.size(0)

                    # calculate average losses
                    train_loss = train_loss / (len(train_loader.dataset))
                    valid_loss = valid_loss / (len(valid_loader.dataset))

                    # calculate average accuracy
                    train_acc = train_acc / (len(train_loader.dataset))
                    valid_acc = valid_acc / (len(valid_loader.dataset))

                    # learning rate scheduler step
                    scheduler.step(valid_loss)

                    history.append([train_loss, valid_loss, train_acc, valid_acc, model.epochs])

                    # Print training and validation results
                    if (model.epochs + 1) % valid_every == 0:
                        print(
                            f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                        )
                        print(
                            f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                        )

                    # save the model if validation loss decreases
                    if valid_loss < valid_loss_min:
                        print("Valid loss decreased ({:.6f} --> {:.6f}). Saving model...".format(valid_loss_min, valid_loss))

                        # save model
                        torch.save(model.state_dict(), save_file_name)

                        checkpoint = {
                            "model": model,
                            "criterion": criterion,
                            "epochs": model.epochs,
                            "optimizer_state": optimizer.state_dict(),
                            "model_state": model.state_dict(),
                            "valid_loss_min": valid_loss
                        }
                        torch.save(checkpoint, checkpoint_file_name)

                        # track improvements
                        epochs_no_improve = 0
                        epochs_overfit = 0
                        valid_loss_min = valid_loss
                        valid_best_acc = valid_acc
                        best_epoch = epoch

                    # otherwise increment count of epochs with no improvement
                    elif train_loss < valid_loss:
                        epochs_overfit += 1
                        if epochs_overfit >= overfit_patience:
                            print(f'\n Valid loss has increased larger than training loss for {epochs_overfit} epochs')
                            print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                            )
                            # load the best state dict
                            model.load_state_dict(torch.load(save_file_name))
                            # attach the optimizer
                            model.optimizer = optimizer

                            # format history
                            history = pd.DataFrame(
                                    history,
                                    columns=[
                                        'train_loss', 'valid_loss', 'train_acc',
                                        'valid_acc', 'epochs'
                                    ])
                            return model, history

                    else:
                        epochs_no_improve += 1
                        #trigger early stopping
                        # this should be not going bad
                        if (epochs_no_improve >= early_stopping_patience):
                            print(
                                f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                            )
                            total_time = timer() - overall_start
                            print(
                                f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                            )

                            # load the best state dict
                            model.load_state_dict(torch.load(save_file_name))
                            # attach the optimizer
                            model.optimizer = optimizer

                            # format history
                            history = pd.DataFrame(
                                    history,
                                    columns=[
                                        'train_loss', 'valid_loss', 'train_acc',
                                        'valid_acc', 'epochs'
                                    ])
                            return model, history
                        
    model.optimizer = optimizer
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (model.epochs):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history

