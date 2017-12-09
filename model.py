from load_data import transform_data

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import time

use_gpu = torch.cuda.is_available()

def train_new_model():
    pre_model = torchvision.models.vgg19_bn(pretrained=True)
    for param in pre_model.parameters():    #----> 1
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    layers = list(pre_model.classifier.children())[:-1]
    pre_last_layer = list(pre_model.classifier.children())[-1]
    new_last_layer = nn.Linear(pre_last_layer.in_features, 2)
    layers += [new_last_layer]
    
    new_classifier = nn.Sequential(*layers)
    pre_model.classifier = new_classifier

    criterion = nn.CrossEntropyLoss()

    learning_rate = [0.001]
    best_acc = 0.0
    best_model = None
    # Observe that all parameters are being optimized
    for lr in learning_rate:
        optimizer = optim.SGD(new_last_layer.parameters(), lr=lr, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        new_model, val_acc = train_model(pre_model, criterion, optimizer,
                             exp_lr_scheduler, num_epochs=5)
        if val_acc > best_acc:
            best_model = new_model
            best_acc = val_acc

    return best_model

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                print inputs.size()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc

if __name__ == "__main__":

    image_datasets, dataloaders = transform_data()
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # get some random training images
    # train_loader, val_loader = transform_data_deprecate()
    # dataloaders = {}
    # dataloaders['train'] = train_loader
    # dataloaders['val'] = val_loader

    # dataset_sizes = {}
    # dataset_sizes['train'] = 6
    # dataset_sizes['val'] = 4

    # print "dataset = ", dataset_sizes

    new_model = train_new_model()
