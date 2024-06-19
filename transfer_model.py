# following: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import time
import os
import argparse
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import models

import numpy as np
import matplotlib.pyplot as plt

from datasets import HymenopteraDataset

def train_model(dataloaders, dataset_sizes, device, model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    # create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        # save first checkpoint
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)

        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            # each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # set model to training mode
                else:
                    model.eval()   # set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # stats
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                # step scheduler for LR
                if phase == 'train':
                    scheduler.step()

                # get stats for this epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # save checkpoint
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        # finished, display stats
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

    return model

def imshow(input, title=None):
    '''
    Displays image from tensor.
    '''
    input = input.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(dataloaders, class_names, model, num_images=6):
    '''
    Tests model on num_images images and visualizes using the custom imshow() function.
    '''
    was_training = model.training  # will set model back to train mode if it was training before visualization
    model.eval()  # evaluation mode
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        # get predictions for everything in validation set an display them using above imshow() function
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[predicted[j]]}')
                imshow(inputs.cpu().data[j])

                # set model to previous state
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        # set model to previous state
        model.train(mode=was_training)

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/hymenoptera/', help="directory of the dataset, assumed to contain folders train/ and val/, where each individual classes images are in folders of <label>/")
    parser.add_argument('--crop_size', type=int, default=224, help="crop size for the dataset")
    parser.add_argument('--train', type=bool, default=False, help="whether we are training the model from scratch (True) or simply loading a pre-existing (False)")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="number of workers")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--step_size', type=int, default=7, help="step size for stepLR")
    parser.add_argument('--gamma', type=float, default=0.1, help="gamma for stepLR")
    parser.add_argument('--epochs', type=int, default=25, help="total number of epochs")
    parser.add_argument('--model_dir', type=str, default='models/transfer_model_weights_temp.pth', help='directory where the model weights are stored, will be pulled from instead if --train is False')
    args = parser.parse_args()

    # set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'cpu cores available: {os.cpu_count()}')
    if device.type == 'cuda':
        print(f'using: {torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}')

    # get data
    print('loading data...')
    image_datasets = {x: HymenopteraDataset(args.data_dir, args.crop_size, False if x == 'train' else True) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print('data loaded!')

    # training from pretrained resnet18
    if args.train == True:
        model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))  # create new classification layer for our desired output
        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.step_size, gamma=args.gamma)

        print('starting training...')
        model_ft = train_model(dataloaders, dataset_sizes, device, model_ft, criterion, optimizer_ft, exp_lr_scheduler, args.epochs)
        print('finished training!')
        print('saving model...')
        torch.save(model_ft.state_dict(), args.model_dir)
        print('saved model!')
    # testing a pre-existing model
    else:
        model_ft = models.resnet18()  # we do not specify ``weights``, i.e. create untrained model
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))  # create new classification layer for our desired output
        try:
            print('loading existing model...')
            model_ft.load_state_dict(torch.load(args.model_dir))
            print('loaded existing model!')
        except:
            print('ERROR loading model path, make sure it points to an existing weights file. Using random weights.')
        model_ft = model_ft.to(device)

    # visualize and test the model
    print('visualizing...')
    visualize_model(dataloaders, class_names, model_ft)
    plt.show()
    print('visualized!')