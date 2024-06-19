import time
import argparse
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import models

import numpy as np

from patch_utils import *
from utils import *
from datasets import HymenopteraDataset

# TODO: review
# Patch attack via optimization
# According to reference [1], one image is attacked each time
# Assert: applied patch should be a numpy
# Return the final perturbated picture and the applied patch. Their types are both numpy
def patch_attack(device, image, applied_patch, mask, target, probability_threshold, model, lr=1, max_iteration=100):
    model.eval()
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    while target_probability < probability_threshold and count < max_iteration:
        count += 1
        # Optimize the patch
        perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        per_image = per_image.to(device)
        output = model(per_image)
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=-3, max=3)
        # Test the patch
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        perturbated_image = perturbated_image.to(device)
        output = model(perturbated_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/hymenoptera/', help='directory of the dataset, assumed to contain folders train/ and val/, where each individual classes images are in folders of <label>/')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size for the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--model_dir', type=str, default='models/hymenoptera_resnet18_weights.pth', help='directory where the model weights are stored')
    parser.add_argument('--noise_percentage', type=float, default=0.1, help='percentage of the patch size compared with the image size')
    parser.add_argument('--probability_threshold', type=float, default=0.9, help='minimum target probability')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--max_iteration', type=int, default=1000, help='max iterations')
    parser.add_argument('--target', type=int, default=0, help='target label')
    parser.add_argument('--epochs', type=int, default=2, help='total number of epochs')
    parser.add_argument('--log_dir', type=str, default='logs/patch_attack_log.csv', help='directory to store the log')
    args = parser.parse_args()

    # set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'cpu cores available: {os.cpu_count()}')
    if device.type == 'cuda':
        print(f'using: {torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}')

    # load the datasets
    train_dataset = HymenopteraDataset(args.data_dir, args.crop_size, False)
    print(f'Training set size: {len(train_dataset)} images')
    test_dataset = HymenopteraDataset(args.data_dir, args.crop_size, True)
    print(f'Test set size: {len(test_dataset)} images')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # load the model
    model = models.resnet18()  # we do not specify ``weights``, i.e. create untrained model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # create new classification layer for our desired output
    try:
        print('loading existing model...')
        model.load_state_dict(torch.load(args.model_dir))
        print('loaded existing model!')
    except:
        print('ERROR loading model path, make sure it points to an existing weights file. Using random weights.')
    model_ft = model.to(device)
    model.eval()

    # test the accuracy of model on clean trainset and testset
    print('starting test...')
    trainset_acc = test(model, train_loader, device)
    print('finished test!')
    print('starting test...')
    test_acc = test(model, test_loader, device)
    print('finished test!')
    print(f'Accuracy of the model on clean trainset and testset is {100 * trainset_acc:.3f}% and {100 * test_acc:.3f}% respectively.')

    # initialize the patch
    patch = patch_initialization(image_size=(3, args.crop_size, args.crop_size), noise_percentage=args.noise_percentage)
    print(f'The shape of the patch is {patch.shape}.')

    # start logging
    with open(args.log_dir, 'w', newline='') as f:  # use '' for newline as we are opening a file
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['epoch', 'train_success', 'test_success'])

    # generate the patch
    print('starting training...')
    since = time.time()
    best_patch_epoch, best_patch_success_rate = 0, 0
    for epoch in range(args.epochs):
        # train using models output on train dataset
        for (image, label) in train_loader:
            # get models prediction
            assert image.shape[0] == 1, 'Only one picture should be loaded each time.'  # batch size must be 1; TODO: Fix this to allow bigger batch sizes
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)

            # if model predicted something that isnt our desired label
            if predicted[0].data.cpu().numpy() != args.target:
                applied_patch, mask, x_location, y_location = mask_generation(patch, image_size=(3, args.crop_size, args.crop_size))
                perturbated_image, applied_patch = patch_attack(device, image, applied_patch, mask, args.target, args.probability_threshold, model, args.lr, args.max_iteration)
                perturbated_image = torch.from_numpy(perturbated_image).to(device)
                patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
        
        # fix before displaying and saving checkpoint
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
        plt.savefig('training_pictures/' + str(epoch) + '_patch.png')

        # show results of this epoch
        train_success_rate = test_patch(device, args.target, patch, test_loader, model)
        print(f'Epoch:{epoch} Patch attack success rate on trainset: {100 * train_success_rate:.3f}%')
        test_success_rate = test_patch(device, args.target, patch, test_loader, model)
        print(f'Epoch:{epoch} Patch attack success rate on testset: {100 * test_success_rate:.3f}%')

        # record the stats for this epoch
        with open(args.log_dir, 'a', newline='') as f:  # use '' for newline as we are opening a file
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([epoch, train_success_rate, test_success_rate])

        # update if this is better
        if test_success_rate > best_patch_success_rate:
            best_patch_success_rate = test_success_rate
            best_patch_epoch = epoch
            plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))  # clip to visible rgb space
            plt.savefig('training_pictures/best_patch.png')

        # load the stats and generate the plot
        log_generation(args.log_dir)

    # finished, final output
    time_elapsed = time.time() - since
    print('training finished!')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.')
    print(f'The best patch is found at epoch {best_patch_epoch} with success rate {100 * best_patch_success_rate}% on the test dataset.')
