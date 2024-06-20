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
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import *
from datasets import HymenopteraDataset

# initialize the patch as random set of pixels
def patch_initialization(image_size=(3, 224, 224), noise_percentage=0.03):
    mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
    patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

# generate the mask and apply the patch
def mask_generation(patch=None, image_size=(3, 224, 224)):
    applied_patch = np.zeros(image_size)

    # patch rotation
    rotation_angle = np.random.choice(4)
    for i in range(patch.shape[0]):
        patch[i] = np.rot90(patch[i], rotation_angle)  # The actual rotation angle is rotation_angle * 90

    # patch location
    x_location, y_location = np.random.randint(low=0, high=image_size[1]-patch.shape[1]), np.random.randint(low=0, high=image_size[2]-patch.shape[2])
    for i in range(patch.shape[0]):
        applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    
    # copy before making everything that is not black (mask) white (rest of image)
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location, y_location

# test the patch on dataset
def test_patch(device, target, patch, dataloader, model):
    model.eval()
    total, success = 0, 0
    for (image, label) in dataloader:
        # get models prediction
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        _, predicted = torch.max(output.data, 1)

        # if model predicted something that isnt our desired label
        if predicted[0].data.cpu().numpy() != target:
            total += 1  # we are interested only in cases where the model originally predicted something that isn't our model, but then we make it do so with the patch

            # get mask and apply to image
            applied_patch, mask, _, _ = mask_generation(patch, image_size=(3, 224, 224))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.to(device)

            # get new output, success if model now picks our target
            output = model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                success += 1
    
    return success / total

# According to Brown et al., one image is attacked each time
def patch_attack(device, image, applied_patch, mask, target, probability_threshold, model, lr, max_iteration):
    # setup model, patch, and mask
    model.eval()
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0
    # create first image by using formula detailed in Brown et al.
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    # do not stop until we are over the probability threshold or reach max count
    while target_probability < probability_threshold and count < max_iteration:
        count += 1

        # optimize the patch by optimizing whole perturbed image on loss function
        perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        per_image = per_image.to(device)
        output = model(per_image)  # model output used for loss
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)  # update whole image that contains patch (remember when we apply the patch using mask only part we care about is used)
        applied_patch = torch.clamp(applied_patch, min=-3, max=3)

        # test the patch and update target probability using softmax
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        perturbated_image = perturbated_image.to(device)
        output = model(perturbated_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]

    # return as numpy arrays
    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/hymenoptera/', help='directory of the dataset, assumed to contain folders train/ and val/, where each individual classes images are in folders of <label>/')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size for the dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--model_dir', type=str, default='models/hymenoptera_resnet18_weights.pth', help='directory where the model weights are stored')
    parser.add_argument('--noise_percentage', type=float, default=0.1, help='percentage of the patch size compared with the image size')
    parser.add_argument('--probability_threshold', type=float, default=0.9, help='minimum target probability')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--max_iteration', type=int, default=1000, help='max iterations')
    parser.add_argument('--target', type=int, default=1, help='target label')
    parser.add_argument('--epochs', type=int, default=2, help='total number of epochs')
    parser.add_argument('--log_path', type=str, default='logs/patch_attack_log.csv', help='filepath to store the log')
    parser.add_argument('--train_dir', type=str, default='training/', help='directory to store the training images/animations/data')
    parser.add_argument('--fps', type=int, default=10, help='number of frames per second for the animation')
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

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)  # batch size must be 1 for images to work
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)

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
    with open(args.log_path, 'w', newline='') as f:  # use '' for newline as we are opening a file
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['epoch', 'train_success', 'test_success'])

    # generate the patch
    print('starting training...')
    since = time.time()
    best_patch_epoch, best_patch_success_rate = 0, 0
    for epoch in range(args.epochs):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{args.epochs}')
        animation_arrs = []
        # train using models output on train dataset
        for (image, label) in tqdm(train_loader, desc='Images loaded'):
            # get models prediction
            assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)

            # if model predicted something that isnt our desired label
            if predicted[0].data.cpu().numpy() != args.target:
                # applied_patch is a whole image with a patch
                # which we pass to the patch_attack with the mask to only use actual patch
                # then when we get it back after all iterations we use location to single out only patch
                applied_patch, mask, x_location, y_location = mask_generation(patch, image_size=(3, args.crop_size, args.crop_size))
                perturbated_image, applied_patch = patch_attack(device, image, applied_patch, mask, args.target, args.probability_threshold, model, args.lr, args.max_iteration)
                perturbated_image = torch.from_numpy(perturbated_image).to(device)
                patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]

                # save patch to animation
                animation_arrs.append(patch)

        # create animation for this epoch
        print("animating...")
        image_animation(animation_arrs, args.fps, args.train_dir + str(epoch) + "_anim.mp4")
        print("animation complete!")
        
        # fix before displaying and saving checkpoint
        plt.imshow(image_fix(patch))
        plt.savefig(args.train_dir + str(epoch) + '_patch.png')
        with open(args.train_dir + str(epoch) + '_data.npy', 'wb') as f:  # saving as a binary numpy file
            np.save(f, patch)

        # show results of this epoch
        print("performing tests...")
        train_success_rate = test_patch(device, args.target, patch, train_loader, model)
        print(f'Patch attack success rate on trainset: {100 * train_success_rate:.3f}%')
        test_success_rate = test_patch(device, args.target, patch, test_loader, model)
        print(f'Patch attack success rate on testset: {100 * test_success_rate:.3f}%')
        print("tests complete!")

        # record the stats for this epoch
        with open(args.log_path, 'a', newline='') as f:  # use '' for newline as we are opening a file
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([epoch, train_success_rate, test_success_rate])

        # update if this is better
        if test_success_rate > best_patch_success_rate:
            print("new best!")
            best_patch_success_rate = test_success_rate
            best_patch_epoch = epoch

        # load the stats and generate the plot
        log_generation(args.log_path, args.train_dir)

    # finished, final output
    time_elapsed = time.time() - since
    print('training finished!')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.')
    print(f'The best patch is found at epoch {best_patch_epoch} with success rate {100 * best_patch_success_rate}% on the test dataset.')

    # TODO: proper test with holdout set
    # testing on most recent patch with some random from test set
    model.eval()
    image, label, answer = None, args.target, args.target
    while label == args.target or answer == args.target:  # only want to test on something that isnt our target (and making sure the model predited correctly when so)
        print("trying image...")
        image, label = next(iter(test_loader))

        # get models prediction
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        answer = predicted[0].data.cpu().numpy()

        # if model predicted something that isnt our desired label
        if answer != args.target:
            # get mask and apply to image
            applied_patch, mask, _, _ = mask_generation(patch, image_size=(3, 224, 224))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.to(device)

            # get new output
            output = model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            answer = predicted[0].data.cpu().numpy()
            if answer == args.target:
                print("success!")
                break
            else:
                print("fail...")
                answer = args.target  # will make loop restart
        else:
            print("bad image...")

    plt.imshow(image_fix(perturbated_image.squeeze().cpu().numpy()))
    plt.savefig(args.train_dir + 'final_test.png')
    plt.show()
