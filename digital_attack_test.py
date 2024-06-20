import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import models

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import *
from datasets import HymenopteraDataset
from digital_attack import mask_generation

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/hymenoptera/', help='directory of the dataset, assumed to contain folders train/ and val/, where each individual classes images are in folders of <label>/')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size for the dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--model_dir', type=str, default='models/hymenoptera_resnet18_weights.pth', help='directory where the model weights are stored')
    parser.add_argument('--patch_path', type=str, default='training/run3/0_data.npy', help='path to the desired patch')
    parser.add_argument('--target', type=int, default=1, help='target label')
    parser.add_argument('--num_images', type=int, default=9, help='number of images to display')
    parser.add_argument('--save_path', type=str, default='training/final_test.png', help='path to where the final output should be stored')
    args = parser.parse_args()

    # set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'cpu cores available: {os.cpu_count()}')
    if device.type == 'cuda':
        print(f'using: {torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}')

    # load the datasets
    test_dataset = HymenopteraDataset(args.data_dir, args.crop_size, True)
    print(f'Test set size: {len(test_dataset)} images')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)

    # load the model
    model = models.resnet18()  # we do not specify ``weights``, i.e. create untrained model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(test_dataset.classes))  # create new classification layer for our desired output
    try:
        print('loading existing model...')
        model.load_state_dict(torch.load(args.model_dir))
        print('loaded existing model!')
    except:
        print('ERROR loading model path, make sure it points to an existing weights file. Using random weights.')
    model_ft = model.to(device)
    model.eval()

    # load the patch data
    with open(args.patch_path, 'rb') as f:
        patch = np.load(f)

    # perform test using patch and data
    model.eval()
    count = 0
    fig = plt.figure()
    for (image, label) in tqdm(test_loader, total=args.num_images, desc='Images loaded'):
        if count >= args.num_images:  # do not do more than specified
            break

        # get models prediction
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        answer = predicted[0].data.cpu().numpy()

        # get mask and apply to image
        applied_patch, mask, _, _ = mask_generation(patch, image_size=(3, 224, 224))
        applied_patch = torch.from_numpy(applied_patch)
        mask = torch.from_numpy(mask)
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = perturbated_image.to(device)

        # get new output
        output = model(perturbated_image)
        _, predicted = torch.max(output.data, 1)
        perturbed_answer = predicted[0].data.cpu().numpy()

        count += 1

        # create plot by placing each cell individually
        ax = plt.subplot(args.num_images//3, 3, count)
        ax.axis('off')
        ax.set_title(f'{test_dataset.classes[answer]} -> {test_dataset.classes[perturbed_answer]}')
        plt.imshow(image_fix(perturbated_image.squeeze().cpu().numpy()))
        plt.pause(0.001)

    plt.savefig(args.save_path)
    plt.show()
