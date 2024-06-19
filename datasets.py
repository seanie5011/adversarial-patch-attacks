import os
from pathlib import Path
import math

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.functional

from PIL import Image

import matplotlib.pyplot as plt

class ImagenetteDataset(object):
    def __init__(self, data_dir, crop_size=224, validation=False):
        self.crop_size = crop_size
        self.validation = validation
        
        self.folder = Path(data_dir + '/train') if not self.validation else Path(data_dir + '/val')
        self.classes = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
                        'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
        self.labels_map = {
            0: 'tench', 
            1: 'english springer', 
            2: 'cassette player', 
            3: 'chain saw', 
            4: 'church', 
            5: 'French horn', 
            6: 'garbage truck', 
            7: 'gas pump', 
            8: 'golf ball', 
            9: 'parachute',
         }

        # collect image paths
        self.images = []
        for cls in self.classes:
            cls_images = list(self.folder.glob(cls + '/*.JPEG'))
            self.images.extend(cls_images)
        
        self.random_resize = torchvision.transforms.RandomResizedCrop(self.crop_size)  # training
        self.random_flip = torchvision.transforms.RandomHorizontalFlip()
        self.regular_resize = torchvision.transforms.Resize(2**(math.ceil(math.log(self.crop_size, 2))))  # validation, scale up to closest power of 2
        self.center_crop = torchvision.transforms.CenterCrop(self.crop_size)
        # normalize as human eye is not equally sensitive to colors (but ML models should)
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # used in most datasets (specific)
        
    def __getitem__(self, index):
        # get the image and corresponing label
        image_fname = self.images[index]
        image = Image.open(image_fname)
        label = image_fname.parent.stem
        label = self.classes.index(label)

        # perform transforms
        if not self.validation: 
            image = self.random_resize(image)
            image = self.random_flip(image)
        else: 
            image = self.regular_resize(image)
            image = self.center_crop(image)
                
        image = torchvision.transforms.functional.to_tensor(image)
        if image.shape[0] == 1: image = image.expand(3, self.crop_size, self.crop_size)
        image = self.normalize(image)
        
        return image, label

    def __len__(self):
        return len(self.images)
    
class HymenopteraDataset(object):
    def __init__(self, data_dir, crop_size=224, validation=False):
        self.crop_size = crop_size
        self.validation = validation
        self.folder = Path(data_dir + '/train') if not self.validation else Path(data_dir + '/val')

        # create transforms for each
        data_transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(self.crop_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.Resize(2**(math.ceil(math.log(self.crop_size, 2)))),
                torchvision.transforms.CenterCrop(self.crop_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # create from imagefolder, so slightly different functionality to Imagenette
        self.dataset = torchvision.datasets.ImageFolder(self.folder, data_transforms['train' if not self.validation else 'val'])
        self.classes = self.dataset.classes
        
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    
if __name__ == "__main__":
    # set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'cpu cores available: {os.cpu_count()}')
    if device.type == 'cuda':
        print(f'using: {torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}')

    # IMAGENETTE TESTS
    # print("IMAGENETTE TESTS")

    # # import datasets and display examples
    # train_dataset = ImagenetteDataset("datasets/imagenette2-320/", 320, False)
    # print(f"Training set size: {len(train_dataset)} images")
    # test_dataset = ImagenetteDataset("datasets/imagenette2-320/", 320, True)
    # print(f"Validation set size: {len(test_dataset)} images")

    # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))

    # for y in range(3):
    #     for x in range(3):
    #         sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    #         img, label = train_dataset[sample_idx]
    #         axes[y, x].imshow(img.numpy().transpose(1,2,0))
    #         axes[y, x].set_axis_off()
    #         axes[y, x].set_title(train_dataset.labels_map[label])

    # # create dataloaders and display
    # train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=4)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # train_features, train_labels = next(iter(train_loader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = int(train_labels[0])
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
    # ax.imshow(img.numpy().transpose(1,2,0))
    # ax.set_axis_off()
    # ax.set_title(train_dataset.labels_map[label])

    # HYMENOPTERA TESTS
    print("HYMENOPTERA TESTS")

    # import datasets and display examples
    train_dataset = HymenopteraDataset("datasets/hymenoptera_data/", 224, False)
    print(f"Training set size: {len(train_dataset)} images")
    test_dataset = HymenopteraDataset("datasets/hymenoptera_data/", 224, True)
    print(f"Validation set size: {len(test_dataset)} images")

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))

    for y in range(3):
        for x in range(3):
            sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
            img, label = train_dataset[sample_idx]
            axes[y, x].imshow(img.numpy().transpose(1,2,0))
            axes[y, x].set_axis_off()
            axes[y, x].set_title(train_dataset.classes[label])

    # create dataloaders and display
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4)

    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = int(train_labels[0])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
    ax.imshow(img.numpy().transpose(1,2,0))
    ax.set_axis_off()
    ax.set_title(train_dataset.classes[label])

    # will show everything at end
    plt.show()
