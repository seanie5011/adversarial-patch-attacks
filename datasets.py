import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
import torchvision.transforms.functional

from PIL import Image
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt

class ImagenetteDataset(object):
    def __init__(self, data_dir, crop_size=320, validation=False):
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
        self.center_resize = torchvision.transforms.CenterCrop(self.crop_size)  # validation
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
        else: 
            image = self.center_resize(image)
        
        image = torchvision.transforms.functional.to_tensor(image)
        if image.shape[0] == 1: image = image.expand(3, self.crop_size, self.crop_size)
        image = self.normalize(image)
        
        return image, label

    def __len__(self):
        return len(self.images)
    
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # IMAGENETTE TESTS

    # import datasets and display examples
    train_dataset = ImagenetteDataset("datasets/imagenette2-320/", 320, False)
    print(f"Training set size: {len(train_dataset)} images")
    test_dataset = ImagenetteDataset("datasets/imagenette2-320/", 320, True)
    print(f"Validation set size: {len(test_dataset)} images")

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))

    for y in range(3):
        for x in range(3):
            sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
            img, label = train_dataset[sample_idx]
            axes[y, x].imshow(img.numpy().transpose(1,2,0))
            axes[y, x].set_axis_off()
            axes[y, x].set_title(train_dataset.labels_map[label])

    # create dataloaders and display
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=2)

    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = int(train_labels[0])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
    ax.imshow(img.numpy().transpose(1,2,0))
    ax.set_axis_off()
    ax.set_title(train_dataset.labels_map[label])

    # Load the model
    print("importing resnet model...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    print("model imported!")
    with torch.no_grad():
        output = model(torch.unsqueeze(train_features[0], 0))
        print(torch.max(output.data, 1))

    # correct, total, loss = 0, 0, 0
    # with torch.no_grad():
    #     for (images, labels) in tqdm(test_loader):
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.shape[0]
    #         correct += (predicted == labels).sum().item()
    # print(f"score: {correct / total}")

    # will show everything at end
    plt.show()

# TODO: Get imagenet dataset itself, and use this instead, dont have to remove imagenette code, but cant use for resnet50 (see that guys code)
# https://www.kaggle.com/c/imagenet-object-localization-challenge/data
# https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html
