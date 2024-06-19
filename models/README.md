# Models

# Hymenoptera Dataset

```
hymenoptera_resnet18_weights.pth
```

Trained via transfer learning using the [PyTorch implementation of Resnet18](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet18.html), starting with weights as specified by: `IMAGENET1K_V1`. A crop size of $224$ on the dataset, with a batch size of $4$ and $4$ workers used, with a learning rate of $0.001$ (decayed by $0.1$ every $7$ epochs) and a momentum of $0.9$. Trained for $25$ epochs, taking $25\mathrm{m} 53\mathrm{s}$ on an NVidia GeForce MX350. This reached a best validation accuracy of $94.1176%$.