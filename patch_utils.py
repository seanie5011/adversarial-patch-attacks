import numpy as np
import torch

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
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'  # batch size must be 1; TODO: Fix this to allow bigger batch sizes
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
