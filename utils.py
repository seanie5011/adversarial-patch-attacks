import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch

# test the model on entire dataset
def test(model, dataloader, device):
    # set model to eval, pass through every object in dataloader
    model.eval()
    correct, total, loss = 0, 0, 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            # get output from model
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # update number of correct
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

    return correct / total

# load the log and generate the training picture
def log_generation(log_dir, train_dir):
    # load the stats in the log
    epochs, train_rate, test_rate = [], [], []
    with open(log_dir, 'r', newline='') as f:  # use '' for newline as we are opening a file
        reader = csv.reader(f, lineterminator='\n')
        flag = 0
        for row in reader:
            # dont load header
            if flag == 0:
                flag += 1
                continue
            # make sure we have a valid row
            elif len(row) == 3:
                epochs.append(int(row[0]))
                train_rate.append(float(row[1]))
                test_rate.append(float(row[2]))

    # generate the training plot
    plt.figure(num=0)
    plt.plot(epochs, test_rate, label='test_success_rate', linewidth=2, color='r')
    plt.plot(epochs, train_rate, label='train_success_rate', linewidth=2, color='b')
    plt.xlabel("epoch")
    plt.ylabel("success rate")
    plt.xlim(-1, max(epochs) + 1)
    plt.ylim(0, 1.0)
    plt.title("patch attack success rate")
    plt.legend()
    plt.savefig(train_dir + "patch_attack_success_rate.png")
    plt.close(0)

def image_fix(arr, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    Clips, transposes and normalizes images to be displayed.
    '''
    return np.clip(np.transpose(arr, (1, 2, 0)) * std + mean, 0, 1)  # clip to visible rgb space

def image_animation(arrs, FPS, save_dir):
    '''
    Animates an image over time, according to numpy array arrs (containing each frame of animation).
    '''
    # set the figure, settings to remove all matplotlib extras
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1, 1)
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)

    # initialize
    a = image_fix(arrs[0])
    im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)

    # set to next part of image
    def animate_func(i):
        a = image_fix(arrs[i])
        im.set_array(a)
        return [im]

    # create and save animation
    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=len(arrs),
        interval=1000 / FPS,
    )

    anim.save(save_dir, fps=FPS, extra_args=['-vcodec', 'libx264'])
