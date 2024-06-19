import csv
import matplotlib.pyplot as plt
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
def log_generation(log_dir):
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
    plt.savefig("training_pictures/patch_attack_success_rate.png")
    plt.close(0)
