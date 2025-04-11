#!/usr/bin/env python3

# Standard libraries
import sys
import os
import numpy as np
import tqdm
# Pytorch
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
# For visualization
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Expand the module search path to parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Load custom-written code
import utils
from visual import visual_plt
from eval.evaluate import test_acc
from models.classifier import Classifier
from data.manipulate import TransformedDataset


################## INITIAL SET-UP ##################

# Specify directories, and if needed create them
p_dir = "./store/plots"
d_dir = "./store/data"
if not os.path.isdir(p_dir):
    print("Creating directory: {}".format(p_dir))
    os.makedirs(p_dir)
if not os.path.isdir(d_dir):
    os.makedirs(d_dir)
    print("Creating directory: {}".format(d_dir))

# Open pdf for plotting
plot_name = "stability_gap_example"
full_plot_name = "{}/{}.pdf".format(p_dir, plot_name)
pp = visual_plt.open_pdf(full_plot_name)
figure_list = []



################## CREATE TASK SEQUENCE ##################

## Download the MNIST dataset
print("\n\n " +' LOAD DATA '.center(70, '*'))
MNIST_trainset = datasets.MNIST(root='data/', train=True, download=True,
                                transform=transforms.ToTensor())
MNIST_testset = datasets.MNIST(root='data/', train=False, download=True,
                               transform=transforms.ToTensor())
config = {'size': 28, 'channels': 1, 'classes': 10}

# Set for each task the amount of rotation to use
rotations = [0, 80, 160]

# Specify for each task the transformed train- and testset
n_tasks = len(rotations)
train_datasets = []
test_datasets = []
for rotation in rotations:
    train_datasets.append(TransformedDataset(
        MNIST_trainset, transform=transforms.RandomRotation(degrees=(rotation,rotation)),
    ))
    test_datasets.append(TransformedDataset(
        MNIST_testset, transform=transforms.RandomRotation(degrees=(rotation,rotation)),
    ))

# Visualize the different tasks
figure, axis = plt.subplots(1, n_tasks, figsize=(3*n_tasks, 4))
n_samples = 49
for task_id in range(len(train_datasets)):
    # Show [n_samples] examples from the training set for each task
    data_loader = torch.utils.data.DataLoader(train_datasets[task_id], batch_size=n_samples, shuffle=True)
    image_tensor, _ = next(iter(data_loader))
    image_grid = make_grid(image_tensor, nrow=int(np.sqrt(n_samples)), pad_value=1) # pad_value=0 would give black borders
    axis[task_id].imshow(np.transpose(image_grid.numpy(), (1,2,0)))
    axis[task_id].set_title("Task {}".format(task_id+1))
    axis[task_id].axis('off')
figure_list.append(figure)



################## SET UP THE MODEL ##################

print("\n\n " + ' DEFINE THE CLASSIFIER '.center(70, '*'))

# Specify the architectural layout of the network to use
fc_lay = 3        #--> number of fully-connected layers
fc_units = 400    #--> number of units in each hidden layer

# Define the model
model = Classifier(image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
                   fc_layers=fc_lay, fc_units=fc_units, fc_bn=False)

# Print some model info to screen
utils.print_model_info(model)



################## TRAINING AND EVALUATION ##################

print('\n\n' + ' TRAINING + CONTINUAL EVALUATION '.center(70, '*'))

# Define a function to train a model, while also evaluating its performance after each iteration
def train_and_evaluate(model, trainset, iters, lr, batch_size, testset,
                       test_size=512, performance=[]):
    '''Function to train a [model] on a given [dataset],
    while evaluating after each training iteration on [testset].'''

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    iters_left = 1
    progress_bar = tqdm.tqdm(range(1, iters+1))

    for _ in range(1, iters+1):
        optimizer.zero_grad()

        # Collect data from [trainset] and compute the loss
        iters_left -= 1
        if iters_left==0:
            data_loader = iter(torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                           shuffle=True, drop_last=True))
            iters_left = len(data_loader)
        x, y = next(data_loader)
        y_hat = model(x)
        loss = torch.nn.functional.cross_entropy(input=y_hat, target=y, reduction='mean')

        # Calculate test accuracy (in %)
        accuracy = 100*test_acc(model, testset, test_size=test_size, verbose=False, batch_size=512)
        performance.append(accuracy)

        # Take gradient step
        loss.backward()
        optimizer.step()
        progress_bar.set_description(
        '<CLASSIFIER> | training loss: {loss:.3} | test accuracy: {prec:.3}% |'
            .format(loss=loss.item(), prec=accuracy)
        )
        progress_bar.update(1)
    progress_bar.close()

# Specify the training parameters
iters = 500         #--> for how many iterations to train?
lr = 0.1            #--> learning rate
batch_size = 128    #--> size of mini-batches
test_size = 2000    #--> number of test samples to evaluate on after each iteration

# Define a list to keep track of the performance on task 1 after each iteration
performance_task1 = []

# Iterate through the contexts
for task_id in range(n_tasks):
    current_task = task_id+1

    # Concatenate the training data of all tasks so far
    joint_dataset = torch.utils.data.ConcatDataset(train_datasets[:current_task])

    # Determine the batch size to use
    batch_size_to_use = current_task*batch_size

    # Train
    print('Training after arrival of Task {}:'.format(current_task))
    train_and_evaluate(model, trainset=joint_dataset, iters=iters, lr=lr,
                      batch_size=batch_size_to_use, testset=test_datasets[0],
                      test_size=test_size, performance=performance_task1)



################## PLOTTING ##################

## Plot per-iteration performance curve
figure = visual_plt.plot_lines(
    [performance_task1], x_axes=list(range(n_tasks*iters)),
    line_names=['Incremental Joint'],
    title="Performance on Task 1 throughout 'Incremental Joint Training'",
    ylabel="Test Accuracy (%) on Task 1",
    xlabel="Total number of training iterations", figsize=(10,5),
    v_line=[iters*(i+1) for i in range(n_tasks-1)], v_label='Task switch', ylim=(70,100),
)
figure_list.append(figure)

## Finalize the pdf with the plots
# -add figures to pdf
for figure in figure_list:
    pp.savefig(figure)
# -close pdf
pp.close()
# -print name of generated plot on screen
print("\nGenerated plot: {}\n".format(full_plot_name))

