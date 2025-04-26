#https://stackoverflow.com/questions/56435961/how-to-access-the-network-weights-while-using-pytorch-nn-sequential

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools








device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

# dataloader arguments
batch_size = 128
data_path='D:\\Neuro Sci\\snntorch_learn_nn_robust\\mnist_data'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Define a transform
transform = transforms.Compose([
			transforms.Resize((28, 28)),
			transforms.Grayscale(),
			transforms.ToTensor(),
			transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=False, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=False, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# neuron and simulation parameters
beta = 0.5
num_steps = 50

# Initialize Network
#yes there is a better way to do this, no I aparently don't know how to do it
dropout_p = 0.0
net = nn.Sequential(
	nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
	nn.MaxPool2d(2),
	nn.ReLU(),
	nn.Dropout(dropout_p),
	
	nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
	nn.MaxPool2d(2),
	nn.ReLU(),
	nn.Dropout(dropout_p),

	nn.Flatten(),

	nn.Linear(32 * 7 * 7, 128),
	nn.ReLU(),
	nn.Dropout(dropout_p),

	nn.Linear(128, 10),
	nn.ReLU()
	).to(device)

criterion = nn.CrossEntropyLoss()
num_epochs = 10
model_path = "current_best_trad_mnist_98\\121_trad_model_mnist.pth"
net.load_state_dict(torch.load(model_path))

# Outer training loop
for epoch in range(num_epochs):  # loop over the dataset multiple times
	
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
		correct = 0
		total = 0
		for data in test_loader:
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)

			# calculate outputs by running images through the network
			outputs = net(inputs)
			# the class with the highest energy is what we choose as prediction
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		print(epoch, 100 * correct / total)



