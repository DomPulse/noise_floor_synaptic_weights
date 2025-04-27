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

noise_levels = np.linspace(0.1, 1, 10)
print(noise_levels)
num_epochs = 50
model_path = "current_best_trad_mnist_98\\121_trad_model_mnist.pth"
net.load_state_dict(torch.load(model_path))

virgin_first_layer = net[0].weight.cpu().detach().numpy()
virgin_second_layer = net[4].weight.cpu().detach().numpy()
virgin_third_layer = net[9].weight.cpu().detach().numpy()
virgin_fourth_layer = net[12].weight.cpu().detach().numpy()

first_shape = (np.shape(virgin_first_layer))
second_shape = (np.shape(virgin_second_layer))
third_shape = (np.shape(virgin_third_layer))
fourth_shape = (np.shape(virgin_fourth_layer))

first_std = np.std(virgin_first_layer)
second_std = np.std(virgin_second_layer)
thrid_std = np.std(virgin_third_layer)
fourth_std = (np.std(virgin_fourth_layer))

plt.figure()
plt.hist(virgin_first_layer.flatten())
plt.figure()
plt.hist(virgin_second_layer.flatten())
plt.figure()
plt.hist(virgin_third_layer.flatten())
plt.figure()
plt.hist(virgin_fourth_layer.flatten())
plt.show()

# Outer training loop
mean_acc = np.zeros((num_epochs, len(noise_levels)))
for index, n in enumerate(noise_levels):
	for epoch in range(num_epochs):  # loop over the dataset multiple times
		with torch.no_grad():
			net.load_state_dict(torch.load(model_path))
			
			noise_1 = np.random.normal(0, first_std*n, first_shape)
			noise_2 = np.random.normal(0, second_std*n, second_shape)
			noise_3 = np.random.normal(0, thrid_std*n, third_shape)
			noise_4 = np.random.normal(0, fourth_std*n, fourth_shape)
			
			net[0].weight[:,:,:,:] = torch.from_numpy(np.add(virgin_first_layer, noise_1)[:,:,:,:])
			net[4].weight[:,:,:,:] = torch.from_numpy(np.add(virgin_second_layer, noise_2)[:,:,:,:])
			net[9].weight[:,:] = torch.from_numpy(np.add(virgin_third_layer, noise_3)[:,:])
			net[12].weight[:,:] = torch.from_numpy(np.add(virgin_fourth_layer, noise_4)[:,:])
			
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
	
			print(n, epoch, 100 * correct / total)
			mean_acc[epoch, index] += (100 * correct) / (total)

plt.plot(noise_levels, np.mean(mean_acc, axis = 0))
acc_std = np.std(mean_acc, axis = 0)
acc_sem = acc_std/np.sqrt(num_epochs)
plt.fill_between(noise_levels, np.mean(mean_acc, axis = 0) + acc_sem, np.mean(mean_acc, axis = 0) - acc_sem, alpha = 0.2)
plt.xlabel('standard deviation noise multiplier')
plt.ylabel('accuracy')
plt.title('trad model noise test')
plt.show()


