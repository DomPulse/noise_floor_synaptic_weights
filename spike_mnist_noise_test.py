#https://stackoverflow.com/questions/56435961/how-to-access-the-network-weights-while-using-pytorch-nn-sequential

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

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

spike_grad = surrogate.fast_sigmoid(slope=25)
lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

#yes there is a better way to do this, no I aparently don't know how to do it
net = nn.Sequential(
	nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
	nn.MaxPool2d(2),
	snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
	nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
	nn.MaxPool2d(2),
	snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

	nn.Flatten(),

	nn.Linear(32 * 7 * 7, 128),
	snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

	nn.Linear(128, 10),
	snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
	).to(device)


def forward_pass(net, num_steps, data):
	mem_rec = []
	spk_rec = []
	utils.reset(net)	# resets hidden states for all LIF neurons in net

	for step in range(num_steps):
		spk_out, mem_out = net(data)
		spk_rec.append(spk_out)
		mem_rec.append(mem_out)

	return torch.stack(spk_rec), torch.stack(mem_rec)

noise_levels = np.linspace(0.1, 1, 10)
print(noise_levels)
num_epochs = 50
model_path = "current_best_spike_mnist_98\\2_spike_model_mnist.pth"
net.load_state_dict(torch.load(model_path))

virgin_first_layer = net[0].weight.cpu().detach().numpy()
virgin_second_layer = net[3].weight.cpu().detach().numpy()
virgin_third_layer = net[7].weight.cpu().detach().numpy()
virgin_fourth_layer = net[9].weight.cpu().detach().numpy()

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
			net[3].weight[:,:,:,:] = torch.from_numpy(np.add(virgin_second_layer, noise_2)[:,:,:,:])
			net[7].weight[:,:] = torch.from_numpy(np.add(virgin_third_layer, noise_3)[:,:])
			net[9].weight[:,:] = torch.from_numpy(np.add(virgin_fourth_layer, noise_4)[:,:])
		
			
			correct = 0
			total = 0
			for data in test_loader:
				inputs, labels = data
				inputs = inputs.to(device)
				labels = labels.to(device)
	
				# calculate outputs by running images through the network
				outputs, _ = forward_pass(net, num_steps, inputs)
				correct += SF.accuracy_rate(outputs, labels) * outputs.size(1)
				total += outputs.size(1)
	
	
			print(n, epoch, 100 * correct / total)
			mean_acc[epoch, index] += (100 * correct) / (total)

plt.plot(noise_levels, np.mean(mean_acc, axis = 0))
acc_std = np.std(mean_acc, axis = 0)
acc_sem = acc_std/np.sqrt(num_epochs)
plt.fill_between(noise_levels, np.mean(mean_acc, axis = 0) + acc_sem, np.mean(mean_acc, axis = 0) - acc_sem, alpha = 0.2)
plt.xlabel('standard deviation noise multiplier')
plt.ylabel('accuracy')
plt.title('snn model noise test')
plt.show()
