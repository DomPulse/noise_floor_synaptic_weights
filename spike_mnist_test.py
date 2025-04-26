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

num_epochs = 10
model_path = "current_best_spike_mnist_98\\2_spike_model_mnist.pth"
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

			#net.conv1.weight[:,:,:,:] = torch.from_numpy(np.add(net.conv1.weight.cpu().detach().numpy(), np.random.normal(0, conv1_std/i, conv1_shape))[:,:,:,:])
			#net.conv2.weight[:,:,:,:] = torch.from_numpy(np.add(net.conv2.weight.cpu().detach().numpy(), np.random.normal(0, conv2_std/i, conv2_shape))[:,:,:,:])
			#net.conv3.weight[:,:,:,:] = torch.from_numpy(np.add(net.conv3.weight.cpu().detach().numpy(), np.random.normal(0, conv3_std/i, conv3_shape))[:,:,:,:])
			#net.conv4.weight[:,:,:,:] = torch.from_numpy(np.add(net.conv4.weight.cpu().detach().numpy(), np.random.normal(0, conv4_std/i, conv4_shape))[:,:,:,:])
			#net.fc1.weight[:,:] = torch.from_numpy(np.add(net.fc1.weight.cpu().detach().numpy(), np.random.normal(0, fc1_std/i, fc1_shape))[:,:])
			#net.fc2.weight[:,:] = torch.from_numpy(np.add(net.fc2.weight.cpu().detach().numpy(), np.random.normal(0, fc2_std/i, fc2_shape))[:,:])
			#net.fc3.weight[:,:] = torch.from_numpy(np.add(net.fc3.weight.cpu().detach().numpy(), np.random.normal(0, fc3_std/i, fc3_shape))[:,:])

			# calculate outputs by running images through the network
			outputs, _ = forward_pass(net, num_steps, inputs)
			correct += SF.accuracy_rate(outputs, labels) * outputs.size(1)
			total += outputs.size(1)


		print(epoch, 100 * correct / total)


