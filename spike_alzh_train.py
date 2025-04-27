#https://stackoverflow.com/questions/56435961/how-to-access-the-network-weights-while-using-pytorch-nn-sequential

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

import pandas as pd
from sklearn.model_selection import train_test_split








device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

# dataloader arguments
batch_size = 128

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load and preprocess data with one-hot encoding for binary labels
def load_and_preprocess_data(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Normalize all but the last column (features)
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    features_normalized = features / features.max()

    # Convert labels to integers instead of one-hot
    labels_encoded = labels.astype(int).values

    # Convert to float32 tensors
    X = torch.tensor(features_normalized.values, dtype=torch.float32)
    y = torch.tensor(labels_encoded, dtype=torch.long)  # <-- important: use long for classification targets

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# Create DataLoaders
def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32):
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, drop_last=True)

    return train_loader, test_loader

# Example usage
csv_path = r'D:\Neuro_Sci\snntorch_learn_nn_robust\alzh_dataset\alzheimers_disease_data.csv'
X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_path)
train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test, batch_size = batch_size)

# neuron and simulation parameters
beta = 0.5
num_steps = 150

spike_grad = surrogate.fast_sigmoid(slope=25)
lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

#yes there is a better way to do this, no I aparently don't know how to do it
hid_size = 64
net = nn.Sequential(
	nn.Linear(33, hid_size),
	snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
	
	nn.Linear(hid_size, hid_size),
	snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

	nn.Linear(hid_size, hid_size),
	snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

	nn.Linear(hid_size, 2),
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

# already imported snntorch.functional as SF
loss_fn = SF.ce_rate_loss() #just for my own notes, cross entropy is standard and rate means that its looking at spike rate or frequency rather than timing

def batch_accuracy(train_loader, net, num_steps):
	with torch.no_grad():
		total = 0
		acc = 0
		net.eval()

		train_loader = iter(train_loader)
	for data, targets in train_loader:
		data = data.to(device)
		targets = targets.to(device)
		spk_rec, _ = forward_pass(net, num_steps, data)

		acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
		total += spk_rec.size(1)
	return acc/total


optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 100
loss_hist = []
test_acc_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):

	# Training loop
	for data, targets in iter(train_loader):
		data = data.to(device)
		targets = targets.to(device)

		# forward pass
		net.train()
		spk_rec, _ = forward_pass(net, num_steps, data)

		# initialize the loss & sum over time
		loss_val = loss_fn(spk_rec, targets)

		# Gradient calculation + weight update
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()

		# Store loss history for future plotting
		loss_hist.append(loss_val.item())

		# Test set
		if counter % 50 == 0:
			with torch.no_grad():
				net.eval()

				# Test set forward pass
				test_acc = batch_accuracy(test_loader, net, num_steps)
				print(f"Epoch: {epoch}, Iteration: {counter}, Test Acc: {test_acc * 100:.2f}%\n")
				test_acc_hist.append(test_acc.item())

		counter += 1
	torch.save(net.state_dict(), f"{epoch}_spike_model_alzh.pth")

# Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(test_acc_hist)
plt.title("Test Set Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

