#https://stackoverflow.com/questions/56435961/how-to-access-the-network-weights-while-using-pytorch-nn-sequential

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

    # Convert labels to one-hot vectors
    labels_one_hot = pd.get_dummies(labels).values  # Assumes only 0 and 1

    # Convert to float32 tensors
    X = torch.tensor(features_normalized.values, dtype=torch.float32)
    y = torch.tensor(labels_one_hot, dtype=torch.float32)

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
num_steps = 50

# Initialize Network
#yes there is a better way to do this, no I aparently don't know how to do it
dropout_p = 0.2
hid_size = 64
act_func = nn.Tanh()
net = nn.Sequential(
	nn.Linear(33, hid_size),
	act_func,
	nn.Dropout(dropout_p),
	
	nn.Linear(hid_size, hid_size),
	act_func,
	nn.Dropout(dropout_p),

	nn.Linear(hid_size, hid_size),
	act_func,
	nn.Dropout(dropout_p),

	nn.Linear(hid_size, 2),
	nn.Softmax()
	).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.8, 0.9))
#optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
num_epochs = 10
loss_hist = []
test_acc_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(train_loader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)

		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 50 == 0:	# print every 2000 mini-batches
			correct = 0
			total = 0
			# since we're not training, we don't need to calculate the gradients for our outputs
			with torch.no_grad():
				for i, data in enumerate(test_loader, 0):
					inputs, labels = data
					inputs = inputs.to(device)
					labels = labels.to(device)
					# calculate outputs by running images through the network
					outputs = net(inputs)
					
					# the class with the highest energy is what we choose as prediction
					_, predicted = torch.max(outputs.data, 1)
					_, true_labels = torch.max(labels.data, 1)
					total += labels.size(0)
					correct += (predicted == true_labels).sum().item()

				print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
	torch.save(net.state_dict(), f"{epoch}_trad_model_alzh.pth")
	print("gamed on")

print('Finished Training')

