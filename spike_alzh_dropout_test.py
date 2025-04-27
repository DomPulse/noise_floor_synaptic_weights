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

drop_levels = np.linspace(0.1, 1, 10)
print(drop_levels)
num_epochs = 50
model_path = "current_best_spike_alzh_83\\20_spike_model_alzh.pth"
net.load_state_dict(torch.load(model_path))

virgin_first_layer = net[0].weight.cpu().detach().numpy()
virgin_second_layer = net[2].weight.cpu().detach().numpy()
virgin_third_layer = net[4].weight.cpu().detach().numpy()
virgin_fourth_layer = net[6].weight.cpu().detach().numpy()

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

mean_acc = np.zeros((num_epochs, len(drop_levels)))
for index, n in enumerate(drop_levels):
	for epoch in range(num_epochs):  # loop over the dataset multiple times
		with torch.no_grad():
			net.load_state_dict(torch.load(model_path))
			
			drop_1 = np.random.rand(*first_shape) < drop_levels[index]
			drop_2 = np.random.rand(*second_shape) < drop_levels[index]
			drop_3 = np.random.rand(*third_shape) < drop_levels[index]
			drop_4 = np.random.rand(*fourth_shape) < drop_levels[index]
			
			net[0].weight[:,:] = torch.from_numpy(np.multiply(virgin_first_layer, drop_1)[:,:])
			net[2].weight[:,:] = torch.from_numpy(np.multiply(virgin_second_layer, drop_2)[:,:])
			net[4].weight[:,:] = torch.from_numpy(np.multiply(virgin_third_layer, drop_3)[:,:])
			net[6].weight[:,:] = torch.from_numpy(np.multiply(virgin_fourth_layer, drop_4)[:,:])
		
			
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

plt.plot(drop_levels, np.mean(mean_acc, axis = 0))
acc_std = np.std(mean_acc, axis = 0)
acc_sem = acc_std/np.sqrt(num_epochs)
plt.fill_between(drop_levels, np.mean(mean_acc, axis = 0) + acc_sem, np.mean(mean_acc, axis = 0) - acc_sem, alpha = 0.2)
plt.xlabel('percent connections remaining')
plt.ylabel('accuracy')
plt.title('snn model dropout test')
plt.show()
