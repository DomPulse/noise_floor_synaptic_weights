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
data_path='D:\\Neuro Sci\\snntorch_learn_nn_robust\\mnist_data'

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
dropout_p = 0.0
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

noise_levels = np.linspace(0.1, 2, 20)
print(noise_levels)
num_epochs = 50
model_path = "current_best_trad_alzh_82\\9_trad_model_alzh.pth"
net.load_state_dict(torch.load(model_path))

virgin_first_layer = net[0].weight.cpu().detach().numpy()
virgin_second_layer = net[3].weight.cpu().detach().numpy()
virgin_third_layer = net[6].weight.cpu().detach().numpy()
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
			
			net[0].weight[:,:] = torch.from_numpy(np.add(virgin_first_layer, noise_1)[:,:])
			net[3].weight[:,:] = torch.from_numpy(np.add(virgin_second_layer, noise_2)[:,:])
			net[6].weight[:,:] = torch.from_numpy(np.add(virgin_third_layer, noise_3)[:,:])
			net[9].weight[:,:] = torch.from_numpy(np.add(virgin_fourth_layer, noise_4)[:,:])
			
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
				_, true_labels = torch.max(labels.data, 1)
				total += labels.size(0)
				correct += (predicted == true_labels).sum().item()
	
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

