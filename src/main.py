import sys
sys.path.extend(['../cmn'])
from scipy.sparse.data import _data_matrix
from dal.data_utils import *
from cmn.fnn import FNN
from cmn.custom_dataset import TFDataset
import torch
from torch import optim 
from torch import nn 
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm  # For nice progress bar!

from scipy.sparse import hstack
import numpy as np

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "../data/raw/dblp.v12.json"

all_authors, all_docs, input_data, output_data = read_data(data_path)

index_to_author, author_to_index = build_index_authors(all_authors)

index_to_skill, skill_to_index = build_index_skills(all_docs)

input_matrix, output_matrix = build_dataset(all_docs, skill_to_index, author_to_index)

# print(output_matrix.shape)

# print(input_matrix.shape)

# Setting the hyperparameters
input_size = len(index_to_skill)
output_size = len(index_to_author) + 1
learning_rate = 0.001
batch_size = 100
num_epochs = 3


# data_matrix = hstack([input_matrix, output_matrix]).tocsr()
# print(data_matrix.shape)


data_matrix = TFDataset(input_matrix, output_matrix)

training_matrix = data_matrix[:700]
validation_matrix = data_matrix[700:850]
test_matrix = data_matrix[850:]


# print(training_matrix.shape)


training_dataloader = DataLoader(training_matrix, batch_size=batch_size, shuffle=False, num_workers=0)
validation_dataloader = DataLoader(validation_matrix, batch_size=batch_size, shuffle=False, num_workers=0)
testing_dataloader = DataLoader(test_matrix, batch_size=batch_size, shuffle=False, num_workers=0)

# print(training_dataloader.dataset)

data_loaders = {"train": training_dataloader, "val": validation_dataloader}


# Initialize network
model = FNN(input_size=input_size, output_size=output_size).to(device)

# Loss and optimizer

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# print(training_dataloader.dataset[0].shape)
# print(training_dataloader.dataset[1].shape)

# # Train Network
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 15)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        for batch_idx, (data, targets) in enumerate(data_loaders[phase]):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = torch.sqrt(criterion(scores, targets))

            # backward
            optimizer.zero_grad()
            if phase == 'train':
                    loss.backward()
                    optimizer.step()

            



print(f"Accuracy on training set: {check_accuracy(training_dataloader, model, device)*100:.2f}")
print(f"Accuracy on training set: {check_accuracy(validation_dataloader, model, device)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(testing_dataloader, model, device)*100:.2f}")