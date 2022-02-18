# to access to effective_dimension module
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import evaluate, train
from effective_dimension.effective_dimension import get_effective_dimension


VALIDATION_RATIO = 0.9
BATCH_SIZE = 64
LEARNING_RATE = 0.01
IMAGE_SIZE = 28
OUTPUT_SIZE = 10

# get device name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pull MNIST datatset and construct loaders
trainset = datasets.MNIST(root="/tmp/data/", train=True, download=True, transform=transforms.ToTensor())
testset = datasets.MNIST(root="/tmp/data/", train=False, download=True, transform=transforms.ToTensor())
n_train_examples = int(len(trainset) * VALIDATION_RATIO)
n_valid_examples = len(trainset) - n_train_examples
train_data, valid_data = data.random_split(trainset, [n_train_examples, n_valid_examples])

trainloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True)

# defining the neural network model
class Model(nn.Module):
    def __init__(self, size: Sequence[int], bias=False):
        super(Model, self).__init__()
        self.size = size
        self.layers = nn.ModuleList([
            nn.Linear(self.size[i - 1], self.size[i], bias=bias) for i in range(1, len(self.size))
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i in range(len(self.size) - 2):
            x = F.leaky_relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

model = Model([IMAGE_SIZE*IMAGE_SIZE, 30, 30, OUTPUT_SIZE]).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training
epochs = 25
best_valid_loss = float('inf')
for epoch in range(epochs):
    train_loss, train_acc = train(model, trainloader, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, validloader, optimizer, criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

# testing
model.load_state_dict(torch.load('model.pt'))
test_loss, test_acc = evaluate(model, testloader, optimizer, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

# computes effective dimension
ed = get_effective_dimension(model, trainloader, OUTPUT_SIZE, n_train_examples, device=device, normalized=False)
print("Effective dimension :", ed)

# computes normalized effective dimension
normalized_ed = get_effective_dimension(model, trainloader, OUTPUT_SIZE, n_train_examples, device=device, normalized=True)
print("Normalized effective dimension :", normalized_ed)
