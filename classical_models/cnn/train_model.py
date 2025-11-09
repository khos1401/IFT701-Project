import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import zipfile
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import Food101
from PIL import Image
import random
from tqdm import tqdm

from dataset.load_dataset import NPZDataLoader

loader = NPZDataLoader("dataset/cats_dogs_32x32.npz")
X = torch.tensor(loader.X, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
y = torch.tensor(loader.y, dtype=torch.long)

class CatsDogsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

dataset = CatsDogsDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



def training_loop(dataloader: torch.utils.data.DataLoader,
                  model: torch.nn.Module,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  device: torch.device = 'cpu'
                  ) -> None:
  """
  Training loop
  """
  size = len(dataloader.dataset)

  model.train()
  pbar = tqdm(dataloader, desc="Training")
  for batch, (X, y) in enumerate(pbar):
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    if batch % 10 == 0:
      loss, current = loss.item(), batch * len(X)
      pbar.set_postfix({'loss': loss, 'current': f'{current}/{size}'})

def testing_loop(dataloader,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 device: torch.device = 'cpu'
                 ) -> None:
  """
  Testing loop
  """
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  with torch.inference_mode():
    test_loss, correct = 0, 0
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      y_pred = model(X)
      test_loss += loss_fn(y_pred, y).item()
      correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {100*correct:>5.1f}%, Avg loss: {test_loss:>10.6f} \n")


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 20
for epoch in range(epochs):
  print(f"Epoch {epoch+1}\n-------------------------------")
  training_loop(dataloader, model, criterion, optimizer)
  testing_loop(dataloader, model, criterion)