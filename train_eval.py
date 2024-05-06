import pandas as pd
import torch
import torchvision
import numpy as np
import optuna
import sys
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(5, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary output

        init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)  # Applying sigmoid to get binary output
        return out

class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.from_numpy(data.values).float()
        self.target = torch.from_numpy(target.values).float().view(-1, 1)  # Reshape target to (batch_size, 1)

        self.data, self.target = self.data.to('cuda'), self.target.to('cuda')
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def train_model(X_train, y_train, lr, hidden_size_1, hidden_size_2):

    model = NeuralNetwork(hidden_size_1, hidden_size_2)
    model.to('cuda')

    criterion = nn.BCELoss()  # Binary cross-entropy loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_data = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def evaluate(model, X_test, y_test):

    test_data = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    model.to('cuda')

    model.eval()  # Set model to evaluation mode
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, targets_batch in test_loader:
            # Move data to the same device as the model
            inputs = inputs.to('cuda')
            targets_batch = targets_batch.to('cuda')

            # Forward pass
            outputs = model(inputs)

            # Convert outputs to binary predictions (0 or 1)
            predicted_classes = (outputs >= 0.5).float()

            # Collect predictions and targets
            predictions.extend(predicted_classes.cpu().numpy())
            targets.extend(targets_batch.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(targets, predictions)
    print('Accuracy: ', accuracy)

if __name__ == '__main__':

    train_path = sys.argv[1]

    test_path = sys.argv[2]

    num_epochs = sys.argv[3]

    lr = sys.argv[4]

    hidden_size_1 = sys.argv[5]

    hidden_size_2 = sys.argv[6]

    save_path = sys.argv[7]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[['Number of Committees', 'Number of Cosponsors', 'Number of Sponsors', 'Majority', 'Partisan Lean']]
    y_train = train_df['Passed']

    X_test = test_df[['Number of Committees', 'Number of Cosponsors', 'Number of Sponsors', 'Majority', 'Partisan Lean']]
    y_test = test_df['Predictions']

    model = train_model(X_train, y_train, lr, hidden_size_1, hidden_size_2)

    model.save()

    torch.save(model.state_dict(), save_path)

    print('Model Saved!')