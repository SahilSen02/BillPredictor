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

def objective(trial, X_train, y_train, X_test, y_test):

    trial_metrics = dict()

    trial_number = trial.number
    epoch_losses = []
    epoch_accuracies = []
    try:
        hidden_size1 = trial.suggest_int('hidden_size1', 10, 100)
        hidden_size2 = trial.suggest_int('hidden_size2', 10, 100)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

        print(f"Trial {trial.number}: Hidden Layer 1 Size = {hidden_size1}, Hidden Layer 2 Size = {hidden_size2}, Learning Rate = {lr}")

        model = NeuralNetwork(hidden_size1, hidden_size2).to('cuda')
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_dataset = CustomDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        min_loss = float('inf')

        for epoch in range(10):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                #loss = bce_loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_losses.append(epoch_loss)

            epoch_loss /= len(train_loader)
            print(f"Trial {trial.number}, Epoch {epoch}: Loss = {epoch_loss}")

            if epoch_loss < min_loss:
                min_loss = epoch_loss

        test_dataset = CustomDataset(X_test, y_test)

        test_loader = DataLoader(test_dataset, batch_size = 32, shuffle=True)

        accuracy = evaluate_accuracy(model, test_loader, 'cuda')

        epoch_accuracies.append(accuracy)
        print(f"Trial {trial.number}: Final accuracy = {accuracy}, Best loss = {min_loss}")

        trial_metrics[trial_number] = {'losses': epoch_losses, 'accuracies': accuracy}

        return accuracy

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':

    train_path = sys.argv[1]

    test_path = sys.argv[2]

    num_trials = sys.argv[3]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[['Number of Committees', 'Number of Cosponsors', 'Number of Sponsors', 'Majority', 'Partisan Lean']]
    y_train = train_df['Passed']

    X_test = test_df[['Number of Committees', 'Number of Cosponsors', 'Number of Sponsors', 'Majority', 'Partisan Lean']]
    y_test = test_df['Predictions']

    study = optuna.create_study(direction='maximize')

    study.optimize(objective, n_trials=num_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f" Accuracy: {trial.value:.4f}")
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


