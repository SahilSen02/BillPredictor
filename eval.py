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

if __name__ == '__main__':

    model_path = sys.argv[1]

    hidden_size_1 = sys.argv[3]

    hidden_size_2 = sys.argv[4]

    test_path = sys.argv[5]

    test_df = pd.read_csv(test_path)

    inputs = test_df[['Number of Committees', 'Number of Cosponsors', 'Number of Sponsors', 'Majority', 'Partisan Lean']]

    model = NeuralNetwork(hidden_size_1, hidden_size_2)

    model.load_state_dict(torch.load(model_path))

    model.eval()

    with torch.no_grad():
        pred = model(inputs)

    test_df['Predictions'] = pred

    test_df.to_csv('Test with Predictions.csv')





