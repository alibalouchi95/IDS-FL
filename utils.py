import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import Dataset


dataset_features = [" Bwd Packet Length Std", " Flow IAT Min", " Fwd IAT Min", " Flow IAT Mean",
                    " Flow Duration", " Flow IAT Std", " Active Min", "Active Mean", " Bwd IAT Mean", " Average Packet Size"]
dataset_label = " Label"


class CustomDataset(Dataset):

    def __init__(self, dataset):

        y = dataset.drop(dataset_features, axis=1)
        X = dataset.drop(dataset_label, axis=1)

        self.x_train = torch.tensor(X.to_numpy('float32'))
        self.y_train = torch.tensor(y.to_numpy('float32'), requires_grad=True)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


def prepare_dataset(number_of_data=None, number_of_devices=4):

    dataset_cols = copy.deepcopy(dataset_features)
    dataset_cols.append(dataset_label)

    print("Reading The dataset...")
    dataset = pd.read_csv(
        "./Dataset.csv", usecols=dataset_cols)

    print(dataset.shape)

    dataset.iloc[np.random.permutation(len(dataset))]

    dataset[' Label'] = dataset[' Label'].replace({
        'Heartbleed': 1, 'PortScan': 1, 'Bot': 1,
        'Infiltration': 1, 'Web Attack � Brute Force': 1,
        'Web Attack � XSS': 1, 'Web Attack � Sql Injection': 1,
        'FTP-Patator': 1, 'SSH-Patator': 1})

    # dataset.dropna()

    dataset[' Label'] = dataset[' Label'].replace(
        {'BENIGN': 0, "DoS GoldenEye": 1, "DoS Hulk": 1, "DoS Slowhttp": 1, "DoS slowloris": 1, "DDoS": 1, "DoS Slowhttptest": 1})

    if (number_of_data):
        dataset = dataset.iloc[:number_of_data]

    _size = len(dataset)

    result = []

    for idx in range(number_of_devices):
        iloc_from = int(_size * (idx * (1/number_of_devices)))
        iloc_to = int(_size * ((idx + 1) * (1/number_of_devices)))
        result.append(dataset.iloc[iloc_from: iloc_to])

    print("Done Reading The dataset. \n")
    return result


# def prepare_dataset():

#     dataset_cols = copy.deepcopy(dataset_features)
#     dataset_cols.append(dataset_label)

#     dataset = pd.read_csv(
#         "./Dataset.csv", usecols=dataset_cols)

#     dataset[' Label'] = dataset[' Label'].replace({
#         'Heartbleed': 1, 'PortScan': 1, 'Bot': 1,
#         'Infiltration': 1, 'Web Attack � Brute Force': 1,
#         'Web Attack � XSS': 1, 'Web Attack � Sql Injection': 1,
#         'FTP-Patator': 1, 'SSH-Patator': 1})

#     dataset.dropna()

#     dataset[' Label'] = dataset[' Label'].replace(
#         {'BENIGN': 0, "DoS GoldenEye": 1, "DoS Hulk": 1, "DoS Slowhttp": 1, "DoS slowloris": 1, "DDoS": 1, "DoS Slowhttptest": 1})

#     # dataset = dataset.iloc[:300000]

#     _size = len(dataset)

#     _size_obj = {
#         "1": dataset.iloc[: int(_size * 0.25)],
#         "2": dataset.iloc[int(_size * 0.25): int(_size * 0.5)],
#         "3": dataset.iloc[int(_size * 0.5): int(_size * 0.75)],
#         "4": dataset.iloc[int(_size * 0.75): _size],
#     }

#     return _size_obj


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 15)
        self.fc2 = nn.Linear(15, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 15)
        self.relu1 = F.relu
        self.relu2 = F.relu
        self.relu3 = F.relu
        self.relu4 = F.relu
        self.fc5 = nn.Linear(15, 1)
        self.sigmoid = torch.sigmoid
        self.softmax = F.softmax

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)

        x = self.sigmoid(x)
        # x = self.softmax(x)
        return x


model = Model()
