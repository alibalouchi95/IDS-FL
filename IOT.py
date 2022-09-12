import torch.nn as nn
import copy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch
from utils import CustomDataset, model, dataset_features, dataset_label
from torch.utils.data import DataLoader

learning_rate = 0.1
loss_function = nn.BCELoss()

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class IOTEdge():
    def __init__(self):
        self.model = model

    def ensemble_models(self, models_coefs, F1_scores):
        F1_sum = sum(F1_scores)
        coefs = []

        for coef in range(len(models_coefs)):
            _coef = []
            for weight in models_coefs[coef]:
                _coef.append(weight.detach().numpy())

            coefs.append(np.multiply(
                _coef, F1_scores[coef]/F1_sum, dtype=object))
        aggregated_model_coefs = np.sum(coefs, axis=0)

        self.model.fc1.weight = nn.Parameter(
            torch.Tensor(aggregated_model_coefs[0]))
        self.model.relu1.weight = nn.Parameter(
            torch.Tensor(aggregated_model_coefs[1]))
        self.model.fc2.weight = nn.Parameter(
            torch.Tensor(aggregated_model_coefs[2]))
        self.model.relu2.weight = nn.Parameter(
            torch.Tensor(aggregated_model_coefs[3]))
        self.model.fc3.weight = nn.Parameter(
            torch.Tensor(aggregated_model_coefs[4]))
        self.model.relu3.weight = nn.Parameter(
            torch.Tensor(aggregated_model_coefs[5]))
        self.model.fc4.weight = nn.Parameter(
            torch.Tensor(aggregated_model_coefs[6]))
        self.model.relu4.weight = nn.Parameter(
            torch.Tensor(aggregated_model_coefs[7]))
        self.model.fc5.weight = nn.Parameter(
            torch.Tensor(aggregated_model_coefs[8]))

        return self.model


class IOTDevice():

    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset.dropna()
        self.dataset_size = len(dataset)
        self.F1_score = 0
        self.recall = 0
        self.precision = 0

    def update_model(self, model):
        self.model = model

    def dataset_prep(self, generation_number, total_generation_number):
        iloc_from = 0
        iloc_to = int((self.dataset_size / total_generation_number) *
                      (generation_number))

        new_dataset = self.dataset.iloc[iloc_from:iloc_to]

        msk = np.random.rand(len(new_dataset)) < 0.8
        train_dataset = new_dataset[msk]
        test_dataset = new_dataset[~msk]

        print("BENIGN data shape in this train epoch dataset is: {0}".format(
            new_dataset[new_dataset[" Label"] == 0].shape))
        print("ATTACK data shape in this train epoch dataset: {0}".format(
            new_dataset[new_dataset[" Label"] == 1].shape))

        self.dataset_loader_train = DataLoader(CustomDataset(
            train_dataset), batch_size=32, shuffle=True)

        self.dataset_test = test_dataset

    def train_model(self, epochs):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        losses = []

        print("training for {0} IOT device".format(self.name))

        for epoch in range(epochs):
            for x_train, y_train in self.dataset_loader_train:
                pred_y = self.model(x_train)

                pred_y = (pred_y > 0.5).float()

                loss = loss_function(pred_y, y_train)
                losses.append(loss.item())

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch % 20 == 0):
                print("     EPOCH %f" % (int(epoch)))
                print("         LOSS %f" % (np.mean(losses)))

        self.test_model()
        print("The {0}'s F1 score is: {1}, Recall is: {2}, Precision is: {3}".format(
            self.name, self.F1_score, self.recall, self.precision), "\n")

    def test_model(self):
        with torch.no_grad():
            dataset = copy.deepcopy(self.dataset)
            y = dataset.drop(dataset_features, axis=1)
            X = dataset.drop(dataset_label, axis=1)
            y_pred = self.model(torch.Tensor(X.values))

            y_pred = (y_pred > 0.5).float()
            if np.array_equal(np.array(y_pred.unique()), np.unique(torch.Tensor(y.values))):
                self.precision, self.recall, self.F1_score = precision_recall_fscore_support(
                    y, y_pred, average='macro', labels=np.unique(y_pred))[: 3]
            else:
                print(
                    "WARNING: The predicted labels and test labels do not have same unique values")
                self.precision, self.recall, self.F1_score = 0, 0, 0

    def return_weights(self):
        return self.model.parameters()

    def draw_chart(self):
        plt.plot(self.losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title("Loss Chart of %f" % (self.name))
        plt.show()
