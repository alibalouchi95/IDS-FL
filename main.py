from IOT import IOTDevice, IOTEdge
from utils import prepare_dataset
import torch


class FederatedLearning():
    def __init__(self):
        self.model = []
        self.test_data = []
        self.model_coefs = []
        self.weights = []
        self.F1_scores = []

    def run(self, total_gen_number=5):

        splitted_dataset = prepare_dataset(
            number_of_devices=4, number_of_data=300000)

        firstIOT = IOTDevice("first_IOT", splitted_dataset[0])
        secondIOT = IOTDevice("second_IOT", splitted_dataset[1])
        thirdIOT = IOTDevice("third_IOT", splitted_dataset[2])
        fourthIOT = IOTDevice("fourth_IOT", splitted_dataset[3])

        IOT_devices = [firstIOT, secondIOT, thirdIOT, fourthIOT]

        IOT_edge = IOTEdge()

        for gen_number in range(total_gen_number):
            self.model = IOT_edge.model

            for IOT_device in IOT_devices:
                IOT_device.update_model(self.model)
                IOT_device.dataset_prep(gen_number + 1, total_gen_number)
                IOT_device.train_model(100)
                self.weights.append(IOT_device.return_weights())
                self.F1_scores.append(IOT_device.F1_score)

                # self.test_data.append([IOT_device.X_test, IOT_device.y_test])
            self.model = IOT_edge.ensemble_models(self.weights, self.F1_scores)
            self.weights = []
            self.F1_scores = []

        torch.save(IOT_edge.model.state_dict(), "./final-model.pt")


FL = FederatedLearning()
FL.run()
