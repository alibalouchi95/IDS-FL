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
        IOT_devices = []
        splitted_dataset = prepare_dataset(number_of_devices=4)

        for idx, _dataset in enumerate(splitted_dataset):
            IOT_devices.append(
                IOTDevice("{0}'s_IOT".format(idx + 1), _dataset))

        IOT_edge = IOTEdge()

        for gen_number in range(total_gen_number):
            self.model = IOT_edge.model

            for IOT_device in IOT_devices:
                IOT_device.update_model(self.model)
                IOT_device.dataset_prep(gen_number + 1, total_gen_number)
                IOT_device.train_model(500)
                self.weights.append(IOT_device.return_weights())
                self.F1_scores.append(IOT_device.F1_score)

            self.model = IOT_edge.ensemble_models(self.weights, self.F1_scores)
            self.weights = []
            self.F1_scores = []

        torch.save(IOT_edge.model.state_dict(), "./final-model.pt")


FL = FederatedLearning()
FL.run()
