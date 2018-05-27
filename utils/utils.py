import numpy as np
from deeplearning_ai.datasets.settings import DATASET_BASE_PATH


def load_data_from_csv(key):
    train_dataset_path = "{base_path}/train.csv".format(
        base_path=DATASET_BASE_PATH)
    test_dataset_path = "{base_path}/test.csv".format(
        base_path=DATASET_BASE_PATH)
    train_data = np.genfromtxt(train_dataset_path, delimiter=',')
    test_data = np.genfromtxt(test_dataset_path, delimiter=',')
    return train_data, test_data
