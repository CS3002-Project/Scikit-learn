import numpy as np
import os
from run_preprocess import label_map
from models import build_window_data, train_rf
import utils


def read_data(data_dir):
    test_size = 0.2
    x_array_train, y_array_train = [], []
    x_array_dev, y_array_dev = [], []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            x_sample = []
            label = label_map(file_name)
            x_raw_data = utils.read_csv(os.path.join(data_dir, file_name), load_header=True, delimiter=",")
            for row in x_raw_data:
                x_snapshot = np.array([float(reading) for reading in row[2:]])
                x_sample.append(x_snapshot)
            y_sample = [label] * len(x_sample)
            test_num = int(len(x_sample) * test_size)

            x_array_train.append(np.array(x_sample[:-test_num]))
            y_array_train.append(np.array(y_sample[:-test_num]))
            x_array_dev.append(np.array(x_sample[-test_num:]))
            y_array_dev.append(np.array(y_sample[-test_num:]))
    return x_array_train, y_array_train, x_array_dev, y_array_dev


def run_svm_seq(x_array_train, y_array_train, x_array_dev, y_array_dev):
    window_size = 10
    x_train, y_train, x_dev, y_dev = [], [], [], []
    for i in range(len(x_array_train)):
        x_window_train, y_window_train, x_window_dev, y_window_dev \
            = build_window_data(x_array_train[i], y_array_train[i], x_array_dev[i], y_array_dev[i], window_size)
        x_window_train = [np.concatenate(x) for x in x_window_train]
        x_window_dev = [np.concatenate(x) for x in x_window_dev]

        x_train += x_window_train
        y_train += y_window_train
        x_dev += x_window_dev
        y_dev += y_window_dev

    train_rf(x_train, y_train, x_dev, y_dev)


def main():
    data_dir = "data"
    x_array_train, y_array_train, x_array_dev, y_array_dev = read_data(data_dir)
    run_svm_seq(x_array_train, y_array_train, x_array_dev, y_array_dev)


if __name__ == "__main__":
    main()