import matplotlib.pyplot as plot
import numpy as np
import os
import pandas as pd
import utils


def visualize(input_times, input_features, input_labels):
    input_features = list(input_features.T)
    args = []
    for dim_values in input_features:
        args += [input_times, dim_values]
    args += [input_times, input_labels]
    plot.plot(*args)
    plot.title("Time series")
    plot.xlabel('Time')
    plot.ylabel('Readings')
    plot.xticks(np.arange(0, 10, 0.5))
    plot.yticks(np.arange(-2, 2, 0.2))
    plot.legend([str(i) for i in range(len(input_features))] + ["labels"])
    plot.grid(True, which='both')
    plot.show()


def label_map(file_name):
    name_map = {
        "bunny": 0,
        "cowboy": 1,
        "handmotor": 2,
        "rocket": 3,
        "tapshoulder": 4,
        "hunchback": 5
    }
    for k, v in name_map.items():
        if k in file_name:
            return v
    return -1


if __name__ == "__main__":
    DATA_ROOT = "data"

    # Configs
    CONVERT_TO_TSV = True
    # VISUALIZE_FILE = "data/wen_hao_bunny.tsv"
    VISUALIZE_FILE = None
    TIME_WINDOW = 500
    SELECTED_COLS = [2, 3, 4, 8, 9, 10, 14, 15, 16]

    # convert to tsv file
    if CONVERT_TO_TSV:
        for data_file in os.listdir(DATA_ROOT):
            data_file_path = os.path.join(DATA_ROOT, data_file)
            df = pd.read_excel(data_file_path, sheetname='Sheet1', header=None)
            output_path = os.path.join(DATA_ROOT, data_file.replace(" ", "_").replace(".xlsx", ".tsv").lower())
            df.to_csv(output_path, sep="\t", index=False, header=None)
    if VISUALIZE_FILE is not None:
        times, features, labels = [], [], []
        dance_data = utils.read_csv(VISUALIZE_FILE, load_header=True, delimiter="\t")
        for row in dance_data[:TIME_WINDOW]:
            times.append(float(row[1]))
            selected_features = []
            for c in SELECTED_COLS:
                selected_features.append(float(row[c]))
            features.append(np.array(selected_features))
            labels.append(label_map(VISUALIZE_FILE))
        visualize(np.array(times), np.array(features), np.array(labels))