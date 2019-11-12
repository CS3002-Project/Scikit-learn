import numpy as np
import os
from run_preprocess import label_map, NAME_MAP
from models import build_window_data, train_rf, train_mlp, feature_extraction, train_rf_batches
import utils
import copy
import random
from collections import deque
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split


reverse_label_map = {
        0: "bunny",
        1: "cowboy",
        2: "handmotor",
        3: "rocket",
        4: "tapshoulder",
        5: "hunchback",
        6: "james",
        7: "chicken",
        8: "movingsalute",
        9: "whip",
    # 99: "idle",
}


def evaluate(predicted_move, input_file):
    if predicted_move not in input_file:
        for k in reverse_label_map.values():
            if k in input_file:
                return 0
    return 1


def read_data(file_paths):
    test_size = 0.2
    x_array_train, y_array_train = [], []
    x_array_dev, y_array_dev = [], []
    for file_path in file_paths:
        x_sample = []
        label = label_map(file_path)
        x_raw_data = utils.read_csv(file_path, load_header=True, delimiter=",")
        for row in x_raw_data:
            x_snapshot = np.array([float(reading) for reading in row[3:]])
            x_sample.append(x_snapshot)
        y_sample = [label] * len(x_sample)
        test_num = int(len(x_sample) * test_size)

        x_array_train.append(np.array(x_sample[:-test_num]))
        y_array_train.append(np.array(y_sample[:-test_num]))
        x_array_dev.append(np.array(x_sample[-test_num:]))
        y_array_dev.append(np.array(y_sample[-test_num:]))
    return x_array_train, y_array_train, x_array_dev, y_array_dev


def prepare_data(x_array_train, y_array_train, x_array_dev, y_array_dev, config):
    prediction_window_size = config["prediction_window_size"]
    feature_window_size = config["feature_window_size"]
    x_train, y_train, x_dev, y_dev = [], [], [], []
    for i in tqdm(range(len(x_array_train)), desc="Preparing data"):
        x_window_train, y_window_train, x_window_dev, y_window_dev \
            = build_window_data(x_array_train[i], y_array_train[i], x_array_dev[i], y_array_dev[i],
                                prediction_window_size, feature_window_size)
        x_train += x_window_train
        y_train += y_window_train
        x_dev += x_window_dev
        y_dev += y_window_dev

    return x_train, y_train, x_dev, y_dev


def divide_into_batches(x_data, y_data, batch_size):
    x_batches, y_batches = [], []
    data_size = len(x_data)
    i = 0
    while i < data_size:
        x_batches.append(x_data[i: i + batch_size])
        y_batches.append(y_data[i: i + batch_size])
        i += batch_size
    return x_batches, y_batches


def split_train_test(data_dir, num_iterations):
    data_stats = {}
    all_moves = list(NAME_MAP.keys())
    move_file_lookup = {}

    for file_name in os.listdir(data_dir):
        person = file_name.split("_")[0]
        for move in all_moves:
            if move in file_name:
                if person not in data_stats:
                    data_stats[person] = []
                data_stats[person].append(move)
                if move not in move_file_lookup:
                    move_file_lookup[move] = {}
                if person not in move_file_lookup[move]:
                    move_file_lookup[move][person] = []
                move_file_lookup[move][person].append(os.path.join(data_dir, file_name))

    full_people = []
    for person, moves in data_stats.items():
        print("{} has {} moves".format(person, len(moves)))
        unique_moves = set(moves)
        missing_moves = set(all_moves).difference(unique_moves)
        if len(missing_moves) > 0:
            print("{} doesn't have {}".format(person, missing_moves))
        else:
            print("{} has all the moves".format(person))
            full_people.append(person)

    iter_train_files, iter_test_files = [], []

    for _ in tqdm(range(num_iterations)):
        all_people = []
        train_files, test_files = [], []
        for move in all_moves:      # for each dance move, choose a different person to test
            if len(all_people) == 0:
                all_people = copy.deepcopy(full_people)
            next_person = random.choice(all_people)
            dance_file = random.choice(move_file_lookup[move][next_person])
            all_people.remove(next_person)          # choose a different person
            test_files.append(dance_file)
        for move, people_files in move_file_lookup.items():
            for people, files in people_files.items():
                train_files.extend([f for f in files if f not in test_files])
        iter_train_files.append(train_files)
        iter_test_files.append(test_files)

    return iter_train_files, iter_test_files


def main():
    num_iters = 1
    batch_size = 1048
    data_dir = "data"
    config = {
        "prediction_window_size": 32,
        "feature_window_size": 10,
        "min_confidence": 0.8,
        "model_type": "rf",
        "max_consecutive_agrees": 2
    }
    iter_train_files, iter_test_files = split_train_test(data_dir, num_iters)
    all_iter_test_accuracy, all_iter_first_correct = [], []
    # model_name = "rf".format(config["model_type"],
    #                                      config["prediction_window_size"],
    #                                      config["feature_window_size"],
    #                                      config["min_confidence"],
    #                                      config["max_consecutive_agrees"])
    model_name = "rf"
    for i in range(num_iters):
        train_files, test_files = iter_train_files[i], iter_test_files[i]
        x_array_train, y_array_train, x_array_dev, y_array_dev = read_data(train_files)
        x_train, y_train, x_dev, y_dev = prepare_data(x_array_train, y_array_train, x_array_dev, y_array_dev, config)
        x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.0001)
        x_train_batches, y_train_batches = divide_into_batches(x_train, y_train, batch_size)
        x_dev_batches, y_dev_batches = divide_into_batches(x_dev, y_dev, batch_size)
        trained_model = train_rf_batches(x_train_batches, y_train_batches, x_dev_batches, y_dev_batches, model_name)
        iter_test_accuracy, iter_first_correct = test(trained_model, test_files, config)
        all_iter_test_accuracy.append(iter_test_accuracy)
        all_iter_first_correct.append(iter_first_correct)
    final_accuracy = np.mean(all_iter_test_accuracy)
    final_first_correct = np.mean(all_iter_first_correct)
    with open("{}_eval.json".format(model_name), "w") as f:
        json.dump({
            "first_correct": final_first_correct,
            "accuracy": final_accuracy
        }, f)


def test(trained_model, test_files, config):
    prediction_window_size = config["prediction_window_size"]
    feature_window_size = config["feature_window_size"]
    min_confidence = config["min_confidence"]

    max_consecutive_agrees = config["max_consecutive_agrees"]
    accuracies = []
    first_corrects = []
    for file_path in test_files:
        reading_buffer = deque()
        print("Reading file {}".format(file_path))
        test_data = utils.read_csv(file_path, True)
        current_prediction = None
        consecutive_agrees = 0
        input_buffer = deque()
        num_prediction, correct = 0., 0.
        first_correct = None
        for row in test_data:
            reading_buffer.append(np.array([float(reading) for reading in row[3:]]))
            if len(reading_buffer) == feature_window_size:
                window_data = np.array(reading_buffer)
                feature_vector = np.array(feature_extraction(window_data))
                input_buffer.append(feature_vector)
                reading_buffer.popleft()
            if len(input_buffer) == prediction_window_size:
                input_feature_vector = np.concatenate(input_buffer)
                prediction_confidences = trained_model.predict_proba(input_feature_vector.reshape(1, -1))[0]
                prediction = np.argmax(prediction_confidences)
                confidence = prediction_confidences[prediction]
                input_buffer.popleft()
                if current_prediction is None or prediction == current_prediction:
                    if confidence > min_confidence:
                        consecutive_agrees += 1
                        if consecutive_agrees == max_consecutive_agrees:
                            predicted_move = reverse_label_map[prediction]
                            correct += evaluate(predicted_move, file_path)
                            first_correct = correct if first_correct is None else first_correct
                            consecutive_agrees = 0
                            num_prediction += 1
                else:
                    consecutive_agrees = 0
                current_prediction = prediction
        if num_prediction == 0.:
            accuracies.append(0.)
        else:
            accuracies.append(correct/num_prediction)
        if first_correct is None:
            first_corrects.append(0)
        else:
            first_corrects.append(first_correct)
    return np.mean(accuracies), np.mean(first_corrects)


if __name__ == "__main__":
    main()
