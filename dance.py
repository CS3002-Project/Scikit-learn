import numpy as np
import os
from run_preprocess import label_map, NAME_MAP
from models import build_window_data, train_mlp, feature_extraction, train_rf_batches, train_knn, train_svm
import utils
import copy
import random
import time
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import joblib


important_idxs = None
#important_idxs = [int(x) for x in utils.load_text_as_list("feat_imp_idx.txt")]


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
        10: "logout",
    # 99: "idle",
}


def evaluate(predicted_move, input_file):
    if predicted_move not in input_file:
        for k in reverse_label_map.values():
            if k in input_file:
                print("mispredicted move:{} ".format(predicted_move))
                return 0
    return 1


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
                # Change this back please
                train_files.extend([f for f in files[:1] if f not in test_files])
        iter_train_files.append(train_files)
        iter_test_files.append(test_files)

    return iter_train_files, iter_test_files


def read_data_from_file(file_path, test_size):
    x_sample = []
    label = label_map(file_path)
    x_raw_data = utils.read_csv(file_path, load_header=True, delimiter=",")
    for row in x_raw_data:
        x_snapshot = np.array([float(reading) for reading in row[3:]])
        x_sample.append(x_snapshot)
    y_sample = [label] * len(x_sample)
    test_num = int(len(x_sample) * test_size)
    x_train = np.array(x_sample[:-test_num])
    y_train = np.array(y_sample[:-test_num])
    x_dev = np.array(x_sample[-test_num:])
    y_dev = np.array(y_sample[-test_num:])
    return x_train, y_train, x_dev, y_dev


def prepare_data(x_train_single, y_train_single, x_dev_single, y_dev_single, config):
    prediction_window_size = config["prediction_window_size"]
    feature_window_size = config["feature_window_size"]
    x_window_train, y_window_train, x_window_dev, y_window_dev \
        = build_window_data(x_train_single, y_train_single, x_dev_single, y_dev_single,
                            prediction_window_size, feature_window_size, config["pad_size"])
    return x_window_train, y_window_train, x_window_dev, y_window_dev


def preprocess_data(train_files, config):
    x_train, y_train, x_dev, y_dev = [], [], [], []
    for file in tqdm(train_files, desc="preprocessing_data"):
        cache_file_path = file.replace("data", "cache").replace(".csv", "_fw{}_pw{}_pad{}_ts{}_cache.pickle".format(
            config["feature_window_size"], config["prediction_window_size"], config["pad_size"], config["test_size"]))
        if os.path.exists(cache_file_path):
            print("Load cache from {}".format(cache_file_path))
            x_window_train, y_window_train, x_window_dev, y_window_dev = utils.load_from_pickle(cache_file_path)
        else:
            x_train_single, y_train_single, x_dev_single, y_dev_single = read_data_from_file(file, config["test_size"])
            x_window_train, y_window_train, x_window_dev, y_window_dev = \
                prepare_data(x_train_single, y_train_single, x_dev_single, y_dev_single, config)
            print("Save cache to {}".format(cache_file_path))
            utils.save_to_pickle((x_window_train, y_window_train, x_window_dev, y_window_dev), cache_file_path)
        x_train.extend(x_window_train)
        y_train.extend(y_window_train)
        x_dev.extend(x_window_dev)
        y_dev.extend(y_window_dev)
    return x_train, y_train, x_dev, y_dev


def main(config):
    num_iters = 1
    batch_size = 4096
    data_dir = "data"
    data_limits = [0.2, 0.4, 0.6, 0.8, 1.0]
    iter_train_files, iter_test_files = split_train_test(data_dir, num_iters)
    model_names = ["rf", "mlp", "knn", "svm"]
    for limit in data_limits:
        model_bench_marks = {
            model: {
                "accuracies": [],
                "prediction_times": []
            } for model in model_names
        }
        print("Training and testing with limit {}".format(limit))
        for i in range(num_iters):
            train_files, test_files = iter_train_files[i], iter_test_files[i]
            utils.save_list_as_text(test_files, "test_files.txt")
            x_train, y_train, x_dev, y_dev = preprocess_data(train_files, config)
            x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.0001)
            limited_train_size, limited_test_size = int(len(x_train) * limit), int(len(x_dev) * limit)
            x_train, y_train = x_train[:limited_train_size], y_train[:limited_train_size]
            x_dev, y_dev = x_dev[:limited_test_size], y_dev[:limited_test_size]
            x_train_batches, y_train_batches = divide_into_batches(x_train, y_train, batch_size)
            x_dev_batches, y_dev_batches = divide_into_batches(x_dev, y_dev, batch_size)
            for model_name in model_names:
                if model_name == "rf":
                    trained_model, scaler = train_rf_batches(x_train_batches, y_train_batches, x_dev_batches, y_dev_batches)
                elif model_name == "mlp":
                    trained_model, scaler = train_mlp(x_train, y_train, x_dev, y_dev)
                elif model_name == "knn":
                    trained_model, scaler = train_knn(x_train, y_train, x_dev, y_dev)
                elif model_name == "svm":
                    trained_model, scaler = train_svm(x_train, y_train, x_dev, y_dev)
                else:
                    raise ValueError("Unsupported model {}".format(model_name))
                iter_accuracy, iter_prediction_time = test(trained_model, scaler, test_files, config)
                model_bench_marks[model_name]["accuracies"].append(iter_accuracy)
                model_bench_marks[model_name]["prediction_times"].append(iter_prediction_time)
        for model, results in model_bench_marks.items():
            model_bench_marks[model]["accuracies"] = np.mean(model_bench_marks[model]["accuracies"])
            model_bench_marks[model]["prediction_times"] = np.mean(model_bench_marks[model]["prediction_times"])
        with open("benchmark_{}.json".format(limit), "w") as f:
            json.dump(model_bench_marks, f)


def test(trained_model, scaler, test_files, config):
    prediction_window_size = config["prediction_window_size"]
    feature_window_size = config["feature_window_size"]
    min_confidence = config["min_confidence"]
    pad_size = config["pad_size"]
    lower_min_confidence = config["lower_min_confidence"]
    num_correct, num_predictions, prediction_sec = 0, 0, 0
    for file_path in test_files:
        reading_buffer, input_buffer = [], []
        print("Reading file {}".format(file_path))
        test_data = utils.read_csv(file_path, True)
        current_prediction = None
        first_correct = None
        for row in test_data:
            reading_buffer.append(np.array([float(reading) for reading in row[3:]]))
            if len(reading_buffer) == feature_window_size:
                window_data = np.array(reading_buffer)
                feature_vector = np.array(feature_extraction(window_data))
                input_buffer.append(feature_vector)
                reading_buffer = reading_buffer[pad_size:]
            if len(input_buffer) == prediction_window_size:
                input_feature_vector = np.concatenate(input_buffer)
                before_prediction = time.time()
                if scaler is not None:
                    input_vector = scaler.transform(input_feature_vector.reshape(1, -1))
                else:
                    input_vector = input_feature_vector.reshape(1, -1)
                prediction_confidences = trained_model.predict_proba(input_vector)[0]
                prediction_sec += time.time() - before_prediction
                num_predictions += 1
                prediction = np.argmax(prediction_confidences)
                confidence = prediction_confidences[prediction]

                if confidence > min_confidence:  # prediction is taken
                    predicted_move = reverse_label_map[prediction]
                    result = evaluate(predicted_move, file_path)
                    if first_correct is None and result == 1:
                        print("First prediction is correct from high threshold")
                        num_correct += 1
                    elif first_correct is None:
                        print("First prediction is wrong from high threshold")
                    break

                elif confidence > lower_min_confidence:  # prediction is taken
                    if prediction == current_prediction:
                        predicted_move = reverse_label_map[prediction]
                        result = evaluate(predicted_move, file_path)
                        if result == 1:
                            print("First prediction is correct from low threshold")
                            num_correct += 1
                        else:
                            print("First prediction is wrong from low threshold")
                        break

                    current_prediction = prediction
                input_buffer = []

    accuracy = num_correct / len(test_files)
    time_per_prediction = prediction_sec / num_predictions
    print("Accuracy {}".format(accuracy))
    print("Time per prediction {}".format(time_per_prediction))
    return accuracy, time_per_prediction


def parse_args():
    parser = argparse.ArgumentParser(description='Community FAQ pre-processing')
    parser.add_argument('-s', '--simulate', action='store_true', help='Only simulate')
    return parser.parse_args()


if __name__ == "__main__":
    p_args = parse_args()
    config = {
        "prediction_window_size": 24,
        "feature_window_size": 10,
        "min_confidence": 0.5,
        "lower_min_confidence": 0.1,
        "model_type": "rf",
        "min_consecutive_agrees": 1,
        "test_size": 0.1,
        "pad_size": 5,
        "mlp": False,
        "mlp_limit": 100000
    }
    if p_args.simulate:
        test_files = utils.load_text_as_list("test_files.txt")
        trained_rf = joblib.load("rf.joblib")
        trained_mlp = joblib.load("mlp.joblib")
        test(trained_rf, test_files, config)
    else:
        main(config)
