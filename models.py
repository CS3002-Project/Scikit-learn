from constants import *
import json
import numpy as np
import pickle
from preprocessing import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from tqdm import tqdm
from joblib import dump, load
import itertools


def read_data(input_path):
    X, y = [], []

    with open(input_path, "r") as f:
        for i, line in enumerate(tqdm(f.readlines())):
            tokens = line.split("\t")
            label = tokens[-1].strip()
            readings = [x for i, x in enumerate(tokens[2: -1]) if i in INCLUDED_FEATURES]
            X.append(np.array(readings).astype(np.float))
            y.append(label)

    return X, y


def train_svm(X_train, y_train, X_dev, y_dev):
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_dev)
    eval_model("svm", y_dev, y_pred_svm)

    with open("svm.p", "wb") as f:
        pickle.dump(svm, f)


def train_rf(X_train, y_train, X_dev, y_dev):
    rfc = RandomForestClassifier(n_estimators=50)
    rfc.fit(X_train, y_train)
    y_pred_rfc = rfc.predict(X_dev)
    eval_model("rf", y_dev, y_pred_rfc)

    dump(rfc, 'rf.joblib')


def train_svm_seq(X_train, y_train, X_dev, y_dev, window_size=5):
    X_window_train, y_window_train, X_window_dev, y_window_dev \
        = build_window_data(X_train, y_train, X_dev, y_dev, window_size)
    X_window_train = [np.concatenate(x) for x in X_window_train]
    X_window_dev = [np.concatenate(x) for x in X_window_dev]
    train_svm(X_window_train, y_window_train, X_window_dev, y_window_dev)


def train_svm_extract(X_train, y_train, X_dev, y_dev, window_size=5):
    X_window_train, y_window_train, X_window_dev, y_window_dev \
        = build_window_data(X_train, y_train, X_dev, y_dev, window_size)
    X_window_train_extracted, X_window_dev_extracted = build_window_extract_data(X_window_train, X_window_dev)
    train_svm(X_window_train_extracted, y_window_train, X_window_dev_extracted, y_window_dev)


def train_rf_seq(X_train, y_train, X_dev, y_dev, window_size=5):
    X_window_train, y_window_train, X_window_dev, y_window_dev \
        = build_window_data(X_train, y_train, X_dev, y_dev, window_size)
    X_window_train = [np.concatenate(x) for x in X_window_train]
    X_window_dev = [np.concatenate(x) for x in X_window_dev]
    train_rf(X_window_train, y_window_train, X_window_dev, y_window_dev)


def train_rf_extract(X_train, y_train, X_dev, y_dev, window_size=5):
    X_window_train, y_window_train, X_window_dev, y_window_dev \
        = build_window_data(X_train, y_train, X_dev, y_dev, window_size)
    X_window_train_extracted, X_window_dev_extracted = build_window_extract_data(X_window_train, X_window_dev)
    train_rf(X_window_train_extracted, y_window_train, X_window_dev_extracted, y_window_dev)


def train_rf_extract_window(X_train, y_train, X_dev, y_dev, window_size=5):
    X_window_train, y_window_train, X_window_dev, y_window_dev \
        = build_window_data(X_train, y_train, X_dev, y_dev, window_size)
    X_train_extract, X_dev_extract = build_window_extract_data(X_window_train, X_window_dev)
    X_train_extract_window, y_train_extract_window, X_dev_extract_window, y_dev_extract_window = \
        build_window_data(X_train_extract, y_window_train, X_dev_extract, y_window_dev, window_size)
    X_train_extract_window_concat = [np.concatenate(x) for x in X_train_extract_window]
    X_dev_extract_window_concat = [np.concatenate(x) for x in X_dev_extract_window]
    train_rf(X_train_extract_window_concat, y_train_extract_window, X_dev_extract_window_concat, y_dev_extract_window)

# Returns an appended list of feature_extracted values
# E.g [Mean1, Mean2, ...., Min1, Min2, ....., Max1, Max2, ...., SD1, ....]


def feature_extraction(window_rows):
    feature_extracted_row = []
    feature_extracted_row.extend(list(itertools.chain.from_iterable(window_rows)))
    feature_extracted_row.extend(window_rows.mean(0))
    feature_extracted_row.extend(window_rows.min(0))
    feature_extracted_row.extend(window_rows.max(0))
    feature_extracted_row.extend(window_rows.std(0))

    return feature_extracted_row


def build_window_extract_data(X_window_train, X_window_dev):
    X_window_train_extracted, X_window_train_dev = [], []
    for window in X_window_train:
        mean = extract_mean(window)
        variance = extract_variance(window)
        poly_fit = extract_poly_fit(window)
        skewness = extract_poly_fit(window)
        average_amplitude_change = extract_average_amplitude_change(window)
        X_window_train_extracted.append(np.concatenate([mean, variance, poly_fit,
                                                        skewness, average_amplitude_change]))
    for window in X_window_dev:
        mean = extract_mean(window)
        variance = extract_variance(window)
        poly_fit = extract_poly_fit(window)
        skewness = extract_poly_fit(window)
        average_amplitude_change = extract_average_amplitude_change(window)
        X_window_train_dev.append(np.concatenate([mean, variance, poly_fit,
                                                  skewness, average_amplitude_change]))
    return X_window_train_extracted, X_window_train_dev


def build_window_data(X_train, y_train, X_dev, y_dev, window_size):
    train_size, dev_size = len(X_train), len(X_dev)
    X_window_train, y_window_train = [], []
    # flag = 0
    for i in range(window_size-1, train_size):
        feature_extracted_row = feature_extraction(X_train[i + 1 - window_size: i + 1])
        X_window_train.append(feature_extracted_row)
        # X_window_train.append(X_train[i + 1 - window_size: i + 1])
        # print(type(X_train[i + 1 - window_size: i + 1]))
        # print(X_train[i + 1 - window_size: i + 1])
        # print(X_train[i + 1 - window_size: i + 1].mean(0))
        # print(X_train[i + 1 - window_size: i + 1].min(0))
        # print(X_train[i + 1 - window_size: i + 1].max(0))
        # print(X_train[i + 1 - window_size: i + 1].std(0))
        #sys.exit()
        y_window_train.append(y_train[i])
    X_window_dev, y_window_dev = [], []
    for i in range(window_size - 1, dev_size):
        feature_extracted_row = feature_extraction(X_dev[i + 1 - window_size: i + 1])
        X_window_dev.append(feature_extracted_row)
        # X_window_dev.append(X_dev[i + 1 - window_size: i + 1])
        y_window_dev.append(y_dev[i])
    return X_window_train, y_window_train, X_window_dev, y_window_dev


def eval_model(model, y_test, y_pred_rfc):
    accuracy = accuracy_score(y_test, y_pred_rfc)
    precision = precision_score(y_test, y_pred_rfc, average="macro")
    recall = recall_score(y_test, y_pred_rfc, average="macro")
    f1 = f1_score(y_test, y_pred_rfc, average="macro")
    eval_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    with open("{}_eval.json".format(model), "w") as f:
        json.dump(eval_results, f)


def train(model, X, y):
    data_size = len(X)
    test_size = int(data_size * TEST_SIZE)
    X_train, y_train = X[:-test_size], y[:-test_size]
    X_dev, y_dev = X[-test_size:], y[-test_size:]
    X_train, X_dev = standardise_dataset(X_train, X_dev)

    if model == "svm":
        train_svm(X_train, y_train, X_dev, y_dev)
    elif model == "rf":
        train_rf(X_train, y_train, X_dev, y_dev)
    elif model == "svm_seq":
        train_svm_seq(X_train, y_train, X_dev, y_dev)
    elif model == "rf_seq":
        train_rf_seq(X_train, y_train, X_dev, y_dev)
    elif model == "svm_extract":
        train_svm_extract(X_train, y_train, X_dev, y_dev)
    elif model == "rf_extract":
        train_rf_extract(X_train, y_train, X_dev, y_dev)
    elif model == "rf_extract_window":
        train_rf_extract_window(X_train, y_train, X_dev, y_dev)


if __name__ == "__main__":
    X, y = read_data("subject4_ideal.log")
    train("rf_extract_window", X, y)