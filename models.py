from constants import *
import json
import pickle
from preprocessing import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from joblib import dump
from sklearn.metrics import mean_squared_error
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

SCALE = 1000


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
    print("Train svm")
    scaler_batch_size = 128
    train_size = len(X_train)
    scaler = MinMaxScaler()
    scaled_X_train = []
    i = 0
    while i < train_size:
        scaler.partial_fit(X_train[i: i + scaler_batch_size])
        i += scaler_batch_size

    i = 0
    while i < train_size:
        scaled_x_train_single = scaler.transform(X_train[i: i + scaler_batch_size])
        scaled_X_train.extend(scaled_x_train_single)
        i += scaler_batch_size

    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train)
    dump(svm, "svm.joblib")
    dump(scaler, "svm_scaler.joblib")
    return svm, scaler


def train_rf_batches(x_train_batches, y_train_batches, x_dev_batches, y_dev_batches):
    print("Train random forest")
    rfc = RandomForestClassifier(warm_start=True, n_estimators=10)
    num_train_batches = len(x_train_batches)
    feature_importance_batches = []
    for i in tqdm(range(num_train_batches), desc="Training batched RF"):  # 10 passes through the data
        X = x_train_batches[i]
        y = y_train_batches[i]
        rfc.fit(X, y)
        feature_importance_batches.append(rfc.feature_importances_)
        rfc.n_estimators += 10
    num_dev_batches = len(x_dev_batches)

    y_pred_rfc_batches = []
    for i in tqdm(range(num_dev_batches), desc="Evaluating batched RF"):
        X_dev = x_dev_batches[i]
        y_pred_rfc = rfc.predict(X_dev)
        y_pred_rfc_batches.append(y_pred_rfc)
    dump(rfc, "rf.joblib")
    return rfc, None


def train_knn(X_train, y_train, X_dev, y_dev):
    print("Train k nearest neighbors")
    scaler_batch_size = 128
    n_neighbors = 5
    train_size = len(X_train)
    scaler = MinMaxScaler()
    scaled_X_train = []
    i = 0
    while i < train_size:
        scaler.partial_fit(X_train[i: i + scaler_batch_size])
        i += scaler_batch_size

    i = 0
    while i < train_size:
        scaled_x_train_single = scaler.transform(X_train[i: i + scaler_batch_size])
        scaled_X_train.extend(scaled_x_train_single)
        i += scaler_batch_size

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(scaled_X_train, y_train)
    dump(knn, "knn.joblib")
    dump(scaler, "knn_scaler.joblib")
    return knn, scaler


def train_mlp(X_train, y_train, X_dev, y_dev):
    print("Train feed-forward neural network")
    num_hidden_1 = 500
    num_hidden_2 = 100
    scaler_batch_size = 128
    train_size = len(X_train)
    scaler = MinMaxScaler()
    scaled_X_train = []
    i = 0
    while i < train_size:
        scaler.partial_fit(X_train[i: i + scaler_batch_size])
        i += scaler_batch_size

    i = 0
    while i < train_size:
        scaled_x_train_single = scaler.transform(X_train[i: i + scaler_batch_size])
        scaled_X_train.extend(scaled_x_train_single)
        i += scaler_batch_size

    mlp = MLPClassifier(solver='adam', batch_size=128, alpha=1e-5,
                        hidden_layer_sizes=(num_hidden_1, num_hidden_2), random_state=1)
    mlp.fit(scaled_X_train, y_train)
    dump(mlp, "mlp.joblib")
    dump(scaler, "mlp_scaler.joblib")
    return mlp, scaler


# Returns an appended list of feature_extracted values
# E.g [Mean1, Mean2, ...., Min1, Min2, ....., Max1, Max2, ...., SD1, ....]


def extract_poly_fit(channel_features):
    poly_coeff = np.polyfit(range(len(channel_features)), channel_features, 1)
    return poly_coeff.flatten()


def extract_skewness(channel_features):
    skewness = skew(channel_features, axis=0)
    return skewness


def extract_average_amplitude_change(channel_features):
    amplitude_changes = []
    for i in range(0, len(channel_features)-1):
        amplitude_changes.append(np.abs(channel_features[i+1]-channel_features[i]))
    return np.mean(amplitude_changes, axis=0)


def extract_average_moving_rms(channel_features):
    moving_rms = []
    for i in range(0, len(channel_features)-1):
        curr_reading = channel_features[i+1]
        next_reading = channel_features[i]
        raw_values = mean_squared_error(curr_reading,
                                        next_reading, multioutput='raw_values')
        moving_rms.append(raw_values)
    average_moving_rms = np.mean(moving_rms, axis=0)
    return average_moving_rms


def feature_extraction(window_rows, important_idxs=None):
    feature_extracted_row = []
    feature_extracted_row.extend(window_rows[0])
    feature_extracted_row.extend(window_rows.mean(0))
    feature_extracted_row.extend(extract_average_moving_rms(window_rows))
    feature_extracted_row.extend(window_rows.min(0))
    feature_extracted_row.extend(window_rows.max(0))
    feature_extracted_row.extend(window_rows.std(0))
    # feature_extracted_row.extend(extract_poly_fit(window_rows))
    # feature_extracted_row.extend(extract_skewness(window_rows))
    # feature_extracted_row.extend(extract_average_amplitude_change(window_rows))
    important_features = [feature_extracted_row[i] for i in important_idxs] if important_idxs is not \
        None else feature_extracted_row
    return important_features


def build_window_data(X_train, y_train, X_dev, y_dev, prediction_window_size, feature_window_size, pad_size=1):
    train_size, dev_size = len(X_train), len(X_dev)

    X_window_train, y_window_train = [], []
    input_buffer, label_buffer = deque(), deque()
    for i in range(train_size-feature_window_size):
        reading_window = X_train[i: i + feature_window_size]
        reading_window_label = y_train[i + feature_window_size - 1]
        input_buffer.append(feature_extraction(reading_window))
        label_buffer.append(reading_window_label)
        if len(input_buffer) == prediction_window_size:  # record the features when the prediction buffer is full
            X_window_train.append(np.concatenate(np.array(input_buffer).astype(np.float16)))
            y_window_train.append(label_buffer[-1])
            for _ in range(pad_size):
                if len(input_buffer) > 0 and len(label_buffer) > 0:
                    input_buffer.popleft()
                    label_buffer.popleft()
                else:
                    break

    X_window_dev, y_window_dev = [], []
    input_buffer, label_buffer = deque(), deque()
    for i in range(dev_size - feature_window_size):
        reading_window = X_dev[i: i + feature_window_size]
        reading_window_label = y_dev[i + feature_window_size - 1]
        extracted_features = feature_extraction(reading_window)
        input_buffer.append(extracted_features)
        label_buffer.append(reading_window_label)
        if len(input_buffer) == prediction_window_size:  # record the features when the prediction buffer is full
            X_window_dev.append(np.concatenate(np.array(input_buffer)))
            y_window_dev.append(label_buffer[-1])
            if len(input_buffer) > 0 and len(label_buffer) > 0:
                input_buffer.popleft()
                label_buffer.popleft()
            else:
                break
    return X_window_train, y_window_train, X_window_dev, y_window_dev


def eval_model_batches(model, y_test_batches, y_pred_rfc_batches, num_test_batches):
    accuracies, f1_scores = [], []
    for i in range(num_test_batches):
        y_test = y_test_batches[i]
        y_pred_rfc = y_pred_rfc_batches[i]
        accuracy = accuracy_score(y_test, y_pred_rfc)
        f1 = f1_score(y_test, y_pred_rfc, average="macro")
        accuracies.append(accuracy)
        f1_scores.append(f1)
    eval_results = {
        "accuracy": np.mean(accuracies),
        "f1": np.mean(f1_scores)
    }
    print(eval_results)
    with open("{}_eval.json".format(model), "w") as f:
        json.dump(eval_results, f)


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
