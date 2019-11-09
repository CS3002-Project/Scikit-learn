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
    rfc = RandomForestClassifier(n_estimators=150)
    rfc.fit(X_train, y_train)
    y_pred_rfc = rfc.predict(X_dev)
    eval_model("rf", y_dev, y_pred_rfc)
    dump(rfc, 'rf.joblib')


def train_mlp(X_train, y_train, X_dev, y_dev):
    num_hidden_1 = 100
    num_hidden_2 = 50
    scaler = MinMaxScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_dev = scaler.transform(X_dev)
    
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(num_hidden_1, num_hidden_2), random_state=1)
    mlp.fit(scaled_X_train, y_train)
    y_pred_mlp = mlp.predict(scaled_X_dev)
    eval_model("mlp", y_dev, y_pred_mlp)
    dump(mlp, "mlp.joblib")


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
        moving_rms.append(mean_squared_error(channel_features[i+1],
                                             channel_features[i], multioutput='raw_values'))
    return np.mean(moving_rms, axis=0)


def feature_extraction(window_rows):
    feature_extracted_row = []
    feature_extracted_row.extend(window_rows[0])
    feature_extracted_row.extend(window_rows.mean(0))
    feature_extracted_row.extend(extract_average_moving_rms(window_rows))
    feature_extracted_row.extend(window_rows.min(0))
    feature_extracted_row.extend(window_rows.max(0))
    feature_extracted_row.extend(window_rows.std(0))
    feature_extracted_row.extend(extract_poly_fit(window_rows))
    feature_extracted_row.extend(extract_skewness(window_rows))
    feature_extracted_row.extend(extract_average_amplitude_change(window_rows))

    return feature_extracted_row


def build_window_data(X_train, y_train, X_dev, y_dev, prediction_window_size, feature_window_size):
    train_size, dev_size = len(X_train), len(X_dev)

    X_window_train, y_window_train = [], []
    input_buffer, label_buffer = deque(), deque()
    for i in range(train_size-feature_window_size):
        reading_window = X_train[i: i + feature_window_size]
        reading_window_label = y_train[i + feature_window_size - 1]
        input_buffer.append(feature_extraction(reading_window))
        label_buffer.append(reading_window_label)
        if len(input_buffer) == prediction_window_size:  # record the features when the prediction buffer is full
            X_window_train.append(np.concatenate(np.array(input_buffer)))
            y_window_train.append(label_buffer[-1])
            input_buffer.popleft()
            label_buffer.popleft()

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
            input_buffer.popleft()
            label_buffer.popleft()
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


if __name__ == "__main__":
    X, y = read_data("subject4_ideal.log")
    train("rf_extract_window", X, y)
