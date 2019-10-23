import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

mode = "raw"

model_name = "mlp_{}".format(mode)
test_file = "subject4_ideal.log"

with open("{}.p".format(model_name), "rb") as f:
    model = pickle.load(f)


def eval_model(y_test, y_pred_rfc):
    accuracy = accuracy_score(y_test, y_pred_rfc)
    precision = precision_score(y_test, y_pred_rfc, average="macro")
    recall = recall_score(y_test, y_pred_rfc, average="macro")
    f1 = f1_score(y_test, y_pred_rfc, average="macro")
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))


def load_raw_test_data(test_file_path):
    start_feature_idx = 0
    end_feature_idx = 117

    X_test, y_test = [], []

    with open(test_file_path, "r") as f:
        for i, line in enumerate(tqdm(f.readlines())):
            tokens = line.split("\t")
            label = tokens[-1].strip()
            readings = tokens[2: 119][start_feature_idx: end_feature_idx]
            X_test.append(np.array(readings).astype(np.float))
            y_test.append(label)

    return X_test, y_test


def raw_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    eval_model(y_test, y_pred)


X_test, y_test = load_raw_test_data(test_file)
raw_eval(model, X_test, y_test)


def extracted_eval(model, X_test, y_test):
    pass
