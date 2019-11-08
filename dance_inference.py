import numpy as np
import os
import utils
from collections import deque
from joblib import load
from models import feature_extraction


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
                print("{} is mistaken as {}".format(k, predicted_move))


def run(test_dir, model_file):
    prediction_window_size = 24
    feature_window_size = 20

    max_consecutive_agrees = 10
    model = load(model_file)
    for file_name in os.listdir(test_dir):
        reading_buffer = deque()
        file_path = os.path.join(test_dir, file_name)
        print("Reading file {}".format(file_path))
        test_data = utils.read_csv(file_path, True)
        current_prediction = None
        consecutive_agrees = 0

        input_buffer = deque()

        for row in test_data:
            reading_buffer.append(np.array([float(reading) for reading in row[3:]]))
            if len(reading_buffer) == feature_window_size:
                window_data = np.array(reading_buffer)
                feature_vector = np.array(feature_extraction(window_data))
                input_buffer.append(feature_vector)
                reading_buffer.popleft()
            if len(input_buffer) == prediction_window_size:
                input_feature_vector = np.concatenate(input_buffer)
                prediction = model.predict(input_feature_vector.reshape(1, -1))[0]
                input_buffer.popleft()
                if current_prediction is None or prediction == current_prediction:
                    consecutive_agrees += 1
                    if consecutive_agrees == max_consecutive_agrees:
                        predicted_move = reverse_label_map[prediction]
                        evaluate(predicted_move, file_name)
                        consecutive_agrees = 0
                else:
                    consecutive_agrees = 0
                current_prediction = prediction


if __name__ == "__main__":
    TEST_DIR = "test"
    MODEL_FILE = "rf.joblib"
    run(TEST_DIR, MODEL_FILE)