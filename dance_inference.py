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
        10: "logout",
    # 99: "idle",
}


def evaluate(predicted_move, input_file):
    if predicted_move not in input_file:
        for k in reverse_label_map.values():
            if k in input_file:
                print("{} is mistaken as {}".format(k, predicted_move))
                return False
    return True


def run(test_dir, model_file):
    prediction_window_size = 18
    feature_window_size = 10
    min_confidence = 0.8

    max_consecutive_agrees = 10 
    model = load(model_file)
    for file_name in os.listdir(test_dir):
        reading_buffer = deque()
        file_path = os.path.join(test_dir, file_name)
        print("Reading file {}".format(file_path))
        test_data = utils.read_csv(file_path, True)
        current_prediction = None
        consecutive_agrees = 0
        result = None
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
                prediction_confidences = model.predict_proba(input_feature_vector.reshape(1, -1))[0]
                prediction = np.argmax(prediction_confidences)
                confidence = prediction_confidences[prediction]
                input_buffer.popleft()
                if current_prediction is None or prediction == current_prediction:
                    if confidence > min_confidence:
                        consecutive_agrees += 1
                        if consecutive_agrees == max_consecutive_agrees:
                            predicted_move = reverse_label_map[prediction]
                            result = evaluate(predicted_move, file_name)
                            consecutive_agrees = 0
                            break
                else:
                    consecutive_agrees = 0
                current_prediction = prediction
        print(result)

if __name__ == "__main__":
    TEST_DIR = "data"
    MODEL_FILE = "rf.joblib"
    run(TEST_DIR, MODEL_FILE)
