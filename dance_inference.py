import numpy as np
import os
import utils
from collections import deque


reverse_label_map = {
        0: "bunny",
        1: "cowboy",
        2: "handmotor",
        3: "rocket",
        4: "tapshoulder",
        5: "hunchback"
}


def run(test_dir, model_file):
    window_size = 10  # change window size need to retrain the model
    max_consecutive_agrees = 10

    model = utils.load_from_pickle(model_file)
    for file_name in os.listdir(test_dir):
        reading_buffer = deque()
        file_path = os.path.join(test_dir, file_name)
        print("Reading file {}".format(file_path))
        test_data = utils.read_csv(file_path, True)
        current_prediction = None
        consecutive_agrees = 0

        for row in test_data:
            reading_buffer.append(np.array([float(reading) for reading in row[3:]]))
            if len(reading_buffer) == window_size:
                feature_vector = np.concatenate(reading_buffer).reshape(1, -1)
                prediction = model.predict(feature_vector)[0]
                if current_prediction is None or prediction == current_prediction:
                    consecutive_agrees += 1
                    if consecutive_agrees == max_consecutive_agrees:
                        print("Predicted move is {}".format(reverse_label_map[prediction]))
                        consecutive_agrees = 0
                else:
                    consecutive_agrees = 0
                current_prediction = prediction
                reading_buffer.popleft()


if __name__ == "__main__":
    TEST_DIR = "test"
    MODEL_FILE = "rf.p"
    run(TEST_DIR, MODEL_FILE)