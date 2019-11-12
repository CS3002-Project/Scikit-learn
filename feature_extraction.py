import json
import utils


def extract_feature():
    threshold = 0.0
    imp_indices = []
    fi_input = "rf_feat_imp.json"
    fi_output = "feat_imp_idx.txt"
    with open(fi_input, "r") as f:
        fi_list = json.load(f)
    for i, score in enumerate(fi_list):
        if float(score) > threshold:
            imp_indices.append(i)
    utils.save_list_as_text(imp_indices, fi_output)


if __name__ == "__main__":
    extract_feature()