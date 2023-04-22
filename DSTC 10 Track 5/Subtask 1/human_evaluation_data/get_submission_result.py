import argparse
import logging
from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
from generate_submission_template import dialogue_test_set_iterator
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO)


def main(user_submission):
    dialogue_ids = []
    dialogue_annotations = []

    dialogue_id_to_dataset = {
        dial["dialogue_id"]: dial["dataset"] for dial in dialogue_test_set_iterator()
    }

    # for _, row in user_submission.iterrows():
    #     dialogue_id, eval_aspect = row["dialogue_id"].rsplit("|", 1)
    #     _, serial_id = dialogue_id.rsplit("_", 1)
    #     dialogue_id = f"{dialogue_id_to_dataset[dialogue_id]}_{serial_id}"
    #     row["dialogue_id"] = dialogue_id
    def get_dialogue_id(row):
        dialogue_id, eval_aspect = row["dialogue_id"].rsplit("|", 1)
        _, serial_id = dialogue_id.rsplit("_", 1)
        dialogue_id = f"{dialogue_id_to_dataset[dialogue_id]}_{serial_id}|{eval_aspect}"
        return dialogue_id

    user_submission["dialogue_id"] = user_submission.apply(get_dialogue_id, axis=1)

    for dial in dialogue_test_set_iterator():
        for eval_aspect, eval_aspect_annotations in dial.get("annotations", {}).items():
            if not eval_aspect_annotations:
                continue
            _, serial_id = dial["dialogue_id"].rsplit("_", 1)
            dialogue_id = f"{dialogue_id_to_dataset[dial['dialogue_id']]}_{serial_id}"
            # dialogue_ids.append(f"{dial['dialogue_id']}|{eval_aspect}")
            dialogue_ids.append(f"{dialogue_id}|{eval_aspect}")
            dialogue_annotations.append(np.mean(eval_aspect_annotations))
    annotations_scores = pd.DataFrame(
        {"dialogue_id": dialogue_ids, "score": dialogue_annotations}
    )
    assert len(user_submission) == len(
        annotations_scores
    ), f"Wrong number of entries in your submission, expected {len(annotations_scores)=} but got {len(user_submission)=}"

    user_submission_dialogue_id = sorted(list(user_submission["dialogue_id"]))
    annotation_dialogue_id = sorted(list(annotations_scores["dialogue_id"]))
    for x, y in zip(user_submission_dialogue_id, annotation_dialogue_id):
        assert (
            x == y
        ), f"Wrong dialogue_id in your submission, expected {y} from annotation but got {x} from your submission"

    user_submitted_scores = list(user_submission["score"])
    for item in user_submitted_scores:
        assert (
            type(item) is float and item >= 0
        ), f"Invalid score {item} in your submission file"
    final_df = user_submission.sort_values("dialogue_id").reset_index(drop=True)
    annotations_df = annotations_scores.sort_values("dialogue_id").reset_index(
        drop=True
    )
    final_df["annotation_score"] = annotations_df["score"]
    dialogue_id_list = list(final_df["dialogue_id"])
    dataset_list = [item.rsplit("_", 1)[0] for item in dialogue_id_list]
    dimension_list = [item.rsplit("|", 1)[1] for item in dialogue_id_list]
    dataset_dimension_list = [
        item.rsplit("_", 1)[0] + "|" + item.rsplit("|", 1)[1]
        for item in dialogue_id_list
    ]
    final_df["dataset_list"] = dataset_list
    final_df["dimension_list"] = dimension_list
    final_df["dataset_dimension_list"] = dataset_dimension_list
    dataset_dimension_set = set(list(dataset_dimension_list))
    dataset_dimension_score = {}
    for item in dataset_dimension_set:
        temp_df = final_df[final_df["dataset_dimension_list"] == item]
        dataset_dimension_score[item] = spearmanr(
            temp_df["score"], temp_df["annotation_score"]
        )[0]
    dataset_to_score = defaultdict(list)
    for k, v in dataset_dimension_score.items():
        d_name = k.split("|")[0]
        d_dimension = k.split("|")[1]
        dataset_to_score[d_name].append(v)
    dataset_to_avg_score = {k: np.mean(v) for k, v in dataset_to_score.items()}
    final_system_score = np.mean(list(dataset_to_avg_score.values()))
    results = {
        "score_by_dataset_dimension": dataset_dimension_score,
        "score_by_dataset": dataset_to_avg_score,
        "final_system_score": final_system_score,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission_name", type=str, default="submission_template.csv"
    )
    args = parser.parse_args()

    user_submission = pd.read_csv(args.submission_name)
    res = main(user_submission)
    pprint(res)
