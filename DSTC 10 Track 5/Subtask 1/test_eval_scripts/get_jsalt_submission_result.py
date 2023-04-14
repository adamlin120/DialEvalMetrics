import argparse
import glob
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def main(user_submission, save_name):
    jsalt_data = json.load(open("jsalt_eval.json"))
    did_list = [
        x["dialogue_id"]
        for x in jsalt_data
        if x["response"] != "" and x["response"] is not None
    ]
    user_submission["dialogue_id"] = user_submission["dialogue_id"].str.split("|").str[0]
    user_submission = user_submission[user_submission["dialogue_id"].isin(did_list)]
    user_submitted_scores = [float(item) for item in list(user_submission["score"])]
    for item in user_submitted_scores:
        assert item is not None

    dialogue_annotations = []
    for x in jsalt_data:
        if x["response"] != "" and x["response"] is not None:
            dialogue_annotations.append(np.mean(x["annotations"]["appropriateness"]))
    dict_1 = {1: did_list, 2: dialogue_annotations}
    annotations_scores = pd.DataFrame(dict_1)
    annotations_scores.columns = ["dialogue_id", "score"]
    final_df = user_submission.sort_values("dialogue_id").reset_index(drop=True)
    annotations_df = annotations_scores.sort_values("dialogue_id").reset_index(
        drop=True
    )

    assert len(user_submission) == len(
        annotations_scores
    ), f"wrong number of entries in your submission, expected {len(annotations_scores)} but got {len(user_submission)}"
    user_submission_dialogue_id = list(final_df["dialogue_id"])
    annotation_dialogue_id = list(annotations_df["dialogue_id"])
    for x, y in zip(user_submission_dialogue_id, annotation_dialogue_id):
        assert x == y, f"please check dialogue ids in your submission file, expected {x} but got {y}"
    final_df["annotation_score"] = annotations_df["score"]
    dialogue_id_list = list(final_df["dialogue_id"])
    score = spearmanr(final_df["annotation_score"], final_df["score"])[0]
    final_df.to_csv(save_name, index=None)
    return np.abs(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission_name", type=str, default="submission_template.csv"
    )
    parser.add_argument("--output_file_name", type=str, default="final_df.csv")
    args = parser.parse_args()

    user_submission = pd.read_csv(args.submission_name)
    res = main(user_submission, args.output_file_name)
    print(res)
