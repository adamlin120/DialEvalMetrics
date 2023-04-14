import argparse
import glob
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def check_validity(item):
    flag = True
    try:
        int(item)
    except Exception as e:
        flag = False
    return flag


def main(user_submission, save_name):
    ncm_data = json.load(open("dstc10-persona_eval.json"))
    did_list = [
        x["dialogue_id"]
        for x in ncm_data
        if x["response"] != "" and x["response"] is not None and "annotations" in x
    ]
    import ipdb
    ipdb.set_trace()
    user_submission = user_submission[
        user_submission["dialogue_id"].isin(did_list)
    ].reset_index(drop=True)
    user_submission["score"] = pd.to_numeric(user_submission["score"], downcast="float")
    user_submission["score"] = user_submission["score"].fillna(0.0)
    user_submitted_scores = list(user_submission["score"])
    for idx, item in enumerate(user_submitted_scores):
        assert check_validity(item), "{}-{}".format(
            user_submission.loc[idx]["dialogue_id"], item
        )

    dialogue_annotations = []
    for x in ncm_data:
        if x["response"] != "" and x["response"] is not None and "annotations" in x:
            dialogue_annotations.append(np.mean(x["annotations"]["grammar"]))
    dict_1 = {1: did_list, 2: dialogue_annotations}
    annotations_scores = pd.DataFrame(dict_1)
    annotations_scores.columns = ["dialogue_id", "score"]
    final_df = user_submission.sort_values("dialogue_id").reset_index(drop=True)
    annotations_df = annotations_scores.sort_values("dialogue_id").reset_index(
        drop=True
    )

    assert len(user_submission) == len(
        annotations_scores
    ), "wrong number of entries in your submission"
    user_submission_dialogue_id = list(final_df["dialogue_id"])
    annotation_dialogue_id = list(annotations_df["dialogue_id"])
    for x, y in zip(user_submission_dialogue_id, annotation_dialogue_id):
        assert x == y, "please check dialogue ids in your submission file"
    final_df["annotation_score"] = annotations_df["score"]
    final_df.to_csv("final_df", index=None)
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
