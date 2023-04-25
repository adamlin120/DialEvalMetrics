from pathlib import Path

import numpy as np
import pandas as pd

HUMAN_EVAL_DIR_PATH = Path("./human_evaluation_data/")

TEST_SET_FILE = {
    "dstc10-persona_clean_eval.json",
    "dstc10-topical_clean_eval.json",
    "esl_eval.json",
    "jsalt_eval.json",
    "ncm_eval.json",
}

PRED_FILE_NAME_TO_CONFIG = {
    "submission_0-5": {
        "use_reference": "Yes",
        "score_range": "0-5",
    },
    "submission_0-5_all_noref": {
        "use_reference": "No",
        "score_range": "0-5",
    },
    "submission_0-5_dev": {
        "use_reference": "Yes",
        "score_range": "0-5",
    },
    "submission_0-100_all": {
        "use_reference": "Yes",
        "score_range": "0-100",
    },
    "submission_0-100_all_noref": {
        "use_reference": "No",
        "score_range": "0-100",
    },
    "submission_0-100_noref_DD-LLMEval": {
        "use_reference": "No",
        "score_range": "0-100",
    },
    "submission_0-5_noref_DD-LLMEval": {
        "use_reference": "No",
        "score_range": "0-5",
    },
    "submission_0-100_ablation_anthropic": {
        "use_reference": "Yes",
        "score_range": "0-100",
    },
    "submission_0-5_ablation_anthropic": {
        "use_reference": "Yes",
        "score_range": "0-5",
    },
    "submission_0-100_noref_ablation_anthropic": {
        "use_reference": "No",
        "score_range": "0-100",
    },
    "submission_0-5_noref_ablation_anthropic": {
        "use_reference": "No",
        "score_range": "0-5",
    },
    "submission_0-100_all_anthropic": {
        "use_reference": "Yes",
        "score_range": "0-100",
    },
    "submission_0-5_all_anthropic": {
        "use_reference": "Yes",
        "score_range": "0-5",
    },
    "submission_0-100_noref_all_anthropic": {
        "use_reference": "No",
        "score_range": "0-100",
    },
    "submission_0-5_noref_all_anthropic": {
        "use_reference": "No",
        "score_range": "0-5",
    },
    "submission_0-100_diallevel_anthropic": {
        "use_reference": "Yes",
        "score_range": "0-100",
    },
    "submission_0-5_diallevel_anthropic": {
        "use_reference": "Yes",
        "score_range": "0-5",
    },
    "submission_0-100_noref_ablation_anthropic-sampling": {
        "use_reference": "No",
        "score_range": "0-100",
    },
    "submission_0-5_noref_ablation_anthropic-sampling": {
        "use_reference": "No",
        "score_range": "0-5",
    },
    "submission_0-5_ablation_anthropic-ins": {
        "use_reference": "Yes",
        "score_range": "0-5",
    },
    "submission_0-100_ablation_anthropic-ins": {
        "use_reference": "Yes",
        "score_range": "0-100",
    },
    "submission_0-5_noref_ablation_anthropic-ins": {
        "use_reference": "No",
        "score_range": "0-5",
    },
    "submission_0-100_noref_ablation_anthropic-ins": {
        "use_reference": "No",
        "score_range": "0-100",
    },
    "submission_0-5_ablation_igpt": {
        "use_reference": "Yes",
        "score_range": "0-5",
    },
    "submission_0-100_ablation_igpt": {
        "use_reference": "Yes",
        "score_range": "0-100",
    },
    "submission_0-5_noref_ablation_igpt": {
        "use_reference": "No",
        "score_range": "0-5",
    },
    "submission_0-100_noref_ablation_igpt": {
        "use_reference": "No",
        "score_range": "0-100",
    },
}


def main():
    results = []

    testset_id_mapping = {}

    df = pd.DataFrame()
    # read all json files in human_evaluation_data except "dstc10-persona_eval.json" and "dstc10-topical_eval.json"
    for path in HUMAN_EVAL_DIR_PATH.glob("*.json"):
        if path.name in ["dstc10-persona_eval.json", "dstc10-topical_eval.json"]:
            continue
        single_df = pd.read_json(path)

        if path.name in TEST_SET_FILE:
            for i, row in single_df.iterrows():
                _id = row["dialogue_id"].rsplit("_", 1)[1]
                testset_id_mapping[row["dialogue_id"]] = f"{path.stem}_{_id}"
            single_df["dialogue_id"] = single_df["dialogue_id"].apply(
                lambda x: x.replace("dstc10-task5.1", path.stem)
            )

        df = pd.concat([df, single_df])

    # remove row without annotations
    df.dropna(subset=["annotations"], inplace=True)

    # check if there are any duplicate dialogue_ids
    if df["dialogue_id"].duplicated().any():
        raise ValueError(
            f"There are duplicate dialogue_ids in the human evaluation data.\n"
            f"Please check the data and remove the duplicates.\n"
            f"Duplicate dialogue_ids: {df['dialogue_id'].duplicated().sum()}"
        )

    # combine the average scores of the different annotators
    df["annotations_average"] = df["annotations"].apply(
        lambda x: {k: np.mean(v) for k, v in x.items()}
    )

    # set index to dialogue_id
    df.set_index("dialogue_id", inplace=True)

    # load predictions from submission_*.csv, ignore "submission_template.csv"
    for prediction_path in Path(HUMAN_EVAL_DIR_PATH).glob("submission_*.csv"):
        if prediction_path.name == "submission_template.csv":
            continue
        prediction_df = pd.read_csv(prediction_path)

        # dialogue_id is composed of dialogue_id and eval_aspect in format "dialogue_id|eval_aspect"
        prediction_df["eval_aspect"] = prediction_df["dialogue_id"].apply(
            lambda x: x.rsplit("|", 1)[1]
        )
        prediction_df["dialogue_id"] = prediction_df["dialogue_id"].apply(
            lambda x: x.rsplit("|", 1)[0]
        )

        # test set id mapping
        prediction_df["dialogue_id"] = prediction_df["dialogue_id"].apply(
            lambda x: testset_id_mapping.get(x, x)
        )

        # each aspect has its own column
        # pivot the dataframe to have one column per aspect
        # column name are appended with "_pred"
        # prediction_df = prediction_df.pivot(index="dialogue_id", columns="eval_aspect", values="score")
        prediction_df = prediction_df.pivot(
            index="dialogue_id", columns="eval_aspect", values="score"
        ).add_prefix("prediction_")

        # add prediction file path
        prediction_df["prediction_method"] = prediction_path.stem

        # check if the dialogue_ids in the prediction file are a subset of the dialogue_ids in the human evaluation data
        if not set(prediction_df.index).issubset(set(df.index)):
            raise ValueError(
                f"The dialogue_ids in the prediction file {prediction_path} are not a subset of the dialogue_ids in the human evaluation data.\n"
                f"{set(prediction_df.index).difference(set(df.index))}"
                f"{set(df.index).difference(set(prediction_df.index))}"
            )

        # join the prediction to the human evaluation data
        prediction_df = prediction_df.join(df, how="inner")

        # unset index
        prediction_df.reset_index(inplace=True)

        # split dialogue_id into dataset and _id in format "dialogue_id_id"
        prediction_df["_id"] = prediction_df["dialogue_id"].apply(
            lambda x: x.rsplit("_", 1)[1]
        )
        prediction_df["dataset"] = prediction_df["dialogue_id"].apply(
            lambda x: x.rsplit("_", 1)[0]
        )

        # set index to dataset and _id
        prediction_df.set_index(["dataset", "_id"], inplace=True)

        # group by dataset
        for dataset, dataset_df in prediction_df.groupby("dataset"):
            # remove columns that are all NaN
            dataset_df.dropna(axis=1, how="all", inplace=True)

            # assert all key in "annotations_average" are in the prediction
            # for _, row in dataset_df.iterrows():
            #     if not set(row["annotations_average"].keys()).issubset(set(row.index)):
            #         raise ValueError(f"The keys in annotations_average are not a subset of the keys in the prediction.\n"
            #                          f"{set(row['annotations_average'].keys()) - set(row.dropna().index)}")

            # get keys in "annotations_average" to use as columns in dataset_df
            keys = dataset_df["annotations_average"].iloc[0].keys()
            for key in keys:
                dataset_df[f"gt_{key}"] = dataset_df["annotations_average"].apply(
                    lambda x: x.get(key, np.nan)
                )

            # calculate correlation between prediction and ground truth
            # pearson, spearman, kendall
            for column in dataset_df.columns:
                if column.startswith("prediction_") and column != "prediction_method":
                    aspect_name = column.split("_", 1)[1]
                    pears_corr = dataset_df[column].corr(
                        dataset_df[f"gt_{aspect_name}"], method="pearson"
                    )
                    spear_corr = dataset_df[column].corr(
                        dataset_df[f"gt_{aspect_name}"], method="spearman"
                    )
                    kend_corr = dataset_df[column].corr(
                        dataset_df[f"gt_{aspect_name}"], method="kendall"
                    )
                    results.append(
                        {
                            "prediction_method": prediction_path.stem,
                            **PRED_FILE_NAME_TO_CONFIG[prediction_path.stem],
                            "dataset": dataset,
                            "aspect": aspect_name,
                            "pearson": pears_corr,
                            "spearman": spear_corr,
                            "kendall": kend_corr,
                        }
                    )

    results_df = pd.DataFrame(results)
    results_df.sort_values(
        by=["dataset", "aspect", "use_reference", "score_range"], inplace=True
    )
    print(results_df.to_latex(index=False))
    print(results_df.to_markdown())
    results_df.to_csv("human_evaluation_results.csv", index=False)


if __name__ == "__main__":
    main()
