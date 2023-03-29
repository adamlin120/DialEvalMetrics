import json
import re
from pathlib import Path

import gspread
import pandas as pd
import scipy.stats


# ignore the following files
# convai2_grade_bert_ranker.json
# convai2_grade_dialogGPT.json
# convai2_grade_transformer_generator.json
# convai2_grade_transformer_ranker.json
# dailydialog_grade_transformer_generator.json
# dailydialog_grade_transformer_ranker.json
# empatheticdialogues_grade_transformer_generator.json
# empatheticdialogues_grade_transformer_ranker.json

IGNORE_FILES = [
    "convai2_grade_bert_ranker.json",
    "convai2_grade_dialogGPT.json",
    "convai2_grade_transformer_generator.json",
    "convai2_grade_transformer_ranker.json",
    "dailydialog_grade_transformer_generator.json",
    "dailydialog_grade_transformer_ranker.json",
    "empatheticdialogues_grade_transformer_generator.json",
    "empatheticdialogues_grade_transformer_ranker.json",
]


def calculate_correlation_coefficient(data_file, response_cache_file):
    # Read the data from the files
    with open(data_file, "r") as f:
        data = json.load(f)
    with open(response_cache_file, "r") as f:
        predict_score = json.load(f)

    gold_scores = {}
    id2system = {}
    for _id, datum in data.items():
        raw_score = datum["gold_score"]
        if isinstance(raw_score, dict):
            raw_score = raw_score["Overall"]
        score = float(raw_score)
        gold_scores[_id] = score

        # get the system name
        system = datum["system"]
        id2system[_id] = system

    predicted_scores = {}
    for _id, raw_score in predict_score.items():
        # match number like 3.2, 0.1, 5.0, 0, 10, 20, 50, 90, 100
        pattern = r"(\d+\.?\d*)"
        match = re.search(pattern, raw_score)
        if match:
            score = float(match.group(1))
        else:
            raise ValueError(
                f"Cannot parse the score: {raw_score}, at id: {id} in {response_cache_file}"
            )
        predicted_scores[_id] = score

    # sort the scores by id
    gold_scores_list, predicted_scores_list = [], []
    for _id in sorted(predicted_scores.keys()):
        gold_scores_list.append(gold_scores[_id])
        predicted_scores_list.append(predicted_scores[_id])

    # for each system, calculate the correlation coefficient
    system2scores = {}
    system2avg_scores = {}
    for _id, system in id2system.items():
        if system not in system2scores:
            system2scores[system] = ([], [])
        system2scores[system][0].append(gold_scores[_id])
        system2scores[system][1].append(predicted_scores[_id])

        # average the scores
        system2avg_scores[system] = (
            sum(system2scores[system][0]) / len(system2scores[system][0]),
            sum(system2scores[system][1]) / len(system2scores[system][1]),
        )

    if len(system2avg_scores) > 1:
        # calculate the correlation coefficient for all systems
        dialog_pearson_corr_coefficient, _ = scipy.stats.pearsonr(
            [v[0] for v in system2avg_scores.values()],
            [v[1] for v in system2avg_scores.values()],
        )
        dialog_spearman_corr_coefficient, _ = scipy.stats.spearmanr(
            [v[0] for v in system2avg_scores.values()],
            [v[1] for v in system2avg_scores.values()],
        )
    else:
        dialog_pearson_corr_coefficient, dialog_spearman_corr_coefficient = -2, -2

    # Compute the Pearson correlation coefficient
    turn_pearson_corr_coefficient, _ = scipy.stats.pearsonr(
        gold_scores_list, predicted_scores_list
    )

    # Compute the Spearman correlation coefficient
    turn_spearman_corr_coefficient, _ = scipy.stats.spearmanr(
        gold_scores_list, predicted_scores_list
    )

    return (
        turn_pearson_corr_coefficient,
        turn_spearman_corr_coefficient,
        dialog_pearson_corr_coefficient,
        dialog_spearman_corr_coefficient,
    )


def main():
    # iterate over all the response cache files under llm/cahce/ directory (e.g. llm/cache/0-100, llm/cache/0.0-5.0)
    # and calculate the correlation coefficient for each of them
    data_dir = Path("data")

    results = []

    for response_cache_file in Path("cache").glob("**/*.json"):
        # ignore the files
        if response_cache_file.name in IGNORE_FILES:
            continue

        data_file = data_dir / response_cache_file.name
        (
            turn_pearson_corr_coefficient,
            turn_spearman_corr_coefficient,
            dialog_pearson_corr_coefficient,
            dialog_spearman_corr_coefficient,
        ) = calculate_correlation_coefficient(data_file, response_cache_file)
        results.append(
            {
                "data_file": str(data_file)[5:],
                "prompt_style": response_cache_file.parts[1],
                "turn_pearson_corr_coefficient": turn_pearson_corr_coefficient,
                "turn_spearman_corr_coefficient": turn_spearman_corr_coefficient,
                "dialog_pearson_corr_coefficient": dialog_pearson_corr_coefficient,
                "dialog_spearman_corr_coefficient": dialog_spearman_corr_coefficient,
            }
        )

    # sort results by data_file
    results.sort(key=lambda x: x["data_file"])

    df = pd.DataFrame(results)
    # print in markdown table format
    print(df.to_markdown())
    # print in latex table format
    # round the float to 3 decimal places
    # data_file
    print(df.to_latex(index=False, float_format="{:0.3f}".format))
    # print in beautiful table format
    print(df.to_string(index=False))

    # print the results in the format of a table
    # float in 3 decimal places
    print(
        f"{'data file':<60}{'style':<10}{'Turn P':<10}{'Turn S':<10}{'System P':<10}{'System S':<10}"
    )
    for result in results:
        print(
            f"{result['data_file']:<60}{result['prompt_style']:<10}{result['turn_pearson_corr_coefficient']:<10.3f}{result['turn_spearman_corr_coefficient']:<10.3f}{result['dialog_pearson_corr_coefficient']:<10.3f}{result['dialog_spearman_corr_coefficient']:<10.3f}"
        )

    # write to google sheet - llm-dial-eval
    gc = gspread.service_account()
    gc.login()
    sh = gc.open("llm-dial-eval")
    worksheet = sh.worksheet("analysis")
    worksheet.clear()
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())


if __name__ == "__main__":
    main()
