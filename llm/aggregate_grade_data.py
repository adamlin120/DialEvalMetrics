"""
Aggregate the grade data in cache/ and data/

All json files in the both cache/*/{name}.json and data/{name}.json

# convai2_grade_bert_ranker.json
# convai2_grade_dialogGPT.json
# convai2_grade_transformer_generator.json
# convai2_grade_transformer_ranker.json
to convai2_grade.json

# dailydialog_grade_transformer_generator.json
# dailydialog_grade_transformer_ranker.json
to dailydialog_grade.json

# empatheticdialogues_grade_transformer_generator.json
# empatheticdialogues_grade_transformer_ranker.json
to empatheticdialogues_grade.json
"""

import json
import os
from pathlib import Path


def main():
    cache_dir = Path("./cache")
    data_dir = Path("./data")

    for prompt_style in ["0-5", "0-100"]:
        for dataset in ["convai2", "dailydialog", "empatheticdialogues"]:
            dataset_data = {}
            # cache
            for model in ["bert_ranker", "dialogGPT", "transformer_generator", "transformer_ranker"]:
                if dataset == "convai2" and model not in ["bert_ranker", "dialogGPT", "transformer_generator", "transformer_ranker"]:
                    continue
                if dataset == "dailydialog" and model not in ["transformer_generator", "transformer_ranker"]:
                    continue
                if dataset == "empatheticdialogues" and model not in ["transformer_generator", "transformer_ranker"]:
                    continue
                cache_file = cache_dir / prompt_style / f"{dataset}_grade_{model}.json"

                with open(cache_file, "r") as f:
                    model_data = json.load(f)
                model_data = {f"{model}_{k}": v for k, v in model_data.items()}
                dataset_data.update(model_data)
            with open(cache_dir / prompt_style / f"{dataset}_grade.json", "w") as f:
                json.dump(dataset_data, f, indent=2)
                print(f"Saved {cache_dir / prompt_style / f'{dataset}_grade.json'}")

            # data
            for model in ["bert_ranker", "dialogGPT", "transformer_generator", "transformer_ranker"]:
                if dataset == "convai2" and model not in ["bert_ranker", "dialogGPT", "transformer_generator", "transformer_ranker"]:
                    continue
                if dataset == "dailydialog" and model not in ["transformer_generator", "transformer_ranker"]:
                    continue
                if dataset == "empatheticdialogues" and model not in ["transformer_generator", "transformer_ranker"]:
                    continue
                data_file = data_dir / f"{dataset}_grade_{model}.json"

                with open(data_file, "r") as f:
                    model_data = json.load(f)
                model_data = {f"{model}_{k}": v for k, v in model_data.items()}
                dataset_data.update(model_data)

            with open(data_dir / f"{dataset}_grade.json", "w") as f:
                json.dump(dataset_data, f, indent=2)
                print(f"Saved {data_dir / f'{dataset}_grade.json'}")







if __name__ == "__main__":
    main()