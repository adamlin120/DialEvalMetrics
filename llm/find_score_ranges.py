"""
Show the score ranges and distributions for each data under ./data/ directory
"""


import json
from pathlib import Path

import pandas as pd


def main():
    data_dir = Path("data")

    for data_file in data_dir.glob("*.json"):
        with open(data_file, "r") as f:
            data = json.load(f)

        scores = []
        for _, datum in data.items():
            raw_score = datum["gold_score"]
            if isinstance(raw_score, dict):
                raw_score = raw_score["Overall"]
            score = float(raw_score)
            scores.append(score)

        # print the score range and distribution in the same format as llm/analysis.py
        print(f"{data_file.name}: {min(scores)}-{max(scores)}")
        # print(pd.Series(scores).value_counts().sort_index())


if __name__ == "__main__":
    main()
