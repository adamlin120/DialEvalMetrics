"""
Plot the score distribution of predicted scores
"""

import re
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    """Main function"""
    # Read the data
    for path in Path("./cache").glob("**/*.json"):
        dataset_name = path.stem
        prompt_style = path.parent.stem
        with open(path, "r") as file:
            # id -> score
            data = json.load(file)
        # extract the scores using regex
        # scores like 0, 1, 2, 10, 100, 0.0, 0.5, 2.4, 5.0
        scores =  [float(score) for score in re.findall(r"\d+\.?\d*", str(data.values()))]

        # Plot the histogram
        plt.hist(scores, bins=20)
        plt.title(f"{dataset_name} ({prompt_style})")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.savefig(f"./plots/{dataset_name}_{prompt_style}.png")
        plt.close()


if __name__ == "__main__":
    main()
