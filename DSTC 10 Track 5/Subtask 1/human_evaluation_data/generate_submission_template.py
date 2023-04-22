import json
import logging
import random
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)

# ignore some test files since DSTC10 official uses filtered test files
IGNORE_FILES = {
    "dstc10-persona_eval.json",
    "dstc10-topical_eval.json",
}

TEST_FILES = {
    "dstc10-persona_clean_eval.json",
    "dstc10-topical_clean_eval.json",
    "esl_eval.json",
    "jsalt_eval.json",
    "ncm_eval.json",
}


_PATH = Path(__file__).parent.absolute()

DD_LLMEval_FILES = _PATH / "DailyDialog_LLMEval.json"


def files_iterator():
    f_list = Path(_PATH).glob("*.json")
    for f in f_list:
        if f.name in IGNORE_FILES:
            logging.warning(f"Skipping {f} (not used in DSTC10 official evaluation)")
            continue
        yield f


def test_files_iterator():
    f_list = Path(_PATH).glob("*.json")
    for f in f_list:
        if f.name in IGNORE_FILES or f.name not in TEST_FILES:
            logging.warning(
                f"Skipping {f} (not used in DSTC10 official TEST evaluation)"
            )
            continue
        yield f


def dev_files_iterator():
    f_list = Path(_PATH).glob("*.json")
    for f in f_list:
        if f.name in IGNORE_FILES or f.name in TEST_FILES:
            logging.warning(
                f"Skipping {f} (not used in DSTC10 official TEST evaluation)"
            )
            continue
        yield f


def dialogue_iterator():
    for f in files_iterator():
        d = json.load(open(f))
        for item in d:
            yield item


def dialogue_test_set_iterator():
    for f in test_files_iterator():
        d = json.load(open(f))
        for item in d:
            item["dataset"] = f.stem
            yield item


def dialogue_dev_set_iterator():
    for f in dev_files_iterator():
        d = json.load(open(f))
        for item in d:
            item["dataset"] = f.stem
            yield item


def dialogue_dd_llmeval_iterator():
    f = DD_LLMEval_FILES
    d = json.load(open(f))
    for item in d:
        item["dataset"] = f.stem
        yield item


if __name__ == "__main__":
    dialogue_ids = []
    dialogue_scores = []

    for dialogue in dialogue_iterator():
        for eval_aspect, eval_aspect_annotations in dialogue.get(
            "annotations", {}
        ).items():
            if not eval_aspect_annotations:
                continue
            dialogue_ids.append(f"{dialogue['dialogue_id']}|{eval_aspect}")
            dialogue_scores.append(random.random())
    df = pd.DataFrame({"dialogue_id": dialogue_ids, "score": dialogue_scores})
    df.to_csv("submission_template.csv", index=None)
