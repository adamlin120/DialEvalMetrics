import logging
import os
from parser import get_pydantic_output_parser
from typing import List
from argparse import ArgumentParser

import pandas as pd
from openai.error import Timeout
from tqdm import tqdm
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate

from human_evaluation_data.generate_submission_template import dialogue_test_set_iterator, dialogue_dev_set_iterator, dialogue_iterator as dialogue_iterator_all
from prompt import turn_eval_template, turn_noref_eval_template,dialogue_eval_template, score_config

logging.basicConfig(level=logging.INFO)


def run_eval_chain(
    score_aspects: List[str],
    score_dtype: type,
    score_min: float,
    score_max: float,
    human_template: str,
    model_name: str = "gpt-3.5-turbo-0301",
    **prompt_kwargs,
):
    chat = ChatOpenAI(temperature=0, model_name=model_name, max_retries=1)

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

    chain = LLMChain(
        prompt=chat_prompt,
        llm=chat,
        # verbose=True,
    )

    parser = get_pydantic_output_parser(
        field_names={aspect: aspect for aspect in score_aspects},
        score_type=score_dtype,
        score_range=(score_min, score_max),
    )

    output = chain.run(
        format_instructions=parser.get_format_instructions(),
        score_min=score_min,
        score_max=score_max,
        **prompt_kwargs,
    )
    try:
        scores = parser.parse(output)
    except:
        logging.warning("Failed to parse output: %s" % output)
        scores = None

    return output, scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--score_type", type=str, default="0-5", choices=["0-5", "0-100"])
    parser.add_argument("--data_set", type=str, required=True, choices=["dev", "test", "all"])
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--reference_free", action="store_true")
    args = parser.parse_args()

    score_type = args.score_type

    if args.data_set == "dev":
        dialogue_iterator = dialogue_dev_set_iterator
    elif args.data_set == "test":
        dialogue_iterator = dialogue_test_set_iterator
    elif args.data_set == "all":
        dialogue_iterator = dialogue_iterator_all
    else:
        raise ValueError("Unknown data set %s" % args.data_set)

    turns_or_dialogues = sorted(list(dialogue_iterator()), key=lambda x: x["dialogue_id"])

    logging.info("Found total %d turns or dialogues" % len(turns_or_dialogues))

    df = pd.read_csv(args.save_path) if os.path.exists(args.save_path) else None
    already_evaluated = set(df["dialogue_id"].values) if df is not None else set()
    results = df.to_dict(orient="records") if df is not None else []

    for _i, datum in tqdm(enumerate(turns_or_dialogues), total=len(turns_or_dialogues)):
        if "annotations" not in datum:
            continue

        aspects = list(datum["annotations"].keys())

        if all(f"{datum['dialogue_id']}|{aspect}" in already_evaluated for aspect in aspects):
            # logging.info("Already evaluated all aspects for %s" % datum["dialogue_id"])
            continue

        is_turn_level = "response" in datum
        has_reference = "reference" in datum

        if args.reference_free and not has_reference:
            continue

        if is_turn_level:
            if has_reference and not args.reference_free:
                human_template = turn_eval_template
            else:
                human_template = turn_noref_eval_template
        else:
            human_template = dialogue_eval_template

        try:
            raw_output, scores = run_eval_chain(
                model_name=args.model_name,
                score_aspects=aspects,
                human_template=human_template,
                context=datum["context"] if is_turn_level else None,
                response=datum["response"] if is_turn_level else None,
                reference=datum["reference"] if is_turn_level and has_reference else None,
                dialog=datum["dialog"] if not is_turn_level else None,
                **score_config[score_type],
            )
        except Timeout:
            logging.warning("Timeout for %s" % datum["dialogue_id"])
            continue
        if scores is None:
            logging.warning("Failed to parse output: %s" % raw_output)
            continue
        for aspect in aspects:
            dialogue_id_with_aspect = f"{datum['dialogue_id']}|{aspect}"
            aspect_score = scores.dict()[aspect]
            results.append(
                {
                    "dialogue_id": dialogue_id_with_aspect,
                    "score": aspect_score,
                    "dialouge_id_wo_aspect": datum["dialogue_id"],
                    "raw_output": raw_output,
                }
            )

        # save periodically
        if _i % 10 == 0:
            df = pd.DataFrame(results)
            # if duplicate on dialogue_id, keep the last one
            df = df.drop_duplicates(subset="dialogue_id", keep="last")
            # sort by dialogue_id
            df = df.sort_values(by="dialogue_id")
            df.to_csv(args.save_path, index=False)
            # logging.info("Saved to %s, %d results" % (args.save_path, len(results)))

    df = pd.DataFrame(results)
    # if duplicate on dialogue_id, keep the last one
    df = df.drop_duplicates(subset="dialogue_id", keep="last")
    # sort by dialogue_id
    df = df.sort_values(by="dialogue_id")
    df.to_csv(args.save_path, index=False)
    logging.info("Saved to %s, %d results" % (args.save_path, len(results)))
