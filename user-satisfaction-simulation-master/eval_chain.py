import logging
import os
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from openai.error import Timeout
from tqdm import tqdm
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import RegexParser

logging.basicConfig(level=logging.INFO)

# eval_template = (
#     "Evaluate the following dialogue based on user satisfaction, using the following scale:\n"
#     "1 - Very dissatisfied: The system fails to understand and fulfill the user's request.\n"
#     "2 - Dissatisfied: The system understands the request but fails to satisfy it in any way.\n"
#     "3 - Normal: The system understands the user's request and either partially satisfies the request or provides information on how the request can be fulfilled.\n"
#     "4 - Satisfied: The system understands and satisfies the user request, but provides more information than what the user requested or takes extra turns before meeting the request.\n"
#     "5 - Very satisfied: The system understands and satisfies the user request completely and efficiently.\n"
#     "Dialogue: {dialog}\n"
#     "Score the dialogue on a scale from 1.0 to 5.0."
#     "Score:"
# )

eval_template = (
    "Evaluate the following dialogue based on user satisfaction, using the following scale:\n"
    "Dialogue: {dialog}\n"
    "Score the dialogue on a scale from 1.0 to 5.0."
    "Score:"
)


def parse_dialogue_scores(filepath):
    dataset_name = Path(filepath).stem
    with open(filepath, 'r') as f:
        content = f.read().strip()

    dialogues = content.split('\n\n')

    scores_list = []
    tid = 0
    did = 0
    for dialogue in dialogues:
        lines = dialogue.strip().split('\n')
        dialogue_level_scores = lines.pop().strip().rsplit('\t', 1)[-1]
        dialogue_level_scores = list(map(int, dialogue_level_scores.split(',')))

        context = []
        for line in lines:
            if line.startswith("USER"):
                speaker_role, text, _, satisfaction = line.strip().split('\t')
                satisfaction_scores = list(map(int, satisfaction.split(',')))
                if context:
                    scores_list.append({
                        'id': f'{dataset_name}_turn_{tid}',
                        'dialog': '\n'.join(context),
                        'satisfaction': np.mean(satisfaction_scores),
                        'level': 'turn'
                    })
                    tid += 1
            elif line.startswith("SYSTEM"):
                speaker_role, text = line.split('\t')[:2]
            else:
                raise ValueError(f"Unknown line: {line}")
            context.append(f"{speaker_role}: {text}")
        scores_list.append({
            'id': f'{dataset_name}_dialogue_{did}',
            'dialog': '\n'.join(context),
            'satisfaction': np.mean(dialogue_level_scores),
            'level': 'dialogue'
        })
        did += 1

    df = pd.DataFrame(scores_list)
    df.set_index('id', inplace=True)
    df.sort_index(inplace=True)
    return df


def run_eval_chain(
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

    pattern = r"(\d+\.?\d*)"
    parser = RegexParser(
        regex=pattern,
        output_keys=["score"],
    )

    output = chain.run(**prompt_kwargs)
    try:
        scores = parser.parse(output)
        scores = float(scores["score"])
    except:
        logging.warning("Failed to parse output: %s" % output)
        scores = None

    return output, scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    save_path = args.save_path

    df = parse_dialogue_scores(args.data_path)

    logging.info(f"Found total {len(df)} turns and dialogues")

    if os.path.exists(save_path):
        df = pd.read_json(save_path)

    pbar = tqdm(enumerate(df.iterrows()), total=len(df))
    for _i, (index, datum) in pbar:
        try:
            if "satisfaction_pred" in datum and not np.isnan(datum["satisfaction_pred"]):
                continue
            raw_output, scores = run_eval_chain(
                model_name=args.model_name,
                human_template=eval_template,
                dialog=datum["dialog"],
            )
        except Timeout:
            logging.warning("Timeout for %s" % datum["dialogue_id"])
            continue
        if scores is None:
            logging.warning("Failed to parse output: %s" % raw_output)
            continue

        # save the raw output and the predicted score
        df.loc[index, "raw_output"] = raw_output
        df.loc[index, "satisfaction_pred"] = scores

        # save periodically
        if _i % 30 == 0:
            df.to_json(save_path)

            # calculate the correlation
            pearson = df["satisfaction"].corr(df["satisfaction_pred"], method="pearson")
            spearman = df["satisfaction"].corr(df["satisfaction_pred"], method="spearman")
            kendall = df["satisfaction"].corr(df["satisfaction_pred"], method="kendall")
            pbar.set_description(f"Pearson: {pearson:.3f}, Spearman: {spearman:.3f}, Kendall: {kendall:.3f}")

    df.to_json(save_path)
    # calculate the correlation
    pearson = df["satisfaction"].corr(df["satisfaction_pred"], method="pearson")
    spearman = df["satisfaction"].corr(df["satisfaction_pred"], method="spearman")
    kendall = df["satisfaction"].corr(df["satisfaction_pred"], method="kendall")
    print(f"Pearson: {pearson}")
    print(f"Spearman: {spearman}")
    print(f"Kendall: {kendall}")
