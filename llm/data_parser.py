import json
import os
from typing import List, Optional

from .count_prompt_tokens import num_tokens_from_messages
from .prompt import prompt_templates


def fill_in_prompt(
    prompt_template: str,
    context: List[str],
    response: str,
    reference: Optional[str] = None,
) -> str:
    # if num_tokens exceed 4050, truncate the context at the front
    while True:
        if reference:
            prompt = prompt_template.format(
                context=flatten_context(context),
                response=response,
                reference=reference,
            )
        else:
            prompt = prompt_template.format(context=flatten_context(context), response=response)
        messages = [
            {"role": "user", "content": prompt},
        ]
        num_tokens = num_tokens_from_messages(messages)
        if num_tokens <= 4050:
            break
        else:
            context = context[1:]
    return prompt


def flatten_context(context: List[str]) -> str:
    context_flatten = ""
    for _turn_idx, turn in enumerate(context):
        speaker = "Speaker1" if _turn_idx % 2 == 0 else "Speaker2"
        context_flatten += f"{speaker}: {turn.strip()}\n"
    context_flatten = context_flatten.strip()
    return context_flatten

def gen_llm_data(data, output_path):
    """
    Args:
        data: the return value of load_data functions e.g. {'contexts': ...}
            data.keys() == dict_keys(['contexts', 'responses', 'references', 'models', 'scores'])
            contexts: List[str]
            responses: str
            references: str
            models: str
            scores: str (but can be converted to float)
        output_path: path to the output file
    """
    format_data = {}
    for idx, (context, response, reference, gold_score, system) in enumerate(
        zip(
            data["contexts"],
            data["responses"],
            data["references"],
            data["scores"],
            data["models"],
        )
    ):
        if reference == "NO REF":
            reference = ""

        # context_flatten = ""
        # for _turn_idx, turn in enumerate(context):
        #     speaker = "Speaker1" if _turn_idx % 2 == 0 else "Speaker2"
        #     context_flatten += f"{speaker}: {turn.strip()}\n"
        # context_flatten = context_flatten.strip()

        speaker = "Speaker1" if len(context) % 2 == 0 else "Speaker2"
        response_with_speaker = f"{speaker}: {response.strip()}"

        format_data[idx] = {
            "id": idx,
            "system": system,
            "prompt": {},
            "gold_score": gold_score,
            "context": context,
            "response": response,
            "reference": reference,
        }

        # TODO: specific prompt templates
        for template_name, template in prompt_templates.items():
            prompt_template_with_context = template["prompt"]
            use_reference = template["use_reference"]

            if use_reference:
                if reference:
                    # prompt = prompt_template_with_context.format(
                    #     context=context_flatten,
                    #     response=response_with_speaker,
                    #     reference=reference,
                    # )
                    prompt = fill_in_prompt(
                        prompt_template_with_context,
                        context,
                        response_with_speaker,
                        reference,
                    )
                else:
                    continue
            else:
                # prompt = prompt_template_with_context.format(
                #     context=context_flatten, response=response_with_speaker
                # )
                prompt = fill_in_prompt(
                    prompt_template_with_context, context, response_with_speaker
                )

            format_data[idx]["prompt"][template_name] = prompt

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w") as f:
        f.write(json.dumps(format_data, indent=2))
