import argparse
import json
import os

import backoff
import openai
from tqdm import tqdm

from prompt import prompt_templates


# Define a function to get the response for a given prompt and store it in the cache
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
@backoff.on_exception(backoff.expo, openai.error.ServiceUnavailableError)
def get_response(
    prompt: str,
    model: str = "gpt-3.5-turbo-0301",
    temperature: float = 0.0,
    max_tokens: int = 10,
    **kwargs,
):
    # Make the API call and store the response in the cache
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,  # Set temperature=0 for greedy decoding
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    response_text = response.choices[0].message.content.strip()
    return response_text


def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts_file", help="path to the JSON file containing prompts")
    parser.add_argument(
        "response_cache_file", help="path to the JSON file containing response cache"
    )
    parser.add_argument("prompt_style", help="prompt style to use")
    args = parser.parse_args()

    prompt_style = args.prompt_style
    if prompt_style not in prompt_templates:
        raise ValueError(
            f"Invalid prompt style: {prompt_style}, must be one of {list(prompt_templates.keys())}"
        )

    response_cache_file = args.response_cache_file
    # if response_cache_file directory does not exist, create it
    if not os.path.exists(os.path.dirname(response_cache_file)):
        os.makedirs(os.path.dirname(response_cache_file))

    # Load the response cache from the file specified in the command-line arguments
    try:
        with open(response_cache_file, "r") as f:
            response_cache = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        response_cache = {}

    # Load the prompts from the JSON file specified in the command-line arguments
    with open(args.prompts_file, "r") as f:
        prompts_dict = json.load(f)

    # Example usage with progress bar
    for prompt_id, datum in tqdm(prompts_dict.items()):
        # Check if the response is already cached
        if prompt_id not in response_cache:
            prompt = datum["prompt"][prompt_style]
            try:
                response = get_response(prompt)
            except openai.error.InvalidRequestError:
                print(f"InvalidRequestError for prompt {prompt_id}")
                continue
            response_cache[prompt_id] = response

            # Save the cache to file
            with open(response_cache_file, "w") as f:
                json.dump(response_cache, f, indent=2)


if __name__ == "__main__":
    main()
