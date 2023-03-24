import json
from pathlib import Path

import tiktoken


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


def main():
    # iterate every json under data/*.json sorted by file size
    for path in sorted(Path("data").glob("*.json"), key=lambda p: p.stat().st_size):
        with open(path, "r") as f:
            prompts = json.load(f)
        num_tokens = 0
        for _, data in prompts.items():
            for _, prompt in data["prompt"].items():
                messages = [
                    {"role": "user", "content": prompt},
                ]
                num_tokens += num_tokens_from_messages(messages)
        # Model	Usage
        # gpt-3.5-turbo	$0.002 / 1K tokens
        # gpt 4 Model	Prompt	Completion
        # 8K context	$0.03 / 1K tokens	$0.06 / 1K tokens
        #  美元 等於 30.56 新臺幣
        price_35 = num_tokens * 0.002 / 1000 * 30.56
        price_4 = num_tokens * 0.03 / 1000 * 30.56
        print(
            f"{path.name}: {num_tokens} tokens, {price_35:.2f} NT$ for gpt-3.5-turbo, {price_4:.2f} NT$ for gpt-4"
        )


if __name__ == "__main__":
    main()
