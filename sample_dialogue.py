"""
Source: https://raw.githubusercontent.com/skywalker023/sodaverse/main/chat_with_cosmo.py
"""

import random
from copy import deepcopy

import openai
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Conversation,
    pipeline,
    set_seed,
)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_input(situation_narrative, role_instruction, conversation_history):
    input_text = " <turn> ".join(conversation_history)

    if role_instruction != "":
        input_text = "{} <sep> {}".format(role_instruction, input_text)

    if situation_narrative != "":
        input_text = "{} <sep> {}".format(situation_narrative, input_text)

    return input_text


def main():
    dd = (
        load_dataset("daily_dialog", split="test")
        .shuffle(seed=42)
        .select(range(100))["dialog"]
    )
    dataset = []

    for conversation in dd:
        length_of_conversation = len(conversation)
        index = random.randint(1, length_of_conversation - 1)
        context = conversation[:index]
        response = conversation[index]
        # add html newline tags and ALICE and BOB tags
        context_html = [
            "<b>ALICE:</b> " + c if i % 2 == 0 else "<b>BOB:</b> " + c
            for i, c in enumerate(context)
        ]
        context_html = "<br>".join(context_html)
        speaker = "ALICE" if len(context) % 2 == 0 else "BOB"
        dataset.append(
            {
                "context": context,
                "response": response,
                "context_html": context_html,
                "speaker": speaker,
            }
        )

    # chatgpt
    prompt_template = "You will be generating the next turn of a given dialogue between two people. " \
                      "Your response should usually be 1-2 sentences.\n" \
                      "Dialogue:\n{dialogue}\n" \
                      "What is the most appropriate next utterance (3 sentences max)?."
    for row in tqdm(dataset):
        conversation = row["context"]
        prompt = prompt_template.format(dialogue="\n".join(conversation))
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            temperature=0,  # Set temperature=0 for greedy decoding
            messages=[{"role": "user", "content": prompt}],
            stop=["\n"],
        )
        response_text = response.choices[0].message.content.strip()
        row["ChatGPT"] = response_text
        print(f"chatgpt: {response_text}")

    dialogpt = pipeline(
        "conversational",
        "microsoft/DialoGPT-large",
        device=0 if torch.cuda.is_available() else -1,
    )
    dialogpt.tokenizer.pad_token_id = dialogpt.tokenizer.eos_token_id
    for row in tqdm(dataset):
        conversation = deepcopy(row["context"])
        new_user_input = conversation.pop()
        if len(conversation) % 2 == 0:
            past_user_inputs = conversation[::2]
            generated_responses = conversation[1::2]
        else:
            past_user_inputs = conversation[1::2]
            generated_responses = conversation[::2]
        conv = Conversation(
            text=new_user_input,
            past_user_inputs=past_user_inputs,
            generated_responses=generated_responses,
        )
        dgpt = dialogpt(conv, top_p=0.95, do_sample=True).generated_responses[-1]
        row["DialoGPT"] = dgpt
        print(f"DialoGPT: {dgpt}")
    del dialogpt
    torch.cuda.empty_cache()

    blenderbot = pipeline(
        "text2text-generation",
        "facebook/blenderbot-3B",
        device=0 if torch.cuda.is_available() else -1,
    )
    for row in tqdm(dataset):
        conversation = row["context"]
        bb = blenderbot(
            "</s> <s>".join(conversation), top_p=0.95, do_sample=True, truncation=True
        )[0]["generated_text"]
        row["BlenderBot-3B"] = bb
        print(f"blenderbot: {bb}")
    del blenderbot
    torch.cuda.empty_cache()

    cosmo_tokenizer = AutoTokenizer.from_pretrained("allenai/cosmo-xl")
    cosmo_model = AutoModelForSeq2SeqLM.from_pretrained("allenai/cosmo-xl").to(device)

    def generate(situation_narrative, role_instruction, conversation_history):
        """
        situation_narrative: the description of situation/context with the characters included (e.g., "David goes to an amusement park")
        role_instruction: the perspective/speaker cosmo_instruction (e.g., "Imagine you are David and speak to his friend Sarah").
        conversation_history: the previous utterances in the conversation in a list
        """

        input_text = set_input(
            situation_narrative, role_instruction, conversation_history
        )

        inputs = cosmo_tokenizer([input_text], return_tensors="pt").to(device)
        outputs = cosmo_model.generate(
            inputs["input_ids"],
            max_new_tokens=128,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
        )
        response = cosmo_tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return response

    situation = ""
    cosmo_instruction = "You are Cosmo and you are talking to a friend."
    for row in tqdm(dataset):
        conversation = row["context"]
        cosmo = generate(situation, cosmo_instruction, conversation)
        row["COSMO-3B"] = cosmo
        print(f"cosmo: {cosmo}")
    del cosmo_model
    torch.cuda.empty_cache()

    godel_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/GODEL-v1_1-large-seq2seq"
    )
    godel_model = AutoModelForSeq2SeqLM.from_pretrained(
        "microsoft/GODEL-v1_1-large-seq2seq"
    ).to(device)

    def godel_generate(instruction, knowledge, dialog):
        if knowledge != "":
            knowledge = "[KNOWLEDGE] " + knowledge
        dialog = " EOS ".join(dialog)
        query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
        input_ids = (
            godel_tokenizer(f"{query}", return_tensors="pt").to(device).input_ids
        )
        outputs = godel_model.generate(
            input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True
        )
        output = godel_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output

    godel_instruction = "You are talking to a friend."
    for row in tqdm(dataset):
        conversation = row["context"]
        godel = godel_generate(godel_instruction, situation, conversation)
        row["GODEL-L"] = godel
        print(f"godel: {godel}")
    del godel_model
    torch.cuda.empty_cache()

    dataset = pd.DataFrame(dataset)
    dataset.to_csv("chitchat_samples.csv", index=False)


if __name__ == "__main__":
    main()
