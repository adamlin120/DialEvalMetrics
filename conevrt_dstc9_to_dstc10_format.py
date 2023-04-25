import json
from itertools import cycle


def main():
    with open("llm/data/dstc9.json", "r") as f:
        dstc9 = json.load(f)

    dstc9_dials = []
    dstc9_turns = []

    for data in dstc9.values():
        context = data['context']
        response = data['response']
        context_str = ""

        dial_output = {
            "dialog": [],
            "annotations": {
                "Overall": [data['gold_score']]
            },
            "model": data['system'].split('.')[0],
            "dialogue_id": f"dstc9-dial_{data['id']}"
        }

        for speaker, text in zip(cycle(["human", "model"]), context):
            dial_output["dialog"].append({"speaker": speaker, "text": text})
            context_str += f"{speaker}: {text}\n"
        dial_output["dialog"].append({"speaker": "model", "text": response})

        turn_output = {
            "context": context_str,
            "response": response,
            "model": data['system'].split('.')[0],
            "annotations": {
                "Overall": [data['gold_score']]
            },
            "dialogue_id": f"dstc9-turn_{data['id']}"
        }
        dstc9_turns.append(turn_output)
        dstc9_dials.append(dial_output)

    with open("DSTC 10 Track 5/Subtask 1/human_evaluation_data/dstc9-dial_eval.json", "w") as f:
        json.dump(dstc9_dials, f, indent=2)
    with open("DSTC 10 Track 5/Subtask 1/human_evaluation_data/dstc9-turn_eval.json", "w") as f:
        json.dump(dstc9_turns, f, indent=2)


if __name__ == "__main__":
    main()