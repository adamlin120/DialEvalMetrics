from pathlib import Path

import pandas as pd
import numpy as np


def parse_dialogue_scores(filepath):
    with open(filepath, 'r') as f:
        content = f.read().strip()

    dialogues = content.split('\n\n')

    turn_level_scores_list = []
    dialogue_scores_list = []
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
                    turn_level_scores_list.append({
                        'dialog': '\n'.join(context),
                        'satisfaction': np.mean(satisfaction_scores)
                    })
            elif line.startswith("SYSTEM"):
                speaker_role, text = line.split('\t')[:2]
            else:
                raise ValueError(f"Unknown line: {line}")
            context.append(f"{speaker_role}: {text}")
        dialogue_scores_list.append({
            'dialog': '\n'.join(context),
            'satisfaction': np.mean(dialogue_level_scores)
        })

    df_turn = pd.DataFrame(turn_level_scores_list)
    df_dialogue = pd.DataFrame(dialogue_scores_list)
    return df_turn, df_dialogue


if __name__ == '__main__':
    for path in Path('dataset/').glob('*.txt'):
        if 'action' in str(path).lower() or path.stem == "JDDC":
            continue
        df_turn, df_dialogue = parse_dialogue_scores(path)
        print(f"Dataset: {path.stem} Turn-level scores:")
        print(df_turn.sample(5))
        print(f"Dataset: {path.stem} Dialogue-level scores:")
        print(df_dialogue.sample(5))

        print(f"There are {len(df_turn)} turn-level scores and "
              f"{len(df_dialogue)} dialogue-level scores in {path.stem} dataset.")
