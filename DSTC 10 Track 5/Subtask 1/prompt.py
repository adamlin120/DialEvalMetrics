turn_eval_template = (
    "{format_instructions}\n"
    "Score the following dialogue response generated on a continuous scale from {score_min} to {score_max}.\n"
    "Context: {context}\n"
    "Reference: {reference}\n"
    "Dialogue response: {response}"
)

dialogue_eval_template = (
    "{format_instructions}\n"
    "Score the following dialogue generated on a continuous scale from {score_min} to {score_max}.\n"
    "Dialogue: {dialog}"
)

score_config = {
    "0-5": {
        "score_min": 0.0,
        "score_max": 5.0,
        "score_dtype": float,
    },
    "0-100": {
        "score_min": 0,
        "score_max": 100,
        "score_dtype": int,
    },
}
