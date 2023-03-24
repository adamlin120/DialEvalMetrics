prompt_templates = {
    "0-100": {
        "prompt": """Score the following dialogue response generated on a continuous scale from 0 to 100, where a score of zero means "completely irrelevant or nonsensical" and a score of one hundred means "completely relevant and coherent".\n\nContext: "{context}"\nDialogue response: "{response}"\nScore:""",
        "use_reference": False,
    },
    "0-100_ref": {
        "prompt": """Score the following dialogue response generated on a continuous scale from 0 to 100, where a score of zero means "completely irrelevant or nonsensical" and a score of one hundred means "completely relevant and coherent".\n\nContext: "{context}"\nDialogue response: "{response}"\nReference: "{reference}"\nScore:""",
        "use_reference": True,
    },
    "0-5": {
        "prompt": """Score the following dialogue response generated on a continuous scale from 0.0 to 5.0, where a score of zero means "completely irrelevant or nonsensical" and a score of five means "completely relevant and coherent".\n\nContext: "{context}"\nDialogue response: "{response}"\nScore:""",
        "use_reference": False,
    },
    "0-5_ref": {
        "prompt": """Score the following dialogue response generated on a continuous scale from 0.0 to 5.0, where a score of zero means "completely irrelevant or nonsensical" and a score of five means "completely relevant and coherent".\n\nContext: "{context}"\nDialogue response: "{response}"\nReference: "{reference}"\nScore:""",
        "use_reference": True,
    },
}
