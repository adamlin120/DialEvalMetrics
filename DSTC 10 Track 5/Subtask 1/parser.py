from typing import Dict

from langchain.output_parsers import PydanticOutputParser
from pydantic import create_model


def generate_score_model(
    field_names: Dict[str, str], score_type: type, score_range: tuple
) -> type:
    fields = {}
    for field_name in field_names:
        fields[field_name] = (score_type, ...)

    ScoreModel = create_model("ScoreModel", **fields)

    for field_name, field_info in ScoreModel.__fields__.items():
        field_info.field_info.description = f"{field_names[field_name]} score in the range of {score_range[0]} to {score_range[1]}"

    return ScoreModel


def get_pydantic_output_parser(*args, **kwargs) -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=generate_score_model(*args, **kwargs))


if __name__ == "__main__":
    score_fields = {"engagement": "Engagement", "relevance": "Relevance"}
    score_type = int
    score_range = (0, 100)
    EngagementAndRelevanceScoreModel = generate_score_model(
        score_fields, score_type, score_range
    )
    print(EngagementAndRelevanceScoreModel.schema_json(indent=2))

    parser = PydanticOutputParser(pydantic_object=EngagementAndRelevanceScoreModel)
    print(parser.get_format_instructions())
