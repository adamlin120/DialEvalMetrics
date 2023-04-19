import json
from argparse import ArgumentParser

import pandas as pd

MAPPING = {
    "1": "COSMO-3B",
    "2": "GODEL-L",
    "3": "response",
    "4": "BlenderBot-3B",
    "5": "chatgpt",
}


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    results = []
    # list of dict of keys: [context, response, model, annotations, dialogue_id]
    # annotations: [consistent, natural, quality]

    df = pd.read_csv(args.input)

    # header: "HITId","HITTypeId","Title","Description","Keywords","Reward","CreationTime","MaxAssignments","RequesterAnnotation","AssignmentDurationInSeconds","AutoApprovalDelayInSeconds","Expiration","NumberOfSimilarHITs","LifetimeInSeconds","AssignmentId","WorkerId","AssignmentStatus","AcceptTime","SubmitTime","AutoApprovalTime","ApprovalTime","RejectionTime","RequesterFeedback","WorkTimeInSeconds","LifetimeApprovalRate","Last30DaysApprovalRate","Last7DaysApprovalRate","Input.context","Input.response","Input.context_html","Input.speaker","Input.chatgpt","Input.BlenderBot-3B","Input.COSMO-3B","Input.GODEL-L","Answer.taskAnswers","Approve","Reject"
    # rename HITId to id
    df = df.rename(columns={'HITId': 'id'})

    # group by HITId and aggregate the columns by first
    # for Answer.taskAnswers gather all the answers
    df = df.groupby('id').agg({
        'Input.context': 'first',
        'Input.response': 'first',
        'Input.chatgpt': 'first',
        'Input.BlenderBot-3B': 'first',
        'Input.COSMO-3B': 'first',
        'Input.GODEL-L': 'first',
        'Answer.taskAnswers': lambda x: list(x)
    })
    # eval Input.context
    df["Input.context"] = df["Input.context"].apply(eval)

    # strip Input
    df = df.rename(columns=lambda x: x.replace('Input.', ''))
    # rename Answer.taskAnswers to answers
    df = df.rename(columns={'Answer.taskAnswers': 'answers'})

    # eval elements in answers
    df["answers"] = df["answers"].apply(lambda x: [eval(y)[0] for y in x])

    """
    [{'consistent1': 3, 'consistent2': 3, 'consistent3': 3, 'consistent4': 4, 'consistent5': 4, 'natural1': 3, 'natural2': 3, 'natural3': 4, 'natural4': 3, 'natural5': 3, 'quality1': 4, 'quality2': 4, 'quality3': 4, 'quality4': 3, 'quality5': 4}, {'consistent1': 3, 'consistent2': 4, 'consistent3': 5, 'consistent4': 4, 'consistent5': 4, 'natural1': 3, 'natural2': 4, 'natural3': 4, 'natural4': 3, 'natural5': 4, 'quality1': 3, 'quality2': 5, 'quality3': 4, 'quality4': 3, 'quality5': 4}, {'consistent1': 4, 'consistent2': 4, 'consistent3': 3, 'consistent4': 5, 'consistent5': 5, 'natural1': 4, 'natural2': 3, 'natural3': 4, 'natural4': 4, 'natural5': 4, 'quality1': 3, 'quality2': 5, 'quality3': 5, 'quality4': 4, 'quality5': 4}]
    to {"consistent1": [3, ....
    """
    df["answers"] = df["answers"].apply(lambda x: {k: [y[k] for y in x] for k in x[0].keys()})


    for id, row in df.iterrows():
        for i, model_name in MAPPING.items():
            results.append(
                {
                    "dialogue_id": f"{id}_{model_name}",
                    "model": model_name,
                    "context": row["context"],
                    "response": row[f"{model_name}"],
                    "annotations": {
                        "consistent": row["answers"][f"consistent{i}"],
                        "natural": row["answers"][f"natural{i}"],
                        "quality": row["answers"][f"quality{i}"],
                    }
                }
            )

    # assert length of results is 5 times the length of df
    assert len(results) == 5 * len(df)

    # assert 3 scores per annotation
    for result in results:
        for annotation in result["annotations"].values():
            assert len(annotation) == 3, f"Expected 3 scores per annotation, got {len(annotation)} in {result}"

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
