import json
from prompt_toolkit.shortcuts import input_dialog, yes_no_dialog, message_dialog
from prompt_toolkit import print_formatted_text as print, prompt
from prompt_toolkit import HTML
from sentence_transformers import SentenceTransformer
import os

from pymongo import MongoClient, collection

import vertexai
from vertexai import language_models
import os

from settings import (
    ATLAS_DB,
    ATLAS_DB_COLLECTION,
    ATLAS_VECTOR_INDEX,
    DOCUMENT_EMBEDDINGS_FIELD,
)

MONGO_URI = os.environ.get("MONGO_URL")
transformer = None


def answer_my_question(user_question: str):

    transformer = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    token_count = 384

    user_question_embedding = transformer.encode(user_question).tolist()

    say(f"Embedding {user_question_embedding[0]}... ({token_count} items)")

    mql_pipeline = format_mql_query(user_question_embedding)

    if confirm("Print the query?"):
        say(f"{json.dumps(mql_pipeline, indent=2)}")

    mongo_client = MongoClient(MONGO_URI)

    reviews_collection: collection = mongo_client[ATLAS_DB][ATLAS_DB_COLLECTION]

    verctor_search_result = reviews_collection.aggregate(mql_pipeline)

    top_restaurant = list(verctor_search_result)[0]

    if confirm("Print the vector search result?"):
        say(top_restaurant)

    llm_prompt = format_llm_prompt(user_question, top_restaurant)

    if confirm("Print llm prompt given to Vertex API?"):
        say(llm_prompt)

    generative_prediction_parameters = {
        "temperature": 0.17,
        "max_output_tokens": 160,
        "top_p": 0.86,
        "top_k": 40,
    }

    model = language_models.TextGenerationModel.from_pretrained("text-bison@002")

    response = model.predict(
        prompt=llm_prompt,
        **generative_prediction_parameters,
    )

    print(HTML(f"<p><violet><i>{response.text}</i></violet></p>"))


def format_mql_query(embedding_of_query: list):
    mql_pipeline = [
        {
            "$vectorSearch": {
                "index": ATLAS_VECTOR_INDEX,
                "queryVector": embedding_of_query,
                "path": DOCUMENT_EMBEDDINGS_FIELD,
                "numCandidates": 150,
                "limit": 100,
                "filter": {"rating": {"$gte": 3}},
            }
        },
        {
            "$project": {
                "text": 1,  # review  field
                "name": 1,  # reviewer field
                "gmap_id": 1,  # locator for restaurant
                "score": {"$meta": "vectorSearchScore"},  # score
            }
        },
        {
            "$group": {
                "_id": "$gmap_id",
                "reviews": {"$push": {"by": "$name", "text": "$text"}},
                "n": {"$count": {}},
            }
        },
        {"$sort": {"n": -1}},
        {"$limit": 1},
    ]

    return mql_pipeline


def format_llm_prompt(user_prompt: str, best_match) -> str:
    reviews_to_consider = "\n\n".join(
        [json.dumps(r) for r in best_match["reviews"][:10]]
    )

    llm_prompt = f"Write a restaurant reccomendation based on the Query and the provided Reviews only. Elaborate why I would love this place.\nQuery: {user_prompt}\nReviews: {reviews_to_consider}."

    return llm_prompt


def prep_vertex_ai():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ[
        "GCP_SERVICE_ACCOUNT_KEY_PATH"
    ]

    gcp_project = os.environ["GCP_PROJECT"]

    gcp_location = os.environ["GCP_LOCATION"]

    # Initialize vertexai
    vertexai.init(project=gcp_project, location=gcp_location)


def say(text: str):
    print(HTML("<hr/>"))
    print(text)


def confirm(text) -> bool:
    return prompt(text + "  ") in ["Y", "y"]


from prompt_toolkit.application import get_app

if __name__ == "__main__":

    prep_vertex_ai()

    while True:
        # user_question = input_dialog(
        #     title="User Question", text="Tell me about your ideal food"
        # ).run()

        user_question = prompt("Tell me about your ideal food experience:")

        answer_my_question(user_question)

        if not confirm("Go again?"):
            print(HTML("<ansigreen><b>Bye!</b></ansigreen>"))
            break
