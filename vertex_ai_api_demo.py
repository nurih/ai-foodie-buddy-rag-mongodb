import json
from prompt_toolkit.cursor_shapes import CursorShape
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

    # The transformer used to create an embedding
    transformer = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    token_count = 384

    user_question_embedding = transformer.encode(user_question).tolist()

    say(f"Embedding {user_question_embedding[0]}... ({token_count} items)")

    # Query for MongoDB Atlas vector search
    mql_pipeline = format_mql_query(user_question_embedding)

    if confirm("Print the Atlas MongoDB vector search query? "):
        say(f"{json.dumps(mql_pipeline, indent=2)}")

    # perform vector search (the R in RAG)
    mongo_client = MongoClient(MONGO_URI)

    reviews_collection: collection = mongo_client[ATLAS_DB][ATLAS_DB_COLLECTION]

    verctor_search_result = reviews_collection.aggregate(mql_pipeline)

    top_restaurant = list(verctor_search_result)[0]

    if confirm("Print the Atlas vector search result document(s)? "):
        say(top_restaurant)

    # Engineer a one-shot prompt (the A in RAG)
    llm_prompt = format_llm_prompt(user_question, top_restaurant)

    if confirm("Print llm prompt given to Vertex API? "):
        say(llm_prompt)

    # Execute the prompt against a pre-built model (The G in RAG)
    generative_prediction_parameters = {
        "temperature": 0.14,
        "max_output_tokens": 180,
        "top_p": 0.86,
        "top_k": 40,
    }

    model = language_models.TextGenerationModel.from_pretrained("text-bison@002")
    

    response = model.predict(
        prompt=llm_prompt,
        **generative_prediction_parameters,
    )

    print(HTML(f"<b><u><yellow>Generated Reccomendation:</yellow></u></b>"))

    print(response.text)


def format_mql_query(embedding_of_query: list):
    mql_pipeline = [
        {
            "$vectorSearch": {
                "index": ATLAS_VECTOR_INDEX,
                "queryVector": embedding_of_query,
                "path": DOCUMENT_EMBEDDINGS_FIELD,
                "numCandidates": 180,
                "limit": 120,
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
        {"$match": {"$expr": {"$gt": [{"$strLenCP": "$text"}, 100]}}},
        {
            "$group": {
                "_id": "$gmap_id",
                "reviews": {"$push": {"by": "$name", "text": "$text"}},
                "n": {"$count": {}},
                "max_score": {"$max": "$score"},
                "avg_score": {"$avg": "$score"},
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

    llm_prompt = f"Summarize these reccomendations to tell me why I should go to the restaurant given my criteria.\n\nCriteria: {user_prompt}\nReviews: {reviews_to_consider}."

    return llm_prompt


def prep_vertex_ai():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ[
        "GCP_SERVICE_ACCOUNT_KEY_PATH"
    ]

    gcp_project = os.environ["GCP_PROJECT"]

    gcp_location = os.environ["GCP_LOCATION"]

    # Initialize vertexai
    vertexai.init(project=gcp_project, location=gcp_location)

    print(gcp_project, gcp_location)


def say(text: str):
    print(HTML("<p>\t</p>"))
    print(text)


def confirm(text) -> bool:
    return prompt(
        HTML(f"<ansicyan><b><u>{text}</u></b></ansicyan>"),
        cursor=CursorShape.BLINKING_BLOCK,
    ) in ["Y", "y"]


from prompt_toolkit.application import get_app

if __name__ == "__main__":

    prep_vertex_ai()

    while True:

        user_question = prompt(
            HTML(
                "<ansiblue><u><b>What food adventure do you seek?: </b></u></ansiblue>"
            )
        )

        answer_my_question(user_question)

        if not confirm("Go again? "):
            print(HTML("<ansiblue><b>Bye!</b></ansiblue>"))
            break
