# %%
import json
from embedder import MULTI_QA_MINILM_L6_COS_V1, Embedder
from llm_predictor import LlmPredictor
from settings import ATLAS_VECTOR_INDEX, DOCUMENT_EMBEDDINGS_FIELD

from mongo_atlas import restaurants_collection, reviews_collection


# %%


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


# %%

# The embedder must be same one used to populate the documents in MongoDB
embedder = Embedder(MULTI_QA_MINILM_L6_COS_V1)


def vector_search(text_query: str):

    if not text_query:
        raise ValueError("Invalid query.")

    embedding_of_query = embedder.encode(text_query)

    if embedding_of_query is None:
        raise ValueError("Embedding generation failed.")

    mql_pipeline = format_mql_query(embedding_of_query)

    with open(".cache/mql_pipeline.json", "w", encoding="utf8") as f:
        f.write(json.dumps(mql_pipeline, indent=None))

    results = reviews_collection.aggregate(mql_pipeline)

    return list(results)


# %%


def format_llm_prompt(user_prompt: str, top_result) -> str:
    """Formats user query + Atlas Vector results
    into a usable prompt for LLM to create final answer.

    user_prompt: the query from the user

    top_result: The response document from querying mongo, which contains reviews for a single place
    """
    reviews_to_consider = "\n\n".join(
        [json.dumps(r) for r in top_result["reviews"][:4]]
    )

    llm_prompt = f"Write a restaurant review based on the question and reviews provieded below. Write why a person would love to eat there.\nQuery: {user_prompt}\nContinue to answer the query by using these actual reviews:\n\n{reviews_to_consider}."

    return llm_prompt

# %%
# Simulare Q-A cycle:

# user_prompt = "I love me some fried chicken and waffle, with crispy breading and made from scratch, served with **real** maple syrup."
# user_prompt = "Where are the best empanadas that are modern and have really good flavor, and the waiters are extra crispy?"

llm_predictor = LlmPredictor().use_api_vertex_ai()

user_prompt = "Find me the best Spam musubi that locals go to and is a hidden gem."


print("1. Given a user prompt, perform vector search.")

search_result_documents = vector_search(text_query=user_prompt)
top_result = search_result_documents[0]

# Get info about the restaurant we will use to display later
print("The vector search yielded restaurant id", top_result["_id"])

restaurant = restaurants_collection.find_one({"_id": top_result["_id"]})

restaurant and print(restaurant)

print("2. Prompt Engineering: Create an LLM prompt.")
print("   Use the top restaurant + reviews.")
llm_prompt = format_llm_prompt(user_prompt, top_result)

with open('./.cache/llm_prompt.txt', 'w') as lph:
    lph.write(llm_prompt)
    
print(llm_prompt)

print("3. Use the LLM prediction engine of choice")

generated_text = llm_predictor.predict(llm_prompt)

print("4. Process complete")
print("*" * 64)
print("Original User Prompt was:")
print(user_prompt)
print("*" * 64)
print(generated_text)

# %%
