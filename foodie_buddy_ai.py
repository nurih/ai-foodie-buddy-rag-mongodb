# %%
from datasets import load_dataset
from huggingface_hub import notebook_login
from pathlib import Path
from pymongo import MongoClient, collection
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import pandas as pd

ATLAS_DB = "demo"
ATLAS_DB_COLLECTION = "restaurant_reviews"
ATLAS_DB_RESTAURANTS_COLLECTION = "restaurant"
ATLAS_VECTOR_INDEX = "restaurant_reviews_index"
DOCUMENT_EMBEDDINGS_FIELD = "embedding"
REVIEWS_DIR = Path(os.environ.get("USERPROFILE")).joinpath(
    "Downloads", "restaurant_dataset"
)

# %%
# Reviews dataset
reviews_df = pd.read_json(REVIEWS_DIR.joinpath("reviews.json"), lines=True)
reviews_df.rename(columns={"user_id": "_id"}, inplace=True)
reviews_df.info()

# %%
# Cleanup reviews dataset

# Embedding generation breaks on null/ missing text. Drop it.
reviews_df = reviews_df.dropna(subset=["text"])
print("Null data in columns")
print(reviews_df.isnull().sum())

reviews_df.drop_duplicates(subset="_id", inplace=True)
print("Unique _id sanity check", len(reviews_df._id.unique()), len(reviews_df._id))

reviews_df["_id"] = reviews_df["_id"].astype("str")
reviews_df.info()

# %%
restuaurants_df = pd.read_json(REVIEWS_DIR.joinpath("restaurants.json"), lines=True)
restuaurants_df.rename(columns={"gmap_id": "_id"}, inplace=True)
restuaurants_df.info()


# %%
gte_large_model = SentenceTransformer("thenlper/gte-large")
asymmetric_model = SentenceTransformer(
    "sentence-transformers/msmarco-distilroberta-base-v2"
)
multi_qa_minilm_ls_cos = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# %%
# Function to create an embedding (a vector) from text

## Choose one of the embedding models:
# embedder = lambda text: gte_large_model.encode(text)
# embedder = lambda text: asymmetric_model.encode(text)
embedder = lambda text: multi_qa_minilm_ls_cos.encode(text)


def get_embedding_vector(text: str | list[str]) -> list[float]:
    if not text.strip():
        raise ValueError("Attempted to get embedding for empty text.")

    result = embedder(text).tolist()

    return result


EMBEDDING_LENGTH = len(get_embedding_vector("some text"))
print(f"embedding length {EMBEDDING_LENGTH}")
print(f"destination field {DOCUMENT_EMBEDDINGS_FIELD}")
print(f"*** Create the following vector index in Atlas Search: ***")
print(f"1. Database '{ATLAS_DB}'")
print(f"2. Collection '{ATLAS_DB_COLLECTION}'")
print(f"3. Index name '{ATLAS_VECTOR_INDEX}':")
print(
    f'4. Index JSON:\n\t{{"fields": [{{"numDimensions": {EMBEDDING_LENGTH},"path": "{DOCUMENT_EMBEDDINGS_FIELD}","similarity": "cosine","type": "vector"}},{{"type": "filter","path": "rating"}}]}}'
)

# %%
# Measure embedding_model's embedding nuance

prompt = "Chances of anything coming from Mars?"
data = [
    "The American approach to everything is Go Big or Go Home!",
    "When it comes to pizza, cover it in cheese completely.",
    "Then add cheese inside the crust, and sprinkle cheese on top.",
]

print(
    "embedding_model:",
    util.cos_sim(gte_large_model.encode(prompt), gte_large_model.encode(data)),
)

print(
    "asymmetric_model:",
    util.cos_sim(asymmetric_model.encode(prompt), asymmetric_model.encode(data)),
)

print(
    "multi_qa_minilm_ls_cos:",
    util.cos_sim(
        multi_qa_minilm_ls_cos.encode(prompt), multi_qa_minilm_ls_cos.encode(data)
    ),
)

# %%
EMBEDDINGS_DF_CACHE = "./.cache/embeddings_dataset.csv"

if Path(EMBEDDINGS_DF_CACHE).exists():
    reviews_df = pd.read_csv(EMBEDDINGS_DF_CACHE)
    if "Unnamed: 0" in reviews_df.columns:
        reviews_df.drop("Unnamed: 0", axis=1, inplace=True)
        print("Extra column dropped....")

else:
    print("Generating embeddings... this can take some time...")
    reviews_df[DOCUMENT_EMBEDDINGS_FIELD] = reviews_df["text"].apply(
        get_embedding_vector
    )
    reviews_df.to_csv(EMBEDDINGS_DF_CACHE, index=False)

reviews_df.info()

# %%

MONGO_URI = os.environ.get("MONGO_URL")


if not MONGO_URI:
    raise ValueError("MONGO_URL environment variable missing or empty.")

mongo_client = MongoClient(MONGO_URI)
print("Connection to MongoDB successful")

reviews_collection: collection = mongo_client[ATLAS_DB][ATLAS_DB_COLLECTION]
restaurants_collection: collection = mongo_client[ATLAS_DB][
    ATLAS_DB_RESTAURANTS_COLLECTION
]


# %%


def upload_to_mongo(mongo_collection, documents, batch_size=1000):
    print(
        f"Inserting {len(documents)} into {mongo_collection.database.name}.{mongo_collection.name}"
    )

    for i in range(0, len(documents), batch_size):
        insert_result = mongo_collection.insert_many(
            documents[i : i + batch_size],
            ordered=False,
        )

        print(f"Inserted batch # {i+1}." + insert_result)


# %%
reviews_collection.delete_many({})
review_documents = reviews_df.to_dict("records")
upload_to_mongo(reviews_collection, review_documents)

# %%
restaurants_collection.delete_many({})
restuaurant_documents = restuaurants_df.to_dict("records")
upload_to_mongo(restaurants_collection, restuaurant_documents)


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


def vector_search(text_query: str):

    if not text_query:
        raise ValueError("Invalid query.")

    embedding_of_query = get_embedding_vector(text_query)

    if embedding_of_query is None:
        raise ValueError("Embedding generation failed.")

    if len(embedding_of_query) != EMBEDDING_LENGTH:
        raise ValueError(
            f"Assumed {EMBEDDING_LENGTH} embedding items, but query produced embedding with {len(embedding_of_query)} items. Did you use the same embedding model?"
        )

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

ANSWER_GENERATION_LLM = "google/gemma-2b-it"
gemma_tokenizer = AutoTokenizer.from_pretrained(ANSWER_GENERATION_LLM)
gemma_model = AutoModelForCausalLM.from_pretrained(
    ANSWER_GENERATION_LLM, device_map="auto"
)

# %%
# Conduct query with retrival of sources
user_prompt = "Where are the best empanadas that are modern and have really good flavor, and the waiters are extra crispy?"
user_prompt = "Who has scruptious fried birds served with some pastry? I want real maple syrup."

search_result_documents = vector_search(text_query=user_prompt)
top_result = search_result_documents[0]

#%%
# Get info about the restaurant we will use to display later
print('Restaurant id' ,top_result['_id'])

restaurant = restaurants_collection.find_one({'_id': top_result['_id']})

restaurant and print(restaurant) 

llm_prompt = format_llm_prompt(user_prompt, top_result)

print(llm_prompt)

# %%

print("1. Tokenize prompt")
input_ids = gemma_tokenizer(llm_prompt, return_tensors="pt")

print("2. Use tokens to generate a response represented as a tensor")
tensore_response: torch.Tensor = gemma_model.generate(**input_ids, max_new_tokens=500)

print("3. Inflate tensor back into human language")
decoded_response: str = gemma_tokenizer.decode(tensore_response[0])

# resposne contains original prompt, so trim it for final result
opinion_portion = decoded_response.replace(
    llm_prompt, "<<generation prompt redacted>>"
)[5:]

print("*" * 32)
print("Here's what we figured:")
print(opinion_portion)
