# %%
%pip install datasets pandas pymongo sentence-transformers
%pip install -U transformers
# Install below if using GPU
%pip install accelerate
%pip install ipywidgets

# %%
from pathlib import Path
from datasets import load_dataset
import pandas as pd

ATLAS_DB = 'demo'
ATLAS_DB_COLLECTION = 'mflix_ai'
ATLAS_VECTOR_INDEX = 'vector_index'
DOCUMENT_EMBEDDINGS_FIELD = 'embedding'

ORIG_DF_CACHE = "./.cache/original_dataset.csv"

if Path(ORIG_DF_CACHE).exists():
    dataset_df = pd.read_csv(ORIG_DF_CACHE)
else:
    # https://huggingface.co/datasets/AIatMongoDB/embedded_movies
    dataset = load_dataset("AIatMongoDB/embedded_movies")

    # Create Pandas data frame from dataset
    dataset_df = pd.DataFrame(dataset["train"])[["fullplot", "title", "rated", "genres", "runtime", "plot"]]
    dataset_df.to_csv(ORIG_DF_CACHE)


# %%
# Data Preparation

# Embedding generation breaks on null/ missing text. Drop it.
dataset_df = dataset_df.dropna(subset=["fullplot"])
print("Null data in columns", dataset_df.isnull().sum())

# %%
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("thenlper/gte-large")
asymmetric_model = SentenceTransformer(
    "sentence-transformers/msmarco-distilroberta-base-v2"
)
multi_qa_minilm_ls_cos = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# %%
# Create an embedding (a vector) from text

## Choose one of the embedding models:
# embedder = lambda text: embedding_model.encode(text)
# embedder = lambda text: asymmetric_model.encode(text)
embedder = lambda text: multi_qa_minilm_ls_cos.encode(text)

def get_embedding_vector(text: str | list[str]) -> list[float]:
    if not text.strip():
        raise ValueError("Attempted to get embedding for empty text.")
    
    result = embedder(text).tolist()

    return result

EMBEDDING_LENGTH = len(get_embedding_vector('some text'))
print(f"embedding length {EMBEDDING_LENGTH}")
print(f"destination field {DOCUMENT_EMBEDDINGS_FIELD}")
print(f"*** Create the following vector index in Atlas Search: ***")
print(f"1. Database '{ATLAS_DB}'")
print(f"2. Collection '{ATLAS_DB_COLLECTION}'")
print(f"3. Index name '{ATLAS_VECTOR_INDEX}':")
print(f'4. Index JSON:\n\t{{"fields": [{{"numDimensions": {EMBEDDING_LENGTH},"path": "{DOCUMENT_EMBEDDINGS_FIELD}","similarity": "cosine","type": "vector"}}]}}')

# %%
# Measure embedding_model's embedding nuance

prompt = "Chances of anything coming to earth"
data = [
    "The American approach to everything is Go Big or Go Home!",
    "When it comes to pizza, cover it in cheese completely.",
    "Then add cheese inside the crust, and sprinkle cheese on top.",
]

print(
    "embedding_model:",
    util.cos_sim(embedding_model.encode(prompt), embedding_model.encode(data)),
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
EMBEDDINGS_DF_CACHE = './.cache/embeddings_dataset.csv'


if Path(EMBEDDINGS_DF_CACHE).exists():
  dataset_df = pd.read_csv(EMBEDDINGS_DF_CACHE)
else:
  # This can take some time...
  dataset_df[DOCUMENT_EMBEDDINGS_FIELD] = dataset_df["fullplot"].apply(get_embedding_vector)
  dataset_df.to_csv(EMBEDDINGS_DF_CACHE)

dataset_df.info()

# %%
from pymongo import MongoClient, collection
import os

# from google.colab import userdata

MONGO_URI = os.environ.get("MONGO_URL")


if not MONGO_URI:
    raise ValueError("MONGO_URL environment variable missing or empty.")

mongo_client = MongoClient(MONGO_URI)
print("Connection to MongoDB successful")

atlas_collection: collection = mongo_client[ATLAS_DB][ATLAS_DB_COLLECTION]

# %%
# Delete any existing records in the collection
atlas_collection.delete_many({})

# %%
documents = dataset_df.to_dict("records")
insert_result = atlas_collection.insert_many(
    documents,
    ordered=False,
)

print(
    f"Inserted into {ATLAS_DB}.{ATLAS_DB_COLLECTION} {len(insert_result.inserted_ids)} documents."
)

# %%
def format_mql_query(embedding_of_query: list):
    mql_pipeline = [
        {
            "$vectorSearch": {
                "index": ATLAS_VECTOR_INDEX,
                "queryVector": embedding_of_query,
                "path": DOCUMENT_EMBEDDINGS_FIELD,
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 4,  # Return top 4 matches
            }
        },
        {
            "$project": {
                "_id": 0,  # Exclude the _id field
                "fullplot": 1,  # Include the plot field
                "title": 1,  # Include the title field
                "genres": 1,  # Include the genres field
                "score": {"$meta": "vectorSearchScore"},  # Include the search score
            }
        },
    ]

    return mql_pipeline

# %%
import json


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

    results = atlas_collection.aggregate(mql_pipeline)

    return list(results)

# %%
def combine_results_to_text(documents: list[dict]) -> str:
    result_as_text = ""
    for doc in documents:
        result_as_text += (
            f"Title: {doc.get('title', 'N/A')}\nPlot: {doc.get('fullplot', 'N/A')}\n\n"
        )

    return result_as_text

# %%
# Conduct query with retrival of sources
text_query = "I feel like aliens are watching are visiting. What movie is about this?"

search_result_documents = vector_search(text_query=text_query)

candidate_items_to_reccomend = combine_results_to_text(search_result_documents)

combined_information = f"Query: {text_query}\nContinue to answer the query by using the Search Results:\n{candidate_items_to_reccomend}."

print(combined_information)

# %%
from huggingface_hub import notebook_login
notebook_login()

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
gemma_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")

# %%
# Set up a one-shot query to generate an opinion
generation_prompt = f"Pick one of these 3 movies at random, and write who would love to see that movie.\n{candidate_items_to_reccomend}"


# %%

input_ids = gemma_tokenizer(generation_prompt, return_tensors="pt")

response = gemma_model.generate(**input_ids, max_new_tokens=500)

print(gemma_tokenizer.decode(response[0]))


