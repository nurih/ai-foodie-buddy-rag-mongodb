from pathlib import Path
import os

ATLAS_DB = "demo"
ATLAS_DB_COLLECTION = "restaurant_reviews"
ATLAS_DB_RESTAURANTS_COLLECTION = "restaurant"
ATLAS_VECTOR_INDEX = "restaurant_reviews_index"
DOCUMENT_EMBEDDINGS_FIELD = "embedding"
REVIEWS_DIR = Path(os.environ.get("USERPROFILE")).joinpath(
    "Downloads", "restaurant_dataset"
)

LLM_PROMPT_CACHE = "./.cache/llm_prompt.txt"
