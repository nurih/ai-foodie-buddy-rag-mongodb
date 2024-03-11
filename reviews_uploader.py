# %%
from embedder import MULTI_QA_MINILM_L6_COS_V1, Embedder
from pathlib import Path
from settings import DOCUMENT_EMBEDDINGS_FIELD, REVIEWS_DIR
from mongo_atlas import reviews_collection, restaurants_collection
import pandas as pd


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
EMBEDDINGS_DF_CACHE = "./.cache/embeddings_dataset.csv"
embedder = Embedder(MULTI_QA_MINILM_L6_COS_V1)

# %%
if Path(EMBEDDINGS_DF_CACHE).exists():
    reviews_df = pd.read_csv(EMBEDDINGS_DF_CACHE)
    if "Unnamed: 0" in reviews_df.columns:
        reviews_df.drop("Unnamed: 0", axis=1, inplace=True)
        print("Extra column dropped....")

else:
    print("Generating embeddings... this can take some time...")
    reviews_df[DOCUMENT_EMBEDDINGS_FIELD] = reviews_df["text"].apply(embedder.encode)
    reviews_df.to_csv(EMBEDDINGS_DF_CACHE, index=False)

reviews_df.info()

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

if reviews_collection.find_one({}):
    print("Restaurant Reviews already populated. Remove all documents to upload again.")
else:
    review_documents = reviews_df.to_dict("records")
    upload_to_mongo(reviews_collection, review_documents)

# %%

if restaurants_collection.find_one({}):
    print(
        "Restaurants already already populated. Remove all documents to upload again."
    )
else:
    restuaurant_documents = restuaurants_df.to_dict("records")
    upload_to_mongo(restaurants_collection, restuaurant_documents)
