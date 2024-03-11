import os

from pymongo import MongoClient, collection

from settings import ATLAS_DB, ATLAS_DB_COLLECTION, ATLAS_DB_RESTAURANTS_COLLECTION


MONGO_URI = os.environ.get("MONGO_URL")

if not MONGO_URI:
    raise ValueError("MONGO_URL environment variable missing or empty.")

mongo_client = MongoClient(MONGO_URI)
print("Connection to MongoDB successful")

reviews_collection: collection = mongo_client[ATLAS_DB][ATLAS_DB_COLLECTION]
restaurants_collection: collection = mongo_client[ATLAS_DB][
    ATLAS_DB_RESTAURANTS_COLLECTION
]
