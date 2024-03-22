# %%
import os
from pathlib import Path
import json
import pandas as pd

# %%
reviews_dir = Path(os.environ.get("USERPROFILE")).joinpath(
    "Downloads", "restaurant_dataset"
)

# %%
BUSINESSES_RAW = reviews_dir.joinpath("meta-Hawaii.json")
BUSINESSES_SLIM = reviews_dir.joinpath("restaurants.json")
BUSINESS_KEYS = [
  "name",
  "address",
  "gmap_id",
  "description",
  "latitude",
  "longitude",
  "category",
  "avg_rating",
  "num_of_reviews",
  "price",
]
#%%
max_lines = 1000000
written = 0
with open(BUSINESSES_RAW, "r", encoding="utf8") as source_file:
    with open(
            BUSINESSES_SLIM, "w", buffering=1, encoding="utf8"
        ) as destination_file:
        for line in source_file:
            if "estaurant" in line:
                original = json.loads(line)
                slim = {k: v for k, v in original.items() if k in BUSINESS_KEYS}
                destination_file.write(json.dumps(slim))
                destination_file.write("\n")
                written += 1
                if written % 1000 == 0:
                    print(f"{written}", end=" ")
                if written >= max_lines:
                    break

# %%

restuaruant_df = pd.read_json(BUSINESSES_SLIM, lines=True)

restuaruant_df.head(3)

# print('MUTTqe8uqyMdBl186RmNeA' in restuaruant_df.gmap_id.values)


# %%
REVIEWS_RAW = reviews_dir.joinpath("review-Hawaii_10.json")
REVIEWS_SLIM = reviews_dir.joinpath("reviews.json")
REVIEWS_KEYS = []


max_lines = 3_000_000
written = 0
with open(REVIEWS_RAW, "r", encoding="utf8", buffering=1) as source_file:
    for line in source_file:
        with open(REVIEWS_SLIM, "a", buffering=1, encoding="utf8") as destination_file:
            original = json.loads(line)
            if not original["gmap_id"] in restuaruant_df.gmap_id.values:
                continue

            destination_file.write(line)  # verbatim, no need to reconvert
            written += 1
            if written % 100 == 0:
                print(f"{written}", end=" ")
            if written >= max_lines:
                break

#%%
