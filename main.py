print("new version #7")

import sys
sys.path.append('CLIP_assisted_data_labeling')

import time
import os
import requests
from bson.objectid import ObjectId
import pymongo
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
import torch
import chromadb
from utils.embedder import AestheticRegressor


IN_DEV = True
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')
CHROMA_HOST = os.getenv('CHROMA_HOST')
print("CHROMA HOST IS", CHROMA_HOST)


model_path = "combo_2023-08-02_03:48:00_8.1k_imgs_80_epochs_-1.0000_mse.pth"
device = "cpu"
generator_names = ["create", "remix", "blend", "upscale", "real2real", "interpolate", "wav2lip"]

# setup mongo
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
creations = db['creations']
generators = db['generators']



import chromadb
from chromadb.config import Settings

# setup chroma
try:
    print("try auth chroma", CHROMA_HOST)
    chroma_client = chromadb.HttpClient(
        host=CHROMA_HOST, 
        port=8000,
        settings=Settings(
            allow_reset=True,
            anonymized_telemetry=False,
            chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
            chroma_client_auth_credentials="chromadb:changeme"
        )
    )

    print(chroma_client)

    # chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
    collection = chroma_client.get_or_create_collection(name="creation_clip_embeddings")
    print(chroma_client.list_collections())
    print(f"Clip embeddings collection size: {collection.count()}")
except Exception as e:
    print(f"Failed to connect to chroma: {e}")
    print("Chroma is not available")

# pre-populate chroma with all creations which already have embeddings
# pipeline2 = [
#     {
#         "$match": {
#             "embedding": {"$exists": True},
#             "embedding.embedding": {"$exists": True},
#             "embedding.score": {"$exists": True}
#         }
#     }
# ]

# pipeline = [
#     {
#         "$match": {
#             "user": {"$exists": True}
#         }
#     },
#     {
#         "$limit": 50
#     }
# ]

# print("e7")
# documents = creations.aggregate(pipeline)
# print("e9")
# for document in documents:
#     print(document)
#     # collection.upsert(
#     #     embeddings=[document['embedding']['embedding']],
#     #     metadatas=[{"user": str(document['user'])}],
#     #     ids=[str(document['_id'])]
#     # )
# print("e8")

# print(f"Chroma now has {collection.count()} creations")

# # load scorer + embedder
aesthetic_regressor = AestheticRegressor(model_path, device)

print(aesthetic_regressor)

def scan_unembedded_creations():
    query = {
        "thumbnail": {"$regex": r"\.webp$"},
        "embedding.score": {"$exists": False}
    }
    sort_order = [("createdAt", -1)]  # Assuming there's an "insertion_timestamp" field

    batch_size = 100
    processed_count = 0
    inductions = 0

    print(f"scan for last {batch_size} creations")

    cursor = creations.find(query).sort(sort_order).skip(processed_count).limit(batch_size)
    
    batch = list(cursor)

    for doc in batch:
        try:
            print("induct:", doc["_id"], doc["thumbnail"])
            induct_creation(doc)
            inductions += 1
        except Exception as e:
            print(f"error for creation {doc['_id']}: {e}")

    processed_count += len(batch)
    cursor.close()

    print(f"Total number of creations scanned through: {processed_count}, inductions: {inductions}")



while True:
    try:
        print("Hello embedder!")
        scan_unembedded_creations()
    except Exception as e:
        print(e)
    time.sleep(5)
