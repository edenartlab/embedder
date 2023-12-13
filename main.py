print("new version #9...")

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
from chromadb.config import Settings
from utils.embedder import AestheticRegressor

MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')
CHROMA_HOST = os.getenv('CHROMA_HOST')
print("CHROMA HOST IS", CHROMA_HOST)

#model_path = "combo_2023-08-02_03:48:00_8.1k_imgs_80_epochs_-1.0000_mse.pth"
model_path = "eden_scorer_2023-12-13_9.4k_imgs_80_epochs_2_crops.pth"
device = "cpu"

# setup mongo
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
creations = db['creations']
generators = db['generators']

# setup chroma
try:
    chroma_client = chromadb.HttpClient(
        host=CHROMA_HOST, 
        port=8000,
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
            chroma_client_auth_credentials="chromadb:changeme"
        )
    )
    print(chroma_client)
    collection = chroma_client.get_or_create_collection(name="creation_clip_embeddings")
    print(chroma_client.list_collections())
    print(f"Clip embeddings collection size: {collection.count()}")
except Exception as e:
    print(f"Failed to connect to chroma: {e}")
    print("Chroma is not available")

# load scorer + embedder
aesthetic_regressor = AestheticRegressor(model_path, device)
print(aesthetic_regressor)


def induct_creation(document):
    uri = document['thumbnail']

    if not uri:
        print(f"skip creation {document['_id']}, no thumbnail")
        return
    
    response = requests.get(uri)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    if not image.mode or not image.size:
        print(f"skip creation {document['_id']}, invalid image")
        return

    # aesthetic score
    score, features = aesthetic_regressor.predict_score(image)
    embedding = features.squeeze().numpy().tolist()
    
    if not embedding:
        print(f"skip creation {document['_id']}, no embedding")
        return
    
    # check if score is valid (should be >0)
    if score <= 0:
        print(f"skip creation {document['_id']}, invalid score={score}")
        return

    try:
        # add to chroma
        collection.upsert(
            embeddings=[embedding],
            metadatas=[{"user": str(document['user'])}],
            ids=[str(document['_id'])]
        )

        # update mongo
        creations.update_one(
            {'_id': document['_id']},
            {
                '$set': {
                    'embedding': {
                        # 'embedding': embedding,
                        'score': score
                    }
                }
            }
        )
        
        print(f"inducted creation {document['_id']}")

    except Exception as e:
        print(f"error for creation {document['_id']}: {e}")
        return
    

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
