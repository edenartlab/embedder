print("new version #10...")

import sys
sys.path.append('CLIP_assisted_data_labeling')
sys.path.append('creator-lora')

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
from creator_lora.models.resnet50 import ResNet50MLP

MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')
CHROMA_HOST = os.getenv('CHROMA_HOST')
print("CHROMA HOST IS", CHROMA_HOST)

# model_path = "combo_2023-08-02_03:48:00_8.1k_imgs_80_epochs_-1.0000_mse.pth"
model_path = "eden_scorer_2023-12-13_9.4k_imgs_80_epochs_2_crops.pth"
model_path_resnet50_mlp = "aesthetic_score_best_model.pth"
device = "cpu"

# setup mongo
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
creations = db['creations']
generators = db['generators']
embeddings = db['embeddings']

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
    # chroma_client.delete_collection(name="creation_clip_embeddings")
    collection = chroma_client.get_or_create_collection(name="creation_clip_embeddings")
    print(chroma_client.list_collections())
    print(f"Clip embeddings collection size: {collection.count()}")
except Exception as e:
    print(f"Failed to connect to chroma: {e}")
    print("Chroma is not available")

# load scorer + embedder
aesthetic_regressor = AestheticRegressor(model_path, device)
aesthetic_regressor_resnet50 = ResNet50MLP(
    model_path=model_path_resnet50_mlp,
    device = device
)
print(aesthetic_regressor)
print(aesthetic_regressor_resnet50)

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
    score_resnet50 = aesthetic_regressor_resnet50.predict_score(image)
    embedding = features.squeeze().numpy().tolist()

    ## take mean score
    score = (score + score_resnet50) / 2
    
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

        print("upsert to id ", str(document['_id']))
        print(embedding)
        print("--->")

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

        embeddings.update_one(
            {'creation': document['_id']},
            {
                '$set': {
                    'embedding': embedding
                }
            },
            upsert=True
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
    sort_order_newest = [("createdAt", -1)]
    sort_order_oldest = [("createdAt", 1)]

    batch_size_newest = 30
    batch_size_oldest = 5
    processed_count = 0
    inductions = 0

    print(f"scan for last {batch_size_newest} creations and first {batch_size_oldest} creations")

    # Fetch newest documents
    cursor_newest = creations.find(query).sort(sort_order_newest).limit(batch_size_newest)
    batch_newest = list(cursor_newest)

    # Fetch oldest documents
    cursor_oldest = creations.find(query).sort(sort_order_oldest).limit(batch_size_oldest)
    batch_oldest = list(cursor_oldest)

    # Combine both batches
    batch = batch_newest + batch_oldest

    for doc in batch:
        try:
            print("induct:", doc["_id"], doc["thumbnail"])
            induct_creation(doc)
            inductions += 1
        except Exception as e:
            print(f"error for creation {doc['_id']}: {e}")

    processed_count += len(batch)
    cursor_newest.close()
    cursor_oldest.close()

    print(f"Total number of creations scanned through: {processed_count}, inductions: {inductions}")



while True:
    try:
        print("Hello embedder")
        #scan_unembedded_creations()    
    except Exception as e:
        print(e)    
    time.sleep(100)
