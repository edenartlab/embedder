print("new version #5")
print("e1C 222 aaaa bbb")
import sys
sys.path.append('CLIP_assisted_data_labeling')
print("e2D333 cccc")

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

print("e3")

IN_DEV = True
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')
CHROMA_HOST = os.getenv('CHROMA_HOST')
print("HOST IS", CHROMA_HOST)
"

model_path = "combo_2023-08-02_03:48:00_8.1k_imgs_80_epochs_-1.0000_mse.pth"
device = "cpu"
generator_names = ["create", "remix", "blend", "upscale", "real2real", "interpolate", "wav2lip"]
print("e4")

# setup mongo
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
creations = db['creations']
generators = db['generators']
print("e5")

import chromadb
from chromadb.config import Settings


# # setup chroma
print("try chroma again !!!", CHROMA_HOST)
try:

    print("lets auth now")

    client = chromadb.HttpClient(
        host=CHROMA_HOST, 
        port=8000,
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
            chroma_client_auth_credentials="chromadb:changeme"
        )
    )

    print(client)

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


print("AESTH", aesthetic_regressor)

def induct_creation(document):
    uri = document['thumbnail']
    print("lets go", uri)
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

    print("SET SCIOR", score)
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
    
    # # add to chroma
    # collection.upsert(
    #     embeddings=[embedding],
    #     metadatas=[{"user": str(document['user'])}],
    #     ids=[str(document['_id'])]
    # )

    print(f"inducted creation {document['_id']}")




def scan_unembedded_creations():
    query = {
        "thumbnail": {"$regex": r"\.webp$"},  # Filter for documents where "thumbnail" ends with ".webp"
    }
    sort_order = [("createdAt", -1)]  # Assuming there's an "insertion_timestamp" field

    batch_size = 1000
    processed_count = 0
    inductions = 0

    while True:
        cursor = creations.find(query).sort(sort_order).skip(processed_count).limit(batch_size)
        
        batch = list(cursor)
        if not batch:
            # No more documents to process
            print(f"Total number of creations scanned through: {processed_count}, inductions: {inductions}")

            break

        for doc in batch:
            try:
                print(":) _- induct -_ -> ", doc["thumbnail"])
                induct_creation(doc)
                inductions += 1
            except Exception as e:
                print(f"error for creation {doc['_id']}: {e}")

        processed_count += len(batch)
        cursor.close()

    client.close()


while True:
    try:
        print("hello embedder! 3")
        scan_unembedded_creations()
    except Exception as e:
        print(e)
    time.sleep(5)
