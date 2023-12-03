import sys
sys.path.append('CLIP_assisted_data_labeling')

import time
import os
import requests
from bson.objectid import ObjectId
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
import torch
import chromadb
from utils.embedder import AestheticRegressor

MONGO_URI = os.getenv('MONGO_URI')
CHROMA_HOST = os.getenv('CHROMA_HOST')
model_path = "combo_2023-08-02_03:48:00_8.1k_imgs_80_epochs_-1.0000_mse.pth"
device = "cpu"
generator_names = ["create", "remix", "blend", "upscale", "real2real", "interpolate", "wav2lip"]

# setup mongo
client = MongoClient(MONGO_URI)
db = client['eden-dev']
creations = db['creations']
generators = db['generators']

# # setup chroma
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
collection = chroma_client.get_or_create_collection(name="creation_clip_embeddings")
print(chroma_client.list_collections())
print(f"Clip embeddings collection size: {collection.count()}")

# pre-populate chroma with all creations which already have embeddings
# pipeline = [
#     {
#         "$match": {
#             "embedding": {"$exists": True},
#             "embedding.embedding": {"$exists": True},
#             "embedding.score": {"$exists": True}
#         }
#     }
# ]
# documents = creations.aggregate(pipeline)
# for document in documents:
#     collection.upsert(
#         embeddings=[document['embedding']['embedding']],
#         metadatas=[{"user": str(document['user'])}],
#         ids=[str(document['_id'])]
#     )

# print(f"Chroma now has {collection.count()} creations")

# # load scorer + embedder
aesthetic_regressor = AestheticRegressor(model_path, device)


print("AESTH", aesthetic_regressor)

# def induct_creation(document):
#     uri = document['thumbnail']

#     if not uri:
#         print(f"skip creation {document['_id']}, no thumbnail")
#         return
    
#     response = requests.get(uri)
#     image = Image.open(BytesIO(response.content)).convert("RGB")

#     if not image.mode or not image.size:
#         print(f"skip creation {document['_id']}, invalid image")
#         return

#     # aesthetic score
#     score, features = aesthetic_regressor.predict_score(image)
#     embedding = features.squeeze().numpy().tolist()
    
#     if not embedding:
#         print(f"skip creation {document['_id']}, no embedding")
#         return
    
#     # update mongo
#     creations.update_one(
#         {'_id': document['_id']},
#         {
#             '$set': {
#                 'embedding': {
#                     'embedding': embedding,
#                     'score': score
#                 }
#             }
#         }
#     )
    
#     # add to chroma
#     collection.upsert(
#         embeddings=[embedding],
#         metadatas=[{"user": str(document['user'])}],
#         ids=[str(document['_id'])]
#     )

#     print(f"inducted creation {document['_id']}")


# def scan_unembedded_creations():
#     PAGE_SIZE = 100
    
#     generator_ids = [g['_id'] for g in generators.find({
#         "generatorName": {"$in": generator_names}
#     })]
    
#     count = 0
#     last_id = None
    
#     while True:
#         pipeline = [
#             {
#                 "$lookup": {
#                     "from": "tasks",
#                     "localField": "task",
#                     "foreignField": "_id",
#                     "as": "task_info"
#                 }
#             },
#             {
#                 "$unwind": "$task_info"
#             },
#             {
#                 "$lookup": {
#                     "from": "generators",
#                     "localField": "task_info.generator",
#                     "foreignField": "_id",
#                     "as": "task_info.generator_info"
#                 }
#             },
#             {
#                 "$unwind": "$task_info.generator_info"
#             },
#             {
#                 "$match": {
#                     "$or": [
#                         {"embedding": None},
#                         {"embedding.embedding": {"$exists": False}},
#                         {"embedding.embedding": None},
#                         {"embedding.embedding": {"$size": 0}},
#                         {"embedding.score": {"$exists": False}},
#                         {"embedding.score": None}
#                     ],
#                     "task_info.generator_info._id": {"$in": generator_ids}
#                 }
#             }
#         ]
        
#         # paginate
#         if last_id is not None:
#             pipeline.append({
#                 "$match": {
#                     "_id": {"$gt": last_id}
#                 }
#             })
        
#         pipeline.append({"$limit": PAGE_SIZE})
#         cursor = creations.aggregate(pipeline)
#         page_documents = list(cursor)
        
#         if not page_documents:
#             print(f"Total number of creations scanned through: {count}")
#             return
        
#         print(f"Try to induct {len(page_documents)} creations")
#         for document in page_documents:
#             try:
#                 induct_creation(document)
#                 count += 1
#             except Exception as e:
#                 print(f"error for creation {document['_id']}: {e}")
        
#         last_id = page_documents[-1]['_id']



while True:
    try:
        #scan_unembedded_creations()
        print("hello embedder!")
    except Exception as e:
        print(e)
    time.sleep(5)
