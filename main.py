import time
import os
import requests
from bson.objectid import ObjectId
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
import torch
import clip
import chromadb

MONGO_URI = os.getenv('MONGO_URI')
CHROMA_HOST = os.getenv('CHROMA_HOST')

# setup mongo
client = MongoClient(MONGO_URI)
db = client['eden-dev']
creations = db['creations']
generators = db['generators']

# setup chroma
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
collection = chroma_client.get_or_create_collection(name="creation_clip_embeddings")
print(chroma_client.list_collections())
print(collection.count())
print(collection.peek())

# setup CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_encoder, preprocess = clip.load("ViT-B/32", device=device)


def induct_creation(document):
        
    # embed with CLIP
    response = requests.get(document['uri'])
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_encoder.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    embedding = image_features.cpu().numpy()
    
    # aesthetic score
    score = 0
    
    # update mongo
    creations.update_one(
        {'_id': document['_id']},
        {
            '$set': {
                'embedding.embedding': embedding.tolist(),
                'embedding.score': score
            }
        }
    )
    
    # add to chroma
    collection.upsert(
        embeddings=[embedding[0].tolist()],
        metadatas=[{"user": str(document['user'])}],
        ids=[str(document['_id'])]
    )


def scan_unembedded_creations():
    PAGE_SIZE = 100
    
    generator_names = ["create", "remix"]
    generator_ids = [g['_id'] for g in generators.find({
        "generatorName": {"$in": generator_names}
    })]
    
    count = 0
    last_id = None
    
    while True:
        pipeline = [
            {
                "$lookup": {
                    "from": "tasks",
                    "localField": "task",
                    "foreignField": "_id",
                    "as": "task_info"
                }
            },
            {
                "$unwind": "$task_info"
            },
            {
                "$lookup": {
                    "from": "generators",
                    "localField": "task_info.generator",
                    "foreignField": "_id",
                    "as": "task_info.generator_info"
                }
            },
            {
                "$unwind": "$task_info.generator_info"
            },
            {
                "$match": {
                    "embedding": None,
                    "task_info.generator_info._id": {"$in": generator_ids}
                }
            }
        ]
        
        # paginate
        if last_id is not None:
            pipeline.append({
                "$match": {
                    "_id": {"$gt": last_id}
                }
            })
        
        pipeline.append({"$limit": PAGE_SIZE})
        cursor = creations.aggregate(pipeline)
        page_documents = list(cursor)
        
        if not page_documents:
            return
        
        for document in page_documents:
            try:
                induct_creation(document)
                count += 1
            except Exception as e:
                print(f"error for creation {document['_id']}: {e}")
        
        last_id = page_documents[-1]['_id']
    
    print(f"Total number of creations scanned through: {count}")
        

while True:
    try:
        #scan_unembedded_creations() #disable for a minute
        print("hello")
    except Exception as e:
        print(e)
    time.sleep(1)
