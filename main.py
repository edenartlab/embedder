# __import__('sqlite3')
# import sys
# sys.modules['pysqlite3'] = sys.modules.pop('sqlite3')
# import chromadb

import time
import os
import requests
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import clip
#dotenv
from dotenv import load_dotenv
load_dotenv()



print("lets do chromadb")
import chromadb
chroma_client = chromadb.HttpClient(host="chromadb.mars", port=8000)
#chroma_client = chromadb.HttpClient(host="chromadb.eden.art", port=8000)
print(chroma_client)
print("great 33")


collection = chroma_client.create_collection(name="test1")
#collection = chroma_client.get_collection(name="test1")


docs = [
    {"doc": "cat", "metadata": {"tag": "animal"}, "id": "id1"},
    {"doc": "dog", "metadata": {"tag": "animal"}, "id": "id2"},
    {"doc": "pig", "metadata": {"tag": "animal"}, "id": "id3"},
    {"doc": "blue", "metadata": {"tag": "color"}, "id": "id4"},
    {"doc": "red", "metadata": {"tag": "color"}, "id": "id5"},
    {"doc": "green", "metadata": {"tag": "color"}, "id": "id6"},
    {"doc": "France", "metadata": {"tag": "country"}, "id": "id7"},
    {"doc": "Germany", "metadata": {"tag": "country"}, "id": "id8"},    
]


collection.add(
    documents=[doc['doc'] for doc in docs],
    metadatas=[doc['metadata'] for doc in docs],
    ids=[doc['id'] for doc in docs]
)

results = collection.query(
    query_texts=["Brazil"],
    n_results=3
)

print(results)


print("now the rest....")

print("GO!!!!")

MONGO_URI = os.getenv('MONGO_URI')

print("CONNECT", MONGO_URI)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

client = MongoClient(MONGO_URI)
db = client['eden-dev']
collection = db['creations']

# docker build -t embedder .
# docker run -p 27017:27017 -it embedder
# 27017



'''

save to Embeddings collection
_id
creation
embedding
knn


'''


while True:

    for document in collection.find():
        print("new doc")
        try:
            print(document)
            
            response = requests.get(document['uri'])
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy()

            print(document['_id'], document['uri'], embedding.shape)

        except Exception as e:
            print("ERROR", e)
            continue
        

    time.sleep(5)
    