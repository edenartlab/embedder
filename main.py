print("main 2")

import time
import os
import requests
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
#import numpy as np
import torch
import clip
#dotenv

#from dotenv import load_dotenv
#load_dotenv()

print("begin this process")


print("lets do chromadb!")
import chromadb
chroma_client = chromadb.HttpClient(host="chromadb.mars", port=8000)
print("great")
print(chroma_client)

print("all collections!")
print(chroma_client.list_collections())
print("---all collections done----")

collection = chroma_client.get_or_create_collection(name="test4")
print(collection)

print("stats :)")
print(collection.count())

print("peek")
print(collection.peek())

print("get documents")
docs2 = collection.get(
    include=["documents"]
)
print(docs2)

print("get documents 2")
docs3 = collection.get()
print(docs3)


print("yay")



docs = [
    {"doc": "cat.", "metadata": {"tag": "animal"}, "id": "idd1", "embedding": [1.2512, -0.5, 2.9] },
    {"doc": "dog.", "metadata": {"tag": "animal"}, "id": "idd2", "embedding": [1.0512, -0.6, 2.7]},
    {"doc": "pig.", "metadata": {"tag": "animal"}, "id": "idd3", "embedding": [1.4512, -0.4, 3.1]},
    {"doc": "blue.", "metadata": {"tag": "color"}, "id": "idd4", "embedding": [-1.2512, 2.5, 1.8]},
    {"doc": "red.", "metadata": {"tag": "color"}, "id": "idd5", "embedding": [-1.3512, 2.4, 1.9]},
    {"doc": "green.", "metadata": {"tag": "color"}, "id": "idd6", "embedding": [-1.2612, 2.2, 1.6]},
    {"doc": "France.", "metadata": {"tag": "country"}, "id": "idd7", "embedding": [0.2512, 1.5, -2.0]},
    {"doc": "Germany.", "metadata": {"tag": "country"}, "id": "idd8", "embedding": [0.3512, 1.4, -2.1]},    
]

# print("lets add")
# print([doc['doc'] for doc in docs])
# print([doc['metadata'] for doc in docs])
# print([doc['id'] for doc in docs])

collection.add(
    documents=[doc['doc'] for doc in docs],
    embeddings=[doc['embedding'] for doc in docs],
    metadatas=[doc['metadata'] for doc in docs],
    ids=[doc['id'] for doc in docs]
)

print("done adding, check again")


print("stats")
print(collection.count())

print("peek")
print(collection.peek())

print("get documents")
docs2 = collection.get(
    include=["documents"]
)
print(docs2)

print("get documents 2")
docs3 = collection.get()
print(docs3)


print("query")

results = collection.query(
    query_embeddings=[[1.113, -0.72, 2.4]],
    n_results=12
)

print(results)




print("now the rest....")





MONGO_URI = os.getenv('MONGO_URI')

print("CONNECT", MONGO_URI)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

client = MongoClient(MONGO_URI)
db = client['eden-dev']
collection = db['creations']


print("-----------------------")

if True:

    for document in collection.find(limit=10):
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
        
    print("sleep")
    time.sleep(5)

print("-----------------------")

while True:
    print("sleep")
    time.sleep(120)

print("this is finished")
    





# print("lets do chromadb 5 345345 5")
# import chromadb
# #chroma_client = chromadb.Client()
# #chroma_client = chromadb.HttpClient(host="chromadb.mars", port=8000)
# chroma_client = chromadb.HttpClient(host="chromadb.eden.art", port=8000)
# print(chroma_client)
# print("great 33 98u2")


# print("hello 1")

# collection = chroma_client.get_or_create_collection(name="test4")
# #collection = chroma_client.get_collection(name="test4")

# print(collection)
# print("hello 2")
# # collection = chroma_client.get_collection(name="test1")


# docs = [
#     {"doc": "cat", "metadata": {"tag": "animal"}, "id": "id1", "embedding": [1.2, -0.5, 2.9] },
#     {"doc": "dog", "metadata": {"tag": "animal"}, "id": "id2", "embedding": [1.0, -0.6, 2.7]},
#     {"doc": "pig", "metadata": {"tag": "animal"}, "id": "id3", "embedding": [1.4, -0.4, 3.1]},
#     {"doc": "blue", "metadata": {"tag": "color"}, "id": "id4", "embedding": [-1.2, 2.5, 1.8]},
#     {"doc": "red", "metadata": {"tag": "color"}, "id": "id5", "embedding": [-1.3, 2.4, 1.9]},
#     {"doc": "green", "metadata": {"tag": "color"}, "id": "id6", "embedding": [-1.25, 2.2, 1.6]},
#     {"doc": "France", "metadata": {"tag": "country"}, "id": "id7", "embedding": [0.2, 1.5, -2.0]},
#     {"doc": "Germany", "metadata": {"tag": "country"}, "id": "id8", "embedding": [0.3, 1.4, -2.1]},    
# ]

# # print("lets add")
# # print(collection)
# # print("hello 3")

# # collection.add(
# #     documents=[doc['doc'] for doc in docs],
# #     embeddings=[doc['embedding'] for doc in docs],
# #     metadatas=[doc['metadata'] for doc in docs],
# #     ids=[doc['id'] for doc in docs]
# # )

# print("hello 3")
# # print([doc['doc'] for doc in docs])
# # print([doc['metadata'] for doc in docs])
# # print([doc['id'] for doc in docs])
# print(collection)

# results = collection.query(
#     query_embeddings=[[1.11, -0.72, 2.4]],
#     n_results=3
# )

# print(results)


# print("now the rest....")

# print("GO!!!!")

# MONGO_URI = os.getenv('MONGO_URI')

# print("CONNECT", MONGO_URI)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# client = MongoClient(MONGO_URI)
# db = client['eden-dev']
# collection = db['creations']

# # docker build -t embedder .
# # docker run -p 27017:27017 -it embedder
# # 27017



# '''

# save to Embeddings collection
# _id
# creation
# embedding
# knn


# '''


# while True:

#     # for document in collection.find(limit=3):
#     #     print("new doc")
#     #     try:
#     #         print(document)
            
#     #         response = requests.get(document['uri'])
#     #         image = Image.open(BytesIO(response.content)).convert("RGB")
#     #         image = preprocess(image).unsqueeze(0).to(device)

#     #         with torch.no_grad():
#     #             image_features = model.encode_image(image)
#     #         image_features /= image_features.norm(dim=-1, keepdim=True)
#     #         embedding = image_features.cpu().numpy()

#     #         print(document['_id'], document['uri'], embedding.shape)

#     #     except Exception as e:
#     #         print("ERROR", e)
#     #         continue
        
#     print("sleep")
#     time.sleep(5000)
    