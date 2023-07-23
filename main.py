import time
import os
import requests
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import clip


MONGO_URI = os.getenv('MONGO_URI')

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

client = MongoClient(MONGO_URI)
db = client['eden-dev']
collection = db['creations']


while True:

    for document in collection.find():
        
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