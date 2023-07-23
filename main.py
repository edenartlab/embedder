print("a1")
import time
import os
import requests
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
#from dotenv import load_dotenv
import torch
#import clip
from PIL import Image
import numpy as np
import zlib


print("a2")



import torch
import clip
from PIL import Image

print("a3")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("a4")


MONGO_URI = os.getenv('MONGO_URI')

print("a5")
print("MONOGP yay open now", MONGO_URI)

client = MongoClient(MONGO_URI)
db = client['eden-dev']
collection = db['creations']

print("a6")

while True:

    print("a7")

    for document in collection.find(limit=50):
        print(document)
        
        
        # Load and preprocess the image
        image_path = document['uri']
        
        
        image = Image.open(image_path).convert("RGB")


        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy()
        
        print("embedding is here ", embedding)
        print("the shape is", embedding.shape)
        
        time.sleep(1)


    time.sleep(5)