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


# Load the CLIP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model, preprocess = torch.hub.load('openai/clip', 'clip400', device=device)
model, preprocess = torch.hub.load('openai/clip', 'ViT-B/32', device=device)

model.eval()

print("test beginning 22")

print("THIS IS MODEL", model)

MONGO_URI = os.getenv('MONGO_URI')

print("MONOGP yay", MONGO_URI)

client = MongoClient(MONGO_URI)
db = client['eden-dev']
collection = db['creations']


while True:

    for document in collection.find(limit=25):
        print(document)
        
        
        # Load and preprocess the image
        image_path = document['uri']
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224), resample=Image.BICUBIC)
        image = np.array(image) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0).to(device)

        # Generate the image embedding
        with torch.no_grad():
            image_features = model.encode_image(image)

        # Normalize the embedding
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Convert the embedding to a numpy array
        embedding = image_features.cpu().numpy()
        
        print("embedding", embedding)
        
        time.sleep(1)


    time.sleep(5)