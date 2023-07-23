import os
import requests
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import torch
import clip

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')

print("MONOGP", MONGO_URI)


# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client['eden-stg']
collection = db['creations']


# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Iterate through documents in the collection
for document in collection.find(limit=5):
    # Get the URL field from the document
    print(document)
    url = document.get('uri')

    # Load the image from the URL using Pillow
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    print(image.size)
    # Perform any image manipulation or processing here
    # For example, you can display the image
    #image.show()
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Embed the image using CLIP
    with torch.no_grad():
        image_embedding = model.encode_image(image_tensor)

    # Print the embedding vector
    print("Image Embedding:", image_embedding.tolist())
