import time
import os
import requests
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import torch
#import clip



MONGO_URI = os.getenv('MONGO_URI')

print("MONOGP", MONGO_URI)


client = MongoClient(MONGO_URI)
db = client['eden-dev']
collection = db['creations']


while True:

    for document in collection.find(limit=25):
        print(document)
        time.sleep(1)

    time.sleep(5)