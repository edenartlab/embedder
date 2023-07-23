import torch
from PIL import Image
import numpy as np
import zlib

# Load the CLIP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = torch.hub.load('openai/clip', 'clip400', device=device)
model.eval()

# Load and preprocess the image
image_path = '/Users/genekogan/Mars/Media/2023/mars23highlights_org/organized/workshop/P1013339-E72C3.jpg'
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

# Save the embedding as a zlib-compressed numpy array
save_path = 'embedding.npy.z'
with open(save_path, 'wb') as f:
    np.save(f, embedding)
    f.seek(0)
    compressed_data = zlib.compress(f.read())

compressed_save_path = 'embedding.npy.z'
with open(compressed_save_path, 'wb') as f:
    f.write(compressed_data)
