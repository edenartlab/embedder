# syntax=docker/dockerfile:1.5
FROM python:3.10-slim-bookworm as builder
LABEL org=edenartlab
WORKDIR /app


# Copy only the file needed to install apt dependencies.
# This utilizes Docker's cache, because if these files have not changed,
# then we don't need to re-install the dependencies.
COPY apt-packages.txt apt-packages.txt

# No need for 2 layers of apt package installs
RUN apt-get update --fix-missing -y \
      && apt-get install -y --fix-missing $(cat apt-packages.txt) \
      && rm -rf /var/lib/apt/lists/*  # Clean up to reduce layer size

COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

RUN (git clone https://github.com/aiXander/CLIP_assisted_data_labeling && \
    cd CLIP_assisted_data_labeling && \
    git checkout 09468d9)

RUN (git clone https://github.com/edenartlab/creator-lora.git && \
    cd creator-lora && \
    git checkout fdf201cf7b6a1a7ce3bb281527e451bd50b68778)

COPY main.py main.py

# RUN gdown 1Pm2apRxk9CbMspwve3Fir3WwX-vDWXoL
RUN gdown 1iEcUy-fAe2h_3E7gMI4pu8tsZ187n4aC

# aesthetic_score_best_model.pth
RUN gdown 1thEIlXVc8lkULVUBY9Ab45tsOERxkjxn

EXPOSE 80

CMD ["python", "-u", "main.py"]
