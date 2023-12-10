FROM python:3.10-slim-bookworm as builder
LABEL org=edenartlab
WORKDIR /app

ADD . /app

RUN apt-get update --fix-missing 

RUN apt-get install -y --fix-missing \
    git build-essential cmake \
    libgl1-mesa-glx libglib2.0-0

RUN pip install --no-cache-dir -r requirements.txt

RUN gdown 1Pm2apRxk9CbMspwve3Fir3WwX-vDWXoL

RUN git clone https://github.com/aiXander/CLIP_assisted_data_labeling

EXPOSE 80

CMD ["python", "-u", "main.py"]
