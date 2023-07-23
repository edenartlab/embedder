FROM python:3.8-slim-buster

WORKDIR /app

ADD . /app

RUN pip install --no-cache-dir torch torchvision numpy pillow pymongo requests tqdm

EXPOSE 80

CMD ["python", "main.py"]
