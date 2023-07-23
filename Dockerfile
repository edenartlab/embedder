FROM python:3.8-slim-buster

WORKDIR /app

ADD . /app

RUN pip install --no-cache-dir torch numpy pillow pymongo requests

EXPOSE 80

CMD ["python", "main.py"]
