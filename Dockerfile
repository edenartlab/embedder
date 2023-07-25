FROM python:3.10-slim-bookworm as builder

WORKDIR /app

ADD . /app

RUN apt-get update --fix-missing 

RUN apt-get install -y --fix-missing \
    git build-essential cmake

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD ["python", "main.py"]
