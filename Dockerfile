FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y git build-essential cmake unzip

ADD . /app

# Download and install the latest version of SQLite
RUN apt-get install -y wget \
    && wget https://www.sqlite.org/2023/sqlite-dll-win64-x64-3420000.zip \
    && unzip sqlite-dll-win64-x64-3420000.zip \
    && cp sqlite3.dll /usr/local/bin

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD ["python", "main.py"]
