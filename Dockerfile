FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/app/nltk_data

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8080

CMD ["python", "main.py"]

