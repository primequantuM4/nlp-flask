# Use an official Python image as the base
FROM python:3.10-slim

# Set environment variables for Python and NLTK
ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/app/nltk_data

# Set the working directory inside the container
WORKDIR /app

# Copy your project files to the working directory
COPY . /app

# Install system dependencies and Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the nltk_data directory exists and download the required datasets
RUN mkdir -p /app/nltk_data && \
    pip install nltk && \
    python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data'); \
               nltk.download('wordnet', download_dir='/app/nltk_data'); \
               nltk.download('averaged_perceptron_tagger', download_dir='/app/nltk_data')"

# Expose the port your app runs on
EXPOSE 8080

# Run your application
CMD ["python", "main.py"]

