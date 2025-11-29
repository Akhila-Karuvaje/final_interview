# Use slim Python base
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OS-level dependencies needed for audio/video/ML packages + NLTK downloader tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    libasound2 \
    libsm6 \
    libxext6 \
    libgl1 \
    pkg-config \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install first (leverages Docker cache)
COPY requirements.txt .

RUN pip install --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Set NLTK data directory and download corpora
ENV NLTK_DATA=/usr/local/nltk_data
RUN mkdir -p $NLTK_DATA
RUN python -m nltk.downloader -d $NLTK_DATA punkt stopwords wordnet

# Copy app code
COPY . .

# Create uploads dir
RUN mkdir -p /app/uploads

EXPOSE 5000

ENV PORT=5000
ENV FLASK_ENV=production

CMD ["python", "app.py"]
