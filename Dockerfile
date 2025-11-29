# Use slim Python base
FROM python:3.10-slim

# Avoid creation of .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/usr/local/nltk_data

# Set workdir
WORKDIR /app

# Install OS-level dependencies, create uploads and NLTK dirs in one step
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
    && mkdir -p /app/uploads $NLTK_DATA \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .

RUN pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader -d $NLTK_DATA punkt stopwords wordnet

# Copy app code
COPY . .

# Expose port (Render sets PORT env; informative)
EXPOSE 5000

# Default environment variables
ENV PORT=5000
ENV FLASK_ENV=production

# Start the app
CMD ["python", "app.py"]
