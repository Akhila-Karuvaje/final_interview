# Use slim Python base
FROM python:3.10-slim

# Avoid creation of .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OS-level dependencies needed for audio/video/ML packages
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
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install first (leverages Docker cache)
COPY requirements.txt .

# Upgrade pip and install wheel first to reduce build issues
RUN pip install --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK corpora at build time to avoid runtime errors
RUN python -m nltk.downloader punkt stopwords wordnet

# Copy app code
COPY . .

# Create uploads dir (app uses uploads/)
RUN mkdir -p /app/uploads

# Expose port (Render sets PORT env; this is just informative)
EXPOSE 5000

# Default environment variables (can be overridden in Render/production)
ENV PORT=5000
ENV FLASK_ENV=production

# Start the app
CMD ["python", "app.py"]
