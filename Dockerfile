FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads nltk_data /root/nltk_data

# Set environment variables
ENV PORT=10000
ENV NLTK_DATA=/root/nltk_data
ENV PYTHONUNBUFFERED=1

EXPOSE $PORT

# ✅ CRITICAL: NLTK data will download at RUNTIME (first app start)
# This avoids the circular import issue during Docker build

# Start app with Gunicorn (REMOVE --preload flag!)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 180 app:app"]
