FROM python:3.10-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download NLTK data (including WordNet!)
RUN python -c "import nltk; \
    nltk.download('wordnet', quiet=True); \
    nltk.download('omw-1.4', quiet=True); \
    nltk.download('punkt', quiet=True); \
    nltk.download('punkt_tab', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    print('✅ NLTK data downloaded successfully')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads nltk_data

# Render uses PORT environment variable (usually 10000)
ENV PORT=10000

# Expose the port
EXPOSE $PORT

# Health check to verify app is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:' + __import__('os').environ.get('PORT', '10000') + '/health')" || exit 1

# Start with Gunicorn, binding to the PORT env var
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --log-level info app:app
