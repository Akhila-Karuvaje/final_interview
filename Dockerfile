FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ✅ KEY FIX: Use 'from nltk import downloader' to avoid circular import
RUN python -c "from nltk import downloader; \
    d = downloader.Downloader(); \
    d.download('wordnet'); \
    d.download('omw-1.4'); \
    d.download('punkt'); \
    d.download('punkt_tab'); \
    d.download('stopwords'); \
    print('✅ NLTK data downloaded')"

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads nltk_data

# Environment variables
ENV PORT=10000
ENV NLTK_DATA=/root/nltk_data

EXPOSE $PORT

# Start app with Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app
