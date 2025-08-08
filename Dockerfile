# Use Python 3.10 full image (not slim to avoid missing libs)
FROM python:3.10

# Avoid interactive prompts during package installs
ARG DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies (removed duplicate and added stability)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    python3-tk \
    wget \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads processed static/css static/js templates

# Set environment variables
ENV FLASK_APP=app.py \
    FLASK_ENV=production \
    PYTHONPATH=/app \
    PORT=5000

# Expose the app port
EXPOSE $PORT

# Create a non-root user and switch to it
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check for Render
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/ || exit 1

# Start the application using Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 300 --keep-alive 2 --max-requests 1000 app:app
