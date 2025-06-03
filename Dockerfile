FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p models

# Make port 8000 available
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Run the application
CMD ["python", "run.py", "api", "--host", "0.0.0.0", "--port", "8000"] 