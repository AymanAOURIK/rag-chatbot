# Use CUDA base image with Ubuntu
FROM nvcr.io/nvidia/cuda:12.6.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV BITSANDBYTES_NO_TRITON=1
ENV DEBIAN_FRONTEND=noninteractive

# Install OS and Python dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python (optional, avoids ambiguity)
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Download NLTK tokenizer data
RUN python -m nltk.downloader punkt

# Download PDFs before startup
RUN python3 download_pdfs.py

# Process the PDFs into chunks (embeds them)
RUN python3 pdf_process.py

# Expose the port the app runs on
EXPOSE 8080

# Start the Flask app with Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "app:app"]
