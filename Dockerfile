# Use GPU-accelerated PyTorch with compilers pre-installed
FROM huggingface/transformers-pytorch-gpu

WORKDIR /app

# Avoid Python writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Optional: system tools & compilers
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    curl \
    wget \
    libgl1-mesa-glx \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and install everything
RUN pip install --upgrade pip && pip install -r requirements.txt

# Add your app code
COPY . .

EXPOSE 8080

# Run your Flask app
CMD ["python", "app.py"]
