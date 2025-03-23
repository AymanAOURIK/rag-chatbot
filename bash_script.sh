#!/bin/bash
# -----------------------------------------------------------
# Script: deploy_chatbot_no_venv.sh
# Purpose: Clone the GitHub repository, install dependencies globally,
#          process PDFs, and launch the chatbot app.
# -----------------------------------------------------------

# Step 1: Clone the GitHub repository if it doesn't exist.
REPO_URL="https://github.com/AymanAOURIK/rag-chatbot.git"
REPO_DIR="rag-chatbot"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository from $REPO_URL..."
    git clone "$REPO_URL"
else
    echo "Repository already exists. Pulling latest changes..."
    cd "$REPO_DIR" || exit 1
    git pull
    cd ..
fi

# Step 2: Change directory to the repository.
cd "$REPO_DIR" || exit 1

# Step 4: Upgrade pip and install the required dependencies globally.
echo "Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

apt-get update && apt-get install vim
pip install pypdf
pip install sentence_transformers
pip install huggingface_hub==0.13.3
pip install --upgrade sentence-transformers
pip install --upgrade torch torchvision torchaudio


python download_pdfs.py
# Step 5: Process the PDFs.
echo "Processing PDFs to extract text and create chunks..."
python pdf_process.py

pip install -U bitsandbytes
# Step 6: Launch the chatbot Flask API on port 8888.
echo "Starting the chatbot application on port 8888..."
python app.py
