# pdf_process.py
import os
import requests
from urllib.parse import urlparse, parse_qs
import concurrent.futures
import json
import re
from bs4 import BeautifulSoup

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Directory to save the PDFs
output_dir = "pdfs"
os.makedirs(output_dir, exist_ok=True)

# Google Drive folder link (public)
DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/1LxRcSNE67H0uzVsmaLhxWKP0qWqtLRC8?usp=sharing"

def get_pdf_urls_from_drive_folder(folder_url):
    """
    Fetch the folder page HTML and parse it to extract PDF download links.
    Note: This approach relies on the public folder HTML structure.
    """
    response = requests.get(folder_url)
    if response.status_code != 200:
        raise Exception(f"Failed to access folder URL: {folder_url}")
    soup = BeautifulSoup(response.text, "html.parser")
    
    pdf_urls = []
    # Look for all anchor tags that might contain PDF links.
    # Google Drive sometimes returns links in format "/uc?id=FILE_ID&export=download"
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Check if the link points to a PDF (simple check)
        if ".pdf" in href.lower():
            # If it's a relative link, make it absolute
            if not href.startswith("http"):
                href = "https://drive.google.com" + href
            pdf_urls.append(href)
    
    # Remove duplicates if any
    pdf_urls = list(set(pdf_urls))
    print(f"Found {len(pdf_urls)} PDF URLs in the folder.")
    return pdf_urls

def download_file(url, directory):
    response = requests.get(url, stream=True)
    # Attempt to extract file id from the URL if filename is not obvious
    parsed = urlparse(url)
    file_name = os.path.basename(parsed.path)
    if not file_name.lower().endswith(".pdf"):
        # Try to get file id from query params and build a filename
        qs = parse_qs(parsed.query)
        file_id = qs.get("id", [None])[0]
        file_name = f"{file_id}.pdf" if file_id else "downloaded.pdf"
    file_path = os.path.join(directory, file_name)

    # Write content to file
    with open(file_path, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=1024):
            fd.write(chunk)
    return file_path

def download_from_url(url):
    try:
        file_path = download_file(url, output_dir)
        print(f'Successfully downloaded file from {url} to {file_path}')
    except Exception as e:
        print(f'Failed to download file from {url}. Reason: {e}')

def download_pdfs_from_drive(folder_url):
    # Get list of PDF URLs from the Google Drive folder page.
    pdf_urls = get_pdf_urls_from_drive_folder(folder_url)
    # Use ThreadPoolExecutor to download PDFs in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(download_from_url, pdf_urls)

def load_and_process_pdfs(pdf_dir="pdfs", output_file="chunks.json", chunk_size=1000, chunk_overlap=200):
    all_chunks = []
    # Iterate over PDF files in the directory
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            print(f"Processing {file_path}...")
            loader = PyPDFLoader(file_path)
            documents = loader.load()  # Returns a list of Document objects
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(documents)
            for chunk in chunks:
                all_chunks.append({
                    "source": file_path,
                    "page_content": chunk.page_content
                })
    # Save the chunks to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_chunks)} chunks to {output_file}")
    return all_chunks

if __name__ == "__main__":
    # Step 1: Download PDFs from the Google Drive folder
    print("Downloading PDFs from Google Drive folder...")
    download_pdfs_from_drive(DRIVE_FOLDER_LINK)
    
    # Step 2: Process downloaded PDFs to extract and chunk text
    print("Processing PDFs to extract text and create chunks...")
    load_and_process_pdfs()

