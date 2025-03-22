# pdf_process.py
import os
import requests
from urllib.parse import urlparse, parse_qs
import concurrent.futures
import json
from bs4 import BeautifulSoup

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Directory to save the PDFs
output_dir = "pdfs"
os.makedirs(output_dir, exist_ok=True)

# Google Drive folder link (public)
DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/1LxRcSNE67H0uzVsmaLhxWKP0qWqtLRC8?usp=sharing"

def get_pdf_urls_from_drive_folder(folder_url):
    # Attempt to scrape the folder page (likely will return nothing)
    response = requests.get(folder_url)
    if response.status_code != 200:
        raise Exception(f"Failed to access folder URL: {folder_url}")
    soup = BeautifulSoup(response.text, "html.parser")
    pdf_urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" in href.lower():
            if not href.startswith("http"):
                href = "https://drive.google.com" + href
            pdf_urls.append(href)
    pdf_urls = list(set(pdf_urls))
    print(f"Found {len(pdf_urls)} PDF URLs in the folder (via scraping).")
    return pdf_urls

def get_pdf_urls_from_file(file_path="pdf_urls.txt"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(urls)} PDF URLs from {file_path}.")
        return urls
    else:
        print(f"No file {file_path} found.")
        return []

def download_file(url, directory):
    response = requests.get(url, stream=True)
    parsed = urlparse(url)
    file_name = os.path.basename(parsed.path)
    if not file_name.lower().endswith(".pdf"):
        qs = parse_qs(parsed.query)
        file_id = qs.get("id", [None])[0]
        file_name = f"{file_id}.pdf" if file_id else "downloaded.pdf"
    file_path = os.path.join(directory, file_name)
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

def download_pdfs():
    # First, try to load URLs from file.
    pdf_urls = get_pdf_urls_from_file()
    # If no URLs from file, fallback to scraping (which may return 0)
    if not pdf_urls:
        pdf_urls = get_pdf_urls_from_drive_folder(DRIVE_FOLDER_LINK)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(download_from_url, pdf_urls)

def load_and_process_pdfs(pdf_dir="pdfs", output_file="chunks.json", chunk_size=1000, chunk_overlap=200):
    all_chunks = []
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            print(f"Processing {file_path}...")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(documents)
            for chunk in chunks:
                all_chunks.append({
                    "source": file_path,
                    "page_content": chunk.page_content
                })
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_chunks)} chunks to {output_file}")
    return all_chunks

if __name__ == "__main__":
    print("Downloading PDFs...")
    download_pdfs()
    print("Processing PDFs to extract text and create chunks...")
    load_and_process_pdfs()
