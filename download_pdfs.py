import os
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import concurrent.futures

# Google Drive folder link
folder_url = "https://drive.google.com/drive/folders/1LxRcSNE67H0uzVsmaLhxWKP0qWqtLRC8?usp=sharing"

# Directory to save PDFs
output_dir = "pdfs"
os.makedirs(output_dir, exist_ok=True)

def extract_folder_id(url):
    match = re.search(r"folders/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def get_drive_file_ids(folder_id):
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(folder_url, headers=headers)

    if response.status_code != 200:
        print("Failed to access Google Drive folder.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    file_ids = set(re.findall(r"https://drive.google.com/file/d/([a-zA-Z0-9_-]+)", response.text))
    return list(file_ids)

def download_file(file_id, directory):
    file_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(file_url, stream=True)
    file_name = f"{file_id}.pdf"

    if "Content-Disposition" in response.headers:
        file_name_match = re.findall('filename="(.+)"', response.headers["Content-Disposition"])
        if file_name_match:
            file_name = file_name_match[0]

    file_path = os.path.join(directory, file_name)

    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    print(f"Downloaded: {file_name}")

# Main flow
if __name__ == "__main__":
    folder_id = extract_folder_id(folder_url)

    if folder_id:
        file_ids = get_drive_file_ids(folder_id)

        if file_ids:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(lambda fid: download_file(fid, output_dir), file_ids)
        else:
            print("No files found in the folder.")
    else:
        print("Invalid Google Drive folder URL.")
