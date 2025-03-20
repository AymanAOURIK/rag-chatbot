# pdf_process.py
import os
import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional: Function to download a PDF from Google Drive using its file ID.
# Requires "gdown" to be installed (see requirements.txt)
def download_pdf_from_drive(file_id, output_path):
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from {url} to {output_path}")
    gdown.download(url, output_path, quiet=False)

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
    # If you need to download a PDF from Google Drive, uncomment and supply the file_id and output_path:
    # download_pdf_from_drive("YOUR_GOOGLE_DRIVE_FILE_ID", "pdfs/example.pdf")
    
    load_and_process_pdfs()
