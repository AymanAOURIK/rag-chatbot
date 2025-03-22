#!/usr/bin/env python3
import os
import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_process_pdfs(pdf_dir="pdfs", output_file="chunks.json", chunk_size=1000, chunk_overlap=200):
    all_chunks = []
    # Iterate over PDF files in the directory
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            print(f"Processing {file_path}...")
            loader = PyPDFLoader(file_path)
            documents = loader.load()  # Load PDF as a list of Document objects
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
    print("Processing PDFs to extract text and create chunks...")
    load_and_process_pdfs()
