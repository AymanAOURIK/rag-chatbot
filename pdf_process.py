#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path
from functools import lru_cache

# LangChain imports for PDF processing and splitting
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use HuggingFaceEmbeddings to avoid using OpenAI APIs
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Configure logging for better traceability
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@lru_cache(maxsize=128)
def count_tokens(text: str) -> int:
    """
    A simple token counter that caches results.
    Replace with your tokenizer if needed.
    """
    return len(text.split())

@lru_cache(maxsize=1)
def get_base_llm_pipeline():
    """
    Initializes and caches the base LLM pipeline for generating augmented context.
    Using google/flan-t5-small which is instruction-tuned and suitable for this task.
    """
    from transformers import pipeline
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
    return llm_pipeline

def generate_contextualized_chunk(whole_document: str, chunk_text: str) -> str:
    """
    Generates augmented contextual information for a given chunk using a base LLM.
    The function constructs a prompt that includes the whole document context and the excerpt,
    then uses the LLM to produce additional clarifications and context.
    """
    # Retrieve the base LLM pipeline.
    llm = get_base_llm_pipeline()
    # Construct a prompt that provides both the document context and the excerpt.
    prompt = (
        f"Document context: {whole_document}\n\n"
        f"Excerpt: {chunk_text}\n\n"
        "Provide additional context, background, and clarifications that enrich the content of the excerpt."
    )
    try:
        generated = llm(prompt, max_length=256, do_sample=False)[0]['generated_text']
    except Exception as e:
        logging.error(f"LLM generation error: {e}")
        generated = "[Context generation failed]"
    return f"{generated}\n\n{chunk_text}"

def load_and_process_pdfs(
    pdf_dir="pdfs",
    output_file="chunks.json",
    chunk_size=1000,
    chunk_overlap=200,
    use_contextualization=True
):
    """
    Loads PDFs from the specified directory, splits them into chunks,
    optionally augments each chunk with contextual information, and
    saves the resulting list of chunks as a JSON file.
    """
    all_chunks = []
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        logging.error(f"PDF directory '{pdf_dir}' does not exist.")
        return all_chunks

    for file in pdf_path.glob("*.pdf"):
        try:
            logging.info(f"Processing {file}...")
            loader = PyPDFLoader(str(file))
            documents = loader.load()  # Returns a list of Document objects
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(documents)
            
            for chunk in chunks:
                original_text = chunk.page_content
                token_count = count_tokens(original_text)
                logging.info(f"Chunk from {file.name} has approx. {token_count} tokens.")
                
                # Enhance the chunk with contextual information if enabled.
                if use_contextualization:
                    contextual_text = generate_contextualized_chunk("Whole document context placeholder", original_text)
                else:
                    contextual_text = original_text
                
                all_chunks.append({
                    "source": str(file),
                    "page_content": contextual_text,
                    "token_count": token_count
                })
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved {len(all_chunks)} chunks to {output_file}")
    except Exception as e:
        logging.error(f"Error saving to output file {output_file}: {e}")
    
    return all_chunks

def index_chunks_with_chromadb(chunks, collection_name="pdf_chunks"):
    """
    Indexes the processed chunks in a ChromaDB vector store.
    Each chunk's 'page_content' is embedded using a HuggingFace model.
    """
    texts = [chunk["page_content"] for chunk in chunks]
    metadata = [{"source": chunk["source"], "token_count": chunk["token_count"]} for chunk in chunks]
    
    # Initialize embeddings using a local HuggingFace model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the ChromaDB vector store from texts and metadata
    vector_store = Chroma.from_texts(texts, embeddings, metadatas=metadata, collection_name=collection_name)
    logging.info(f"Indexed {len(texts)} chunks into ChromaDB collection '{collection_name}'.")
    return vector_store

def similarity_search(vector_store, query, k=5):
    """
    Performs a similarity search against the vector store.
    """
    results = vector_store.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    logging.info("Starting PDF processing to extract text and create chunks...")
    
    # Process PDFs and create chunks with contextual augmentation.
    chunks = load_and_process_pdfs()
    
    # Index the processed chunks into ChromaDB for efficient retrieval.
    vector_store = index_chunks_with_chromadb(chunks)
    
    # Example similarity search query.
    query = "What was the revenue growth for ACME Corp in Q2 2023?"
    logging.info(f"Running similarity search for query: {query}")
    results = similarity_search(vector_store, query, k=5)
    
    logging.info("Similarity search results:")
    for i, res in enumerate(results):
        print(f"Result {i + 1}:")
        print(res)
        print()
