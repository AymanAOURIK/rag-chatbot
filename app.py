import logging
from flask import Flask, request, jsonify
import json
import os
from model import generate_response
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import re

# Setup logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

# Download NLTK data if needed
nltk.download('punkt')

# Determine the directory where app.py is located
here = os.path.dirname(os.path.abspath(__file__))
app.logger.debug(f"app.py directory: {here}")

# Load preprocessed chunks from file using an absolute path
with open(os.path.join(here, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Prepare the BM25 retriever by tokenizing each chunk's text.
documents = [chunk["page_content"] for chunk in chunks]
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

def refine_response(raw_response: str, max_new_tokens=300) -> str:
    """
    Refines the raw response by:
      1. Removing internal markers.
      2. Instructing the model to generate a final answer in a structured format.
      3. Extracting only the final answer.
      
    The final answer will:
      - Start with a brief summary/introduction that sets the stage for the answer.
      - Provide a domain-specific structured answer with clear headings and bullet points covering key sectors.
      - Include three contextual sections based on reranked documents, each indicating the document name and a concise summary.
    """
    # Remove lines starting with "[Contextualized]:" and normalize whitespace.
    cleaned = re.sub(r'\[Contextualized\]:.*?\n', '', raw_response)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Enhanced instruction prompt to generate a structured, domain-specific answer.
    instruction = (
        "You are a highly sophisticated AI expert in international development, global policies, and humanitarian affairs. "
        "Based solely on the analysis provided below, produce a final answer that is direct, comprehensive, and formatted in a structured way. "
        "Start with a brief summary or introduction that sets the stage for the answer. Then, present the domain-specific answer with clear headings "
        "and bullet points covering key sectors such as health, education, poverty alleviation, etc. "
        "After the domain-specific answer, include three contextual sections derived from reranked documents. "
        "Each context should clearly indicate the document name and provide a concise summary of the relevant information. "
        "Do not include any internal processing details or chain-of-thought notes. "
        "Output only the final answer, starting with 'Final Answer:' followed by the structured answer."
    )

    # Construct the prompt for refinement.
    new_prompt = f"{instruction}\n\nAnalysis:\n{cleaned}\n\nFinal Answer:"
    
    # Generate the refined answer.
    final_response_raw = generate_response(new_prompt, max_new_tokens=max_new_tokens)
    
    # Extract only the part after the last occurrence of "Final Answer:".
    if "Final Answer:" in final_response_raw:
        parts = final_response_raw.split("Final Answer:")
        final_answer = parts[-1].strip()
    else:
        final_answer = final_response_raw.strip()
    
    # Further clean stray markers.
    final_answer = re.sub(r'\[.*?\]', '', final_answer).strip()
    
    return final_answer

@app.route("/ask", methods=["GET", "POST"])
def ask():
    try:
        if request.method == "GET":
            return jsonify({
                "message": "Welcome to the RAG Chatbot API. Please use POST to ask a question with JSON payload: {'question': 'your question here'}"
            })

        data = request.get_json()
        app.logger.debug(f"Received data: {data}")

        question = data.get("question", "")
        if not question:
            app.logger.error("No question provided in the request")
            return jsonify({"error": "Question is required"}), 400

        # Tokenize the query
        tokenized_query = word_tokenize(question.lower())
        app.logger.debug(f"Tokenized query: {tokenized_query}")

        # Get BM25 scores for each document
        scores = bm25.get_scores(tokenized_query)
        app.logger.debug(f"BM25 scores: {scores}")

        top_n = 3
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        context = "\n\n".join([documents[i] for i in top_indices])
        app.logger.debug(f"Context for response: {context}")

        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        app.logger.debug(f"Generated prompt: {prompt}")

        raw_response = generate_response(prompt)
        app.logger.debug(f"Raw response: {raw_response}")

        refined_response = refine_response(raw_response)
        return jsonify({"response": refined_response})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "RAG Chatbot API is running."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
