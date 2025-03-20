# app.py
from flask import Flask, request, jsonify
import json
from model import generate_response
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data if needed
nltk.download('punkt')

app = Flask(__name__)

# Load preprocessed chunks from file
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Prepare the BM25 retriever by tokenizing each chunk's text.
documents = [chunk["page_content"] for chunk in chunks]
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# GET endpoint to check API status or provide instructions for /ask
@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "GET":
        return jsonify({
            "message": "Welcome to the RAG Chatbot API. "
                       "Please use the POST method to ask a question with JSON payload: {'question': 'your question here'}"
        })
    
    # Process POST request
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question is required"}), 400

    tokenized_query = word_tokenize(question.lower())
    scores = bm25.get_scores(tokenized_query)
    top_n = 3
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    context = "\n\n".join([documents[i] for i in top_indices])
    
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    response = generate_response(prompt)
    return jsonify({"response": response})

# GET endpoint to check API status
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "RAG Chatbot API is running."})

if __name__ == "__main__":
    # Run the app on port 8888
    app.run(host="0.0.0.0", port=8888, debug=True)
