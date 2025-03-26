from flask import Flask, request, jsonify
import json
from model import generate_response
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import re

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

def refine_response(raw_response: str, max_new_tokens=200) -> str:
    """
    Cleans the raw response by removing internal markers and instructs the model,
    acting as a highly sophisticated AI expert in international development, to produce a final answer.
    The function then extracts only the final answer (the text following the "Final Answer:" marker)
    to ensure no internal instructions appear in the output.
    """
    # Remove lines starting with "[Contextualized]:" and normalize whitespace.
    cleaned = re.sub(r'\[Contextualized\]:.*?\n', '', raw_response)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Define an enhanced expert instruction prompt.
    instruction = (
        "You are a highly sophisticated AI system with deep expertise in international development, "
        "global policies, and humanitarian affairs. Disregard all internal analysis and chain-of-thought details. "
        "Based solely on the analysis provided below, provide a final answer that is direct, concise, and free of any internal markers. "
        "Do not repeat or include any internal processing details—only output the final answer after the marker 'Final Answer:' and nothing else."
    )

    # Combine the instruction and the cleaned analysis to form a new prompt.
    new_prompt = f"{instruction}\n\nAnalysis:\n{cleaned}\n\nFinal Answer:"
    
    # Generate the refined answer.
    final_response_raw = generate_response(new_prompt, max_new_tokens=max_new_tokens)
    
    # Extract only the part after "Final Answer:" marker.
    if "Final Answer:" in final_response_raw:
        final_answer = final_response_raw.split("Final Answer:", 1)[1].strip()
    else:
        final_answer = final_response_raw.strip()
    
    # Optional: Further remove any stray square bracket markers.
    final_answer = re.sub(r'\[.*?\]', '', final_answer).strip()
    return final_answer

@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "GET":
        return jsonify({
            "message": (
                "Welcome to the RAG Chatbot API. Please use POST to ask a question with JSON payload: "
                "{'question': 'your question here'}"
            )
        })

    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Retrieve top relevant chunks via BM25.
    tokenized_query = word_tokenize(question.lower())
    scores = bm25.get_scores(tokenized_query)
    top_n = 3
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    context = "\n\n".join([documents[i] for i in top_indices])

    # Create the prompt for generation.
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    
    # Generate the raw response.
    raw_response = generate_response(prompt)
    
    # Refine the raw response to produce a clean final answer.
    refined_response = refine_response(raw_response)
    
    return jsonify({"response": refined_response})

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "RAG Chatbot API is running."})

if __name__ == "__main__":
    # Run the app on the desired port.
    app.run(host="0.0.0.0", port=8080, debug=True)
