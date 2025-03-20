# model.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure the Hugging Face token is set via environment variable.
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("Hugging Face token is not set. Make sure to export HF_TOKEN before running.")

# Set your model ID – here we use Meta-Llama 3.1 8B Instruct.
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load Model and Tokenizer using the provided token.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=HF_TOKEN
)

def generate_response(prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Simple test prompt
    sample_prompt = "What data types does Milvus support?"
    print("Generated response:", generate_response(sample_prompt))
