from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

# Ensure the Hugging Face token is set in the environment.
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set. Please export HF_TOKEN before running.")

# Specify the model identifier
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)

# Load the model with 8-bit quantization.
# The parameter `ignore_mismatched_sizes=True` is used to bypass tensor shape mismatches.
# Use it with caution—ideally, the model and checkpoint should be fully compatible.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    load_in_8bit=True,
    ignore_mismatched_sizes=True
)

def generate_response(prompt, max_new_tokens=128):
    """
    Generate a response for a given prompt using the quantized Llama 3.1 model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
