import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

# Ensure the Hugging Face token is set
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set. Please export HF_TOKEN before running.")

# Hugging Face model ID
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Configure 4-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer with trust_remote_code enabled
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN, trust_remote_code=True)

# Load model configuration and patch rope_scaling to the expected format
config = AutoConfig.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
    # Replace the rope_scaling with only the supported keys: "type" and "factor"
    config.rope_scaling = {"type": "linear", "factor": config.rope_scaling.get("factor", 8.0)}

# Load model with the patched configuration and trust_remote_code enabled
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN,
    trust_remote_code=True
)

def generate_response(prompt, max_new_tokens=512):
    """
    Generate a response from LLaMA 3.1 with a 4-bit quantized model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
