# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Save the model and tokenizer
model.save_pretrained("llama-3.2-1B")
tokenizer.save_pretrained("llama-3.2-1B")

