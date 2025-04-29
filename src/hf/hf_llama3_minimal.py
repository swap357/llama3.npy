from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch, time
import sys
import os

# Add repo root to path to import from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import ModelArgs, HF_MODEL_PATH, HF_TOKENIZER_PATH

def generate(prompt=None, args=None):
    if prompt is None:
        prompt = "Once upon a time"

    if args is None:
        args = ModelArgs()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        HF_TOKENIZER_PATH,
        trust_remote_code=True
    )

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    streamer = TextStreamer(tokenizer)

    start = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        streamer=streamer,
        do_sample=args.do_sample
    )
    elapsed = time.time() - start

    tokens_generated = output_ids.shape[-1] - inputs.input_ids.shape[-1]

    print("\n<generation>")
    print(f"Token count: {tokens_generated}, elapsed: {elapsed:.2f}s, {tokens_generated/elapsed:.0f} tokens/s")

    return output_ids

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    generate(prompt)

if __name__ == "__main__":
    main()
