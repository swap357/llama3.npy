import torch
import argparse
import warnings
import sys
import os
import time

# Add repo root to path to import from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import ModelArgs, HF_MODEL_PATH, HF_TOKENIZER_PATH

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

DTYPE = torch.float32

warnings.filterwarnings("ignore", category=UserWarning, message=".*`do_sample` is set to `False`*")

def generate_text_manual(prompt: str, args: ModelArgs = None) -> str:
    """Generate text token by token and save intermediate tensor statistics."""
    if args is None:
        args = ModelArgs()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    batch_size, seq_len = input_ids.shape

    generated_ids = input_ids.clone()
    generated_text = ""

    # Print the prompt without a newline
    print(prompt, end="", flush=True)

    # Track tokens and time
    tokens_generated = 0
    start_time = time.time()

    # Generate tokens one by one
    for i in range(args.max_new_tokens):

        # Get embeddings for current sequence
        with torch.no_grad():
            # 1. Get embeddings
            hidden_states = model.model.embed_tokens(generated_ids)

            # 2. Process through each layer
            for layer_idx in range(model.config.num_hidden_layers):
                layer = model.model.layers[layer_idx]
                layer_prefix = f"layer_{layer_idx}"

                # Input layernorm
                norm_out = layer.input_layernorm(hidden_states)

                # Prepare attention inputs
                seq_len = generated_ids.shape[1]
                causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=model.device))
                causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
                position_ids = torch.arange(seq_len, device=model.device).unsqueeze(0)

                # Compute RoPE embeddings
                head_dim = model.config.hidden_size // model.config.num_attention_heads
                base = model.config.rope_theta
                inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=model.device, dtype=torch.float32) / head_dim))
                t = torch.arange(seq_len, dtype=torch.float32, device=model.device)
                freqs = torch.outer(t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                freqs_cos = emb.cos()[None, :, :].to(dtype=hidden_states.dtype)
                freqs_sin = emb.sin()[None, :, :].to(dtype=hidden_states.dtype)
                position_embeddings = (freqs_cos, freqs_sin)

                # Self attention
                attn_output, _ = layer.self_attn(
                    hidden_states=norm_out,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )

                # First residual connection
                hidden_states = hidden_states + attn_output

                # Post attention norm
                norm_out = layer.post_attention_layernorm(hidden_states)

                # MLP
                gate = layer.mlp.gate_proj(norm_out)
                up = layer.mlp.up_proj(norm_out)
                gate_activated = torch.nn.functional.silu(gate)
                mlp_output = layer.mlp.down_proj(gate_activated * up)

                # Second residual connection
                hidden_states = hidden_states + mlp_output

            # Final norm
            hidden_states = model.model.norm(hidden_states)

            # Get logits for next token
            logits = model.lm_head(hidden_states)

            # Get next token (greedy decoding)
            next_token = torch.argmax(logits[0, -1])

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

            # Decode the new token and print it inline
            new_token_text = tokenizer.decode(next_token.unsqueeze(0))
            print(new_token_text, end="", flush=True)

            generated_text = tokenizer.decode(generated_ids[0])
            tokens_generated += 1

            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Print a newline after generation is complete
    print()

    # Calculate and print stats
    elapsed = time.time() - start_time
    print(f"Token count: {tokens_generated}, elapsed: {elapsed:.2f}s, {tokens_generated/elapsed:.0f} tokens/s")

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text and save intermediate tensor statistics")
    parser.add_argument('--prompt', type=str, default="Once upon a time", help="Input prompt for generation")
    parser.add_argument('--max-new-tokens', type=int, default=10, help="Maximum number of tokens to generate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args_cli = parser.parse_args()

    hf_args = ModelArgs()
    hf_args.max_new_tokens = args_cli.max_new_tokens
    hf_args.seed = args_cli.seed

    generate_text_manual(args_cli.prompt, hf_args)


if __name__ == "__main__":
    main()
