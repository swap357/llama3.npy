import os
import torch
import numpy as np
import argparse
import warnings
import json
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from config import ModelArgs, HF_MODEL_PATH, HF_TOKENIZER_PATH

DTYPE = torch.float32

warnings.filterwarnings("ignore", category=UserWarning, message=".*`do_sample` is set to `False`*")

def tensor_to_dict(tensor, name, token_idx=None):
    """Convert tensor to dictionary with statistics, matching llama3_stats format."""
    # Convert to numpy for consistency
    full_np_tensor = tensor.detach().cpu().numpy()

    # For matching llama3_stats format, we need to keep the shape as-is
    # and extract specific token data for stats calculation only
    if token_idx is not None:
        # For statistics calculation only, extract the token
        seq_len_dim = 1  # Assuming shape is (batch, seq_len, features)
        if -full_np_tensor.shape[seq_len_dim] <= token_idx < full_np_tensor.shape[seq_len_dim]:
            stats_tensor = full_np_tensor[:, token_idx]
            if stats_tensor.shape[0] == 1:
                stats_tensor = stats_tensor.squeeze(0)
        else:
            stats_tensor = full_np_tensor.copy()
    else:
        stats_tensor = full_np_tensor.copy()

    # Calculate stats on the extracted tensor for statistics
    mean_val = float(stats_tensor.mean()) if stats_tensor.size > 0 else 0.0
    std_val = float(stats_tensor.std()) if stats_tensor.size > 0 else 0.0
    min_val = float(stats_tensor.min()) if stats_tensor.size > 0 else 0.0
    max_val = float(stats_tensor.max()) if stats_tensor.size > 0 else 0.0

    # For data, keep the full structure to match llama3_stats format
    data = full_np_tensor.tolist()

    return {
        "name": name,
        "shape": [int(dim) for dim in full_np_tensor.shape],  # Match llama3_stats format
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "data": data  # Full tensor data with original structure
    }


def save_token_stats(stats_dict, token_idx, output_dir):
    """Save statistics for a token generation step"""
    output_dir = Path(output_dir)

    # Create directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return  # Stop if directory cannot be created

    # Save each tensor's stats to a separate file
    for tensor_name, tensor_stats in stats_dict.items():
        file_path = output_dir / f"token_{token_idx}_{tensor_name}.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(tensor_stats, f, indent=2)
        except Exception:
            pass  # Continue with other files if one fails


def generate_text_manual(prompt: str, args: ModelArgs, output_dir: str = "hf_stats") -> str:
    """Generate text token by token and save intermediate tensor statistics."""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Generate tokens one by one
    for i in range(args.max_new_tokens):
        # Dictionary to store all tensor stats for this token
        token_stats = {}

        # Get embeddings for current sequence
        with torch.no_grad():
            # 1. Get embeddings
            hidden_states = model.model.embed_tokens(generated_ids)
            token_stats["embeddings"] = tensor_to_dict(hidden_states, "embeddings", -1)

            # 2. Process through each layer
            for layer_idx in range(model.config.num_hidden_layers):
                layer = model.model.layers[layer_idx]
                layer_prefix = f"layer_{layer_idx}"

                # Input layernorm
                norm_out = layer.input_layernorm(hidden_states)
                token_stats[f"{layer_prefix}_input_norm"] = tensor_to_dict(norm_out, f"{layer_prefix}_input_norm", -1)

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
                token_stats[f"{layer_prefix}_attention_output"] = tensor_to_dict(attn_output, f"{layer_prefix}_attention_output", -1)

                # First residual connection
                hidden_states = hidden_states + attn_output
                token_stats[f"{layer_prefix}_after_first_residual"] = tensor_to_dict(hidden_states, f"{layer_prefix}_after_first_residual", -1)

                # Post attention norm
                norm_out = layer.post_attention_layernorm(hidden_states)
                token_stats[f"{layer_prefix}_post_attention_norm"] = tensor_to_dict(norm_out, f"{layer_prefix}_post_attention_norm", -1)

                # MLP
                gate = layer.mlp.gate_proj(norm_out)
                up = layer.mlp.up_proj(norm_out)
                gate_activated = torch.nn.functional.silu(gate)
                mlp_output = layer.mlp.down_proj(gate_activated * up)
                token_stats[f"{layer_prefix}_mlp_output"] = tensor_to_dict(mlp_output, f"{layer_prefix}_mlp_output", -1)

                # Second residual connection
                hidden_states = hidden_states + mlp_output
                token_stats[f"{layer_prefix}_final_output"] = tensor_to_dict(hidden_states, f"{layer_prefix}_final_output", -1)

            # Final norm
            hidden_states = model.model.norm(hidden_states)
            token_stats["final_norm"] = tensor_to_dict(hidden_states, "final_norm", -1)

            # Get logits for next token
            logits = model.lm_head(hidden_states)
            token_stats["logits"] = tensor_to_dict(logits, "logits", -1)

            # Save all stats for this token
            save_token_stats(token_stats, i, output_dir)

            # Get next token (greedy decoding)
            next_token = torch.argmax(logits[0, -1])

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            generated_text = tokenizer.decode(generated_ids[0])

            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text and save intermediate tensor statistics")
    parser.add_argument('--prompt', type=str, default="Once upon a time", help="Input prompt for generation")
    parser.add_argument('--output-dir', type=str, default="hf_stats", help="Directory to save tensor statistics")
    parser.add_argument('--max-new-tokens', type=int, default=10, help="Maximum number of tokens to generate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args_cli = parser.parse_args()

    hf_args = ModelArgs()
    hf_args.max_new_tokens = args_cli.max_new_tokens
    hf_args.seed = args_cli.seed

    text = generate_text_manual(args_cli.prompt, hf_args, args_cli.output_dir)
    print(text)