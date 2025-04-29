import sys
import time
import os
import torch
import numpy as np
import argparse
import warnings
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)
from config import ModelArgs, HF_MODEL_PATH, HF_TOKENIZER_PATH

DTYPE = torch.float32

warnings.filterwarnings("ignore", category=UserWarning, message=".*`do_sample` is set to `False`*")
def generate_text(prompt: str, args: ModelArgs) -> str:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_PATH, trust_remote_code=True)

    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    load_time = time.time() - load_start

    print(f"HF Model Class: {type(model)}")
    print(f"HF Model Config RMS Norm Epsilon: {model.config.rms_norm_eps}")

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    print("\nUser: " + prompt)
    print("\nllama3.2-1B: ", end="", flush=True)

    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = len(inputs.input_ids[0])
    print(f"HF Input IDs: {inputs.input_ids.tolist()}")

    hf_tensors = {}
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs.input_ids)
        hf_tensors['embeddings'] = embeddings[0, -1, :].cpu().numpy()

        first_norm = model.model.layers[0].input_layernorm(embeddings)
        hf_tensors['first_norm'] = first_norm[0, -1, :].cpu().numpy()

        outputs = model(**inputs, output_hidden_states=True)
        batch_size, seq_len = inputs.input_ids.shape

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=model.device)).view(1, 1, seq_len, seq_len)
        position_ids = torch.arange(seq_len, device=model.device).unsqueeze(0)

        head_dim = model.config.hidden_size // model.config.num_attention_heads
        base = model.config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=model.device, dtype=torch.float32) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=model.device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        freqs_cos = emb.cos()[None, :, :].to(dtype=embeddings.dtype)
        freqs_sin = emb.sin()[None, :, :].to(dtype=embeddings.dtype)
        position_embeddings = (freqs_cos, freqs_sin)

        attn_output, _ = model.model.layers[0].self_attn(
            hidden_states=first_norm,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings
        )
        hf_tensors['attn_output'] = attn_output[0, -1, :].cpu().numpy()

        residual_1 = embeddings + attn_output
        hf_tensors['residual_1'] = residual_1[0, -1, :].cpu().numpy()

        norm_layer = model.model.layers[0].post_attention_layernorm
        original_dtype = norm_layer.weight.dtype
        norm_layer.weight.data = norm_layer.weight.data.float()
        print(f"\nHF PostAttnNorm Weight (L0): Mean={norm_layer.weight.mean():.6f}, Std={norm_layer.weight.std():.6f}, Shape={norm_layer.weight.shape}")
        print(f"  First 5: {norm_layer.weight[:5].tolist()}")
        print(f"  Last 5: {norm_layer.weight[-5:].tolist()}")
        post_attn_norm = norm_layer(residual_1.float()).to(dtype=embeddings.dtype)
        norm_layer.weight.data = norm_layer.weight.data.to(original_dtype)
        hf_tensors['post_attn_norm'] = post_attn_norm[0, -1, :].cpu().numpy()

        ffn_input = post_attn_norm.float()
        mlp = model.model.layers[0].mlp

        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data.float()
        mlp.up_proj.weight.data = mlp.up_proj.weight.data.float()
        mlp.down_proj.weight.data = mlp.down_proj.weight.data.float()

        gate = mlp.gate_proj(ffn_input)
        up = mlp.up_proj(ffn_input)
        gate_activated = torch.nn.functional.silu(gate)
        ffn_out = mlp.down_proj(gate_activated * up)

        hf_tensors['ffn_gate'] = gate_activated.to(embeddings.dtype)[0, -1, :].cpu().numpy()
        hf_tensors['ffn_up'] = up.to(embeddings.dtype)[0, -1, :].cpu().numpy()
        hf_tensors['ffn_output'] = ffn_out.to(embeddings.dtype)[0, -1, :].cpu().numpy()
        hf_tensors['ffn_down'] = hf_tensors['ffn_output']

        hf_tensors['layer_0_output'] = outputs.hidden_states[1][0, -1, :].cpu().numpy()

        final_norm_input = outputs.hidden_states[args.n_layers][0, -1, :]
        final_norm_out = model.model.norm(final_norm_input)
        hf_tensors['final_norm'] = final_norm_out.cpu().numpy()

        logits = outputs.logits[0, -1, :]
        hf_tensors['logits'] = logits.cpu().numpy()

    np.savez(os.path.join(os.path.dirname(__file__), "hf_tensors.npz"), **hf_tensors)
    print("\n[INFO] HF tensors saved to hf_tensors.npz")

    print(f"HF Layer 0 Output: Mean={hf_tensors['layer_0_output'].mean():.4f}, Std={hf_tensors['layer_0_output'].std():.4f}")
    print(f"HF Final Norm Output: Mean={hf_tensors['final_norm'].mean():.4f}, Std={hf_tensors['final_norm'].std():.4f}")

    top_k = torch.topk(logits, 5)
    print("HF Initial Top 5 Logits:")
    for idx, score in zip(top_k.indices, top_k.values):
        print(f"  - Token {idx.item()} ('{tokenizer.decode([idx.item()])}'): {score.item():.4f}")

    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "streamer": streamer,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": False,
    }
    
    print(f"User Prompt: {prompt}")
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    first_token_time = None
    first_token_received = False
    generated_text = ""

    for text in streamer:
        current_time = time.time()
        if not first_token_received:
            first_token_time = current_time - start_time
            first_token_received = True
        print(text, end="", flush=True)
        generated_text += text

    thread.join()
    total_time = time.time() - start_time

    total_tokens = len(tokenizer.encode(generated_text))
    generated_tokens = total_tokens - input_tokens
    decode_speed = generated_tokens / (total_time - first_token_time) if generated_tokens > 0 else 0
    prefill_speed = input_tokens / first_token_time if first_token_time > 0 else 0

    print("\nPerformance Statistics:")
    print(f"Decode Speed: {decode_speed:.1f} tokens/sec")
    print(f"Time to First Token: {first_token_time:.1f} seconds")
    print(f"Prefill Speed: {prefill_speed:.1f} tokens/sec")
    print(f"Token Count: {total_tokens}, Elapsed: {total_time:.2f}s, {round(total_tokens / total_time)} tokens/s")

    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="Once upon a time")
    args_cli = parser.parse_args()

    hf_args = ModelArgs()
    generate_text(args_cli.prompt, hf_args)
