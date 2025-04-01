import sys
import time
import os
from typing import Optional
import numpy as np
import warnings
import logging

# Suppress specific transformers warnings about do_sample=False
warnings.filterwarnings("ignore", category=UserWarning, message=".*`do_sample` is set to `False`.*")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, LlamaForCausalLM
from threading import Thread
from config import ModelArgs, HF_MODEL_PATH, HF_TOKENIZER_PATH

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DTYPE = torch.float32

def get_model_size_mb(model_path: str) -> float:
    """Get model file size in MB"""
    return os.path.getsize(model_path) / (1024 * 1024)

def generate_text(prompt: str, args: Optional[ModelArgs] = None) -> str:
    """Generate text using a LLaMA model with HuggingFace transformers."""
    if args is None:
        args = ModelArgs()

    # Set seeds for deterministic generation
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    # Initialize tokenizer and model
    logger.info(f"Loading tokenizer from {HF_TOKENIZER_PATH}")
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_PATH, trust_remote_code=True)
    logger.info(f"Tokenizer initialized with vocab size: {tokenizer.vocab_size}")
    logger.info(f"BOS ID: {tokenizer.bos_token_id}, EOS ID: {tokenizer.eos_token_id}")
    # Optional: Add check for space token if needed, like in check_hf_tokens.py
    
    # Load model with appropriate dtype and device map
    logger.info(f"Loading model from {HF_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_PATH, 
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True  # Helps with memory efficiency
    )
    load_time = time.time() - load_start
    print(f"HF Model Class: {type(model)}") # Print the loaded model class
    print(f"HF Model Config RMS Norm Epsilon: {model.config.rms_norm_eps}") # Print the epsilon value

    # Log model weights statistics
    logger.info("==== HuggingFace Model Weight Statistics ====")
    for name, tensor in model.state_dict().items():
        try:
            # Ensure tensor is on CPU and float32 for stats calculation
            tensor_cpu_f32 = tensor.float().cpu().numpy()
            if np.isnan(tensor_cpu_f32).any():
                 logger.warning(f"Weight {name}: Contains NaN values")
                 tensor_min, tensor_max, tensor_mean, tensor_std = 0.0, 0.0, 0.0, 0.0
            else:
                tensor_min = tensor_cpu_f32.min()
                tensor_max = tensor_cpu_f32.max()
                tensor_mean = tensor_cpu_f32.mean()
                tensor_std = tensor_cpu_f32.std()

            logger.info(
                f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                f"min={tensor_min:.4f}, max={tensor_max:.4f}, "
                f"mean={tensor_mean:.4f}, std={tensor_std:.4f}"
            )
        except Exception as e:
            logger.error(f"Could not process weight {name}: {e}")
    logger.info("==========================================")

    # Set up streamer for token-by-token output
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    
    # Print prompt
    print("\nUser: " + prompt)
    print("\nllama3.2-1B: ", end="", flush=True)
    
    start_time = time.time()
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = len(inputs.input_ids[0])
    print(f"HF Input IDs: {inputs.input_ids.tolist()}") # Print input IDs

    # === Get Initial Logits and Hidden States ===
    hf_tensors = {} # Dictionary to store tensors
    with torch.no_grad():
        # --- Get Embeddings First --- 
        embeddings = model.model.embed_tokens(inputs.input_ids)
        embeddings_last_token = embeddings[0, -1, :]
        hf_tensors['embeddings'] = embeddings_last_token.cpu().numpy()
        # print(f"HF Embeddings ...") # Keep prints for now, or remove if saving is enough
        
        # --- Manually apply first layer's input norm --- 
        first_norm_output_hf = model.model.layers[0].input_layernorm(embeddings)
        first_norm_output_hf_last_token = first_norm_output_hf[0, -1, :]
        hf_tensors['first_norm'] = first_norm_output_hf_last_token.cpu().numpy()
        # print(f"HF First Norm Output ...")
        
        # --- Manually apply first layer's attention block --- 
        # Need position_ids and attention_mask for the attention call
        # HF models typically generate these internally if not provided
        # Let's use the causal mask generated by the model internals (simplest)
        # We grab the attention output from the full model pass hidden states instead
        # This requires running the model once to get states, then extracting
        outputs_for_attn = model(**inputs, output_hidden_states=True)
        # The attention output is BEFORE the first residual connection, which isn't directly in hidden_states
        # We need to recompute it manually using the norm output
        # Get causal mask from the model forward pass
        # This is tricky as the mask isn't directly exposed easily. 
        # Alternative: Compute attention output manually step-by-step (more complex)
        # Easiest: Calculate the attention output by subtracting the original embedding from the first hidden state
        # first_hidden_state = embeddings + attention_output + ff_output? No, that's not right.
        # Let's rethink: Can we call the attention block directly? Yes.
        # We need the attention mask. For causal LM, it's created based on seq len.
        batch_size, seq_len = inputs.input_ids.shape
        # Create a causal mask (lower triangular) [bsz, 1, tgt_seq_len, src_seq_len]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=model.device)).view(1, 1, seq_len, seq_len)
        # The HF LlamaAttention module expects mask shape [bsz, 1, tgt_seq_len, src_seq_len]
        # where True indicates positions to attend to.
        # The module internally handles the conversion to additive mask (-inf)
        # Position IDs are usuallyarange(seq_len)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=model.device).unsqueeze(0)

        # --- Calculate RoPE --- 
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        base = model.config.rope_theta # Typically 10000.0 or higher for newer models
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=model.device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=model.device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Create the tuple (cos, sin) expected by position_embeddings
        freqs_cos = emb.cos()[None, :, :].to(dtype=embeddings.dtype)
        freqs_sin = emb.sin()[None, :, :].to(dtype=embeddings.dtype)
        position_embeddings = (freqs_cos, freqs_sin)
        # ----------------------

        # Call layer 0 attention block
        # The forward method of LlamaAttention returns (attn_output, attn_weights)
        attn_output_hf, _ = model.model.layers[0].self_attn(
            hidden_states=first_norm_output_hf, 
            attention_mask=causal_mask, 
            position_ids=position_ids,
            position_embeddings=position_embeddings
        )
        attn_output_hf_last_token = attn_output_hf[0, -1, :]
        hf_tensors['attn_output'] = attn_output_hf_last_token.cpu().numpy()
        # print(f"HF Attention Output ...")
        
        # --- Calculate state after first residual connection --- 
        residual_1_hf = embeddings + attn_output_hf
        residual_1_hf_last_token = residual_1_hf[0, -1, :]
        hf_tensors['residual_1'] = residual_1_hf_last_token.cpu().numpy()
        # print(f"HF Residual 1 Output ...")

        # --- Manually apply post-attention norm --- 
        # Force float32 calculation for RMSNorm comparison
        post_attn_layernorm = model.model.layers[0].post_attention_layernorm
        residual_1_hf_float32 = residual_1_hf.float() 
        original_weight_dtype = post_attn_layernorm.weight.dtype
        post_attn_layernorm.weight.data = post_attn_layernorm.weight.data.float()

        # --- Print HF Weight Summary ---
        hf_weight = post_attn_layernorm.weight.data
        print(f"\nHF PostAttnNorm Weight (L0): Mean={hf_weight.mean():.6f}, Std={hf_weight.std():.6f}, Shape={hf_weight.shape}")
        print(f"  First 5: {hf_weight[:5].tolist()}")
        print(f"  Last 5: {hf_weight[-5:].tolist()}")
        # -----------------------------

        post_attn_norm_output_hf = post_attn_layernorm(residual_1_hf_float32)
        post_attn_norm_output_hf = post_attn_norm_output_hf.to(embeddings.dtype) # Cast back to original dtype

        # Restore original weight dtype
        post_attn_layernorm.weight.data = post_attn_layernorm.weight.data.to(original_weight_dtype)

        post_attn_norm_output_hf_last_token = post_attn_norm_output_hf[0, -1, :]
        hf_tensors['post_attn_norm'] = post_attn_norm_output_hf_last_token.cpu().numpy()
        
        # --- Capture FFN states ---
        # Get FFN input (after post-attention norm)
        ffn_input_hf = post_attn_norm_output_hf
        hf_tensors['ffn_input'] = ffn_input_hf[0, -1, :].cpu().numpy()
        
        # Get gate and up projections - Force float32 for SiLU comparison
        ffn_input_hf_float32 = ffn_input_hf.float()
        
        # Temporarily cast MLP weights to float32 if needed (match NumPy)
        mlp = model.model.layers[0].mlp
        original_gate_dtype = mlp.gate_proj.weight.dtype
        original_up_dtype = mlp.up_proj.weight.dtype
        original_down_dtype = mlp.down_proj.weight.dtype
        
        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data.float()
        mlp.up_proj.weight.data = mlp.up_proj.weight.data.float()
        mlp.down_proj.weight.data = mlp.down_proj.weight.data.float()
        
        # Calculate in float32
        gate_hf = mlp.gate_proj(ffn_input_hf_float32)
        up_hf = mlp.up_proj(ffn_input_hf_float32)
        # Manually apply SiLU in float32 for comparison
        gate_hf_activated = torch.nn.functional.silu(gate_hf)
        ffn_intermediate_hf = gate_hf_activated * up_hf
        ffn_output_hf = mlp.down_proj(ffn_intermediate_hf)

        # Cast results back to original embedding dtype
        gate_hf = gate_hf.to(embeddings.dtype)
        up_hf = up_hf.to(embeddings.dtype)
        ffn_output_hf = ffn_output_hf.to(embeddings.dtype)

        # Restore original weight dtypes
        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data.to(original_gate_dtype)
        mlp.up_proj.weight.data = mlp.up_proj.weight.data.to(original_up_dtype)
        mlp.down_proj.weight.data = mlp.down_proj.weight.data.to(original_down_dtype)
        
        # Store potentially float32 results for analysis if needed, or casted results?
        # Let's store the final casted results to match what subsequent layers see
        hf_tensors['ffn_gate'] = gate_hf_activated.to(embeddings.dtype)[0, -1, :].cpu().numpy() # Store activated gate
        hf_tensors['ffn_up'] = up_hf[0, -1, :].cpu().numpy()
        hf_tensors['ffn_output'] = ffn_output_hf[0, -1, :].cpu().numpy()
        hf_tensors['ffn_down'] = ffn_output_hf[0, -1, :].cpu().numpy() # ffn_output IS ffn_down

        # --- Get Full Layer 0 Output --- 
        outputs = outputs_for_attn # Reuse the previous run
        first_layer_output_hf = outputs.hidden_states[1][0, -1, :] 
        hf_tensors['layer_0_output'] = first_layer_output_hf.cpu().numpy()
        # print(f"HF Layer 0 Output ...")

        # --- Get Final Norm Output --- 
        final_norm_input_hf = outputs.hidden_states[args.n_layers][0, -1, :]
        final_norm_output_hf = model.model.norm(final_norm_input_hf)
        hf_tensors['final_norm'] = final_norm_output_hf.cpu().numpy()
        # print(f"HF Final Norm Output ...")

        # --- Get Initial Logits --- 
        initial_logits = outputs.logits[0, -1, :]
        hf_tensors['logits'] = initial_logits.cpu().numpy()
        # print(f"HF Initial Top 5 Logits ...")

    # --- Save HF Tensors --- 
    hf_save_path = os.path.join(os.path.dirname(__file__), "hf_tensors.npz")
    np.savez(hf_save_path, **hf_tensors)
    print(f"\n[INFO] HF tensors saved to {hf_save_path}")
    # -----------------------
        
    # Print summaries (e.g., mean, std, first few values)
    print(f"HF Layer 0 Output (Last Token): Mean={first_layer_output_hf.mean():.4f}, Std={first_layer_output_hf.std():.4f}, First 5={first_layer_output_hf[:5].tolist()}")
    print(f"HF Final Norm Output (Last Token): Mean={final_norm_output_hf.mean():.4f}, Std={final_norm_output_hf.std():.4f}, First 5={final_norm_output_hf[:5].tolist()}")

    # Print top 5 logits and their corresponding tokens
    top_k_logits, top_k_indices = torch.topk(initial_logits, 5)
    print(f"HF Initial Top 5 Logits:")
    for i in range(5):
        token_id = top_k_indices[i].item()
        token_str = tokenizer.decode([token_id])
        print(f"  - Token {token_id} ('{token_str}'): {top_k_logits[i].item():.4f}")
    # ==========================

    # Set up generation parameters for strict greedy decoding
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "streamer": streamer,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": False,  # Explicitly set greedy decoding
    }

    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Track time to first token
    first_token_time = None
    first_token_received = False
    
    # Print tokens as they're generated
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
    
    # Calculate metrics
    total_tokens = len(tokenizer.encode(generated_text))
    generated_tokens = total_tokens - input_tokens
    
    # Decode speed (tokens/sec) for the generation phase
    decode_speed = generated_tokens / (total_time - first_token_time) if generated_tokens > 0 else 0
    
    # Prefill speed (tokens/sec) for the initial processing
    prefill_speed = input_tokens / first_token_time if first_token_time > 0 else 0
    
    # Get model and memory sizes
    model_size = get_model_size_mb(HF_MODEL_PATH)

    # Print statistics as text
    print("\nPerformance Statistics:")
    print(f"Decode Speed: {decode_speed:.1f} tokens/sec")
    print(f"Time to First Token: {first_token_time:.1f} seconds")
    print(f"Prefill Speed: {prefill_speed:.1f} tokens/sec")
    print(f"Model Size: {model_size:.0f} MB")
    print(f"Token Count: {total_tokens}, Elapsed: {total_time:.2f}s, {round(total_tokens / total_time)} tokens/s")

    return generated_text

if __name__ == "__main__":
    generate_text(sys.argv[1] if len(sys.argv) > 1 else "Once upon a time") 