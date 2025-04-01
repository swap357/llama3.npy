from transformers import AutoTokenizer
import json
import os
import sys
from config import HF_TOKENIZER_PATH, NP_TOKENIZER_PATH

def convert_hf_tokenizer_to_np(model_path, output_path):
    """Convert HuggingFace tokenizer to NumPy format"""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Get special token IDs and mappings
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    special_tokens = tokenizer.all_special_tokens
    special_tokens_map = tokenizer.special_tokens_map
    
    # Extract vocabulary and merge rules
    vocab = tokenizer.get_vocab()
    merges = tokenizer.merges if hasattr(tokenizer, 'merges') else []
    
    # Create token list preserving original IDs
    max_id = max(vocab.values())
    tokens = [""] * (max_id + 1)  # Initialize with empty strings
    for token, idx in vocab.items():
        tokens[idx] = token
    
    # Create scores list (used for BPE merging priority)
    scores = []
    for i in range(len(tokens)):
        token = tokens[i]
        if not token:
            scores.append(0.0)
            continue
            
        # Higher scores for special tokens
        if token.startswith('Ġ'):  # This is the special space token
            scores.append(1000.0)  # Higher score for space-prefixed tokens
        else:
            scores.append(0.0)
    
    # Create model with all necessary information
    token_model = {
        "tokens": tokens,
        "scores": scores,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "special_tokens": special_tokens,
        "special_tokens_map": special_tokens_map,
        "merges": merges,
        "vocab_size": len(tokens),
        "model_type": "llama",
        "space_token": "Ġ"  # Add explicit marker for the space token
    }
    
    print(f"Saving tokenizer to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(token_model, f, ensure_ascii=False, indent=2)
    
    # Verify conversion
    print("\nVerifying conversion...")
    print(f"Original vocab size: {len(vocab)}")
    print(f"Converted vocab size: {len(tokens)}")
    print(f"BOS token: {tokens[bos_id]} (ID: {bos_id})")
    print(f"EOS token: {tokens[eos_id]} (ID: {eos_id})")
    
    # Check special space token
    space_tokens = [(i, token) for i, token in enumerate(tokens) if token == 'Ġ']
    if space_tokens:
        print(f"Space token 'Ġ' found at ID: {space_tokens[0][0]}")
    else:
        print("Space token 'Ġ' not found")
    
    # Check some space-prefixed tokens
    space_prefixed = [(i, token) for i, token in enumerate(tokens) if token and token.startswith('Ġ')]
    print(f"Found {len(space_prefixed)} tokens with 'Ġ' prefix")
    print(f"Sample space-prefixed tokens: {space_prefixed[:5]}")
    
    # Check a few common tokens
    test_tokens = ["the", "Ġthe", "and", "Ġand", "is", "Ġis"]
    print("\nChecking common tokens:")
    for token in test_tokens:
        orig_id = vocab.get(token, None)
        if orig_id is not None:
            conv_token = tokens[orig_id]
            print(f"Token: '{token}' - Original ID: {orig_id}, Converted token at ID: '{conv_token}'")
    
    print("\nDone!")
    return token_model

if __name__ == "__main__":
    if not os.path.exists(HF_TOKENIZER_PATH):
        print(f"Error: HuggingFace tokenizer not found at {HF_TOKENIZER_PATH}")
        print("Please download the model first using:")
        print(f"huggingface-cli download {HF_TOKENIZER_PATH}")
        sys.exit(1)
        
    convert_hf_tokenizer_to_np(HF_TOKENIZER_PATH, NP_TOKENIZER_PATH)