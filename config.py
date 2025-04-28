from dataclasses import dataclass
import os
import numpy as np

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
HF_MODEL_PATH = os.path.join(BASE_DIR, "llama-3.2-1B")
HF_TOKENIZER_PATH = os.path.join(BASE_DIR, "llama-3.2-1B")

NP_MODEL_PATH = os.path.join(BASE_DIR, "model.npz")
NP_TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.model.np")

# Data type configuration
USE_FLOAT32 = True   # Use float32 as default

if USE_FLOAT32:
    NP_DTYPE = np.float32

@dataclass
class ModelArgs:
    def __init__(self):
        # Model architecture parameters (from config.json)
        self.dim = 2048              # Embedding dimension
        self.n_layers = 16           # Number of layers
        self.n_heads = 32            # Number of attention heads
        self.n_kv_heads = 8          # Number of key/value heads (GQA)
        self.head_dim = 64           # Head dimension
        self.vocab_size = 128256     # Vocabulary size
        self.norm_eps = 1e-5         # Normalization epsilon for RMSNorm (matching PyTorch's value)
        # Runtime parameters
        self.max_batch_size = 1      # Batch size for inference
        self.max_seq_len = 256       # Maximum sequence length
        self.max_new_tokens = 10    # Maximum new tokens to generate

        # Generation parameters (deterministic mode)
        self.do_sample = False       # Using greedy decoding
        self.num_beams = 1           # Single beam for greedy decoding
        self.seed = 42               # Random seed for reproducibility
        self.temperature = 1.0       # Temperature for sampling (not used in deterministic mode)
        self.top_k = 1               # Top-k sampling (not used in deterministic mode)
        self.top_p = 1.0             # Top-p sampling (not used in deterministic mode)
        self.repetition_penalty = 1.0 # Repetition penalty (not used in deterministic mode)

        # Special tokens (matching HuggingFace tokenizer)
        self.bos_token_id = 128000   # Beginning of sequence token ID
        self.eos_token_id = 128001   # End of sequence token ID
        self.pad_token_id = -1       # Padding token ID

        # Data type
        self.dtype = NP_DTYPE        # For numpy implementation        self.torch_dtype = TORCH_DTYPE  # For PyTorch implementation
