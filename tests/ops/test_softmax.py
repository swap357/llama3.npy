import pytest
import numpy as np
import torch
import sys
import os

# Add repo root to path to import from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Assuming llama3.softmax can be imported directly
from llama3 import softmax as llama3_softmax

def test_softmax():
    # Generate some random data
    # shape: (batch_size, sequence_length, vocab_size-like_dim)
    data_np = np.random.rand(2, 10, 50).astype(np.float32)
    data_torch = torch.tensor(data_np)

    # llama3.py softmax
    softmax_llama3_val = llama3_softmax(data_np)

    # PyTorch softmax
    softmax_torch_val = torch.nn.functional.softmax(data_torch, dim=-1).cpu().numpy()

    print("Llama3 Softmax shape:", softmax_llama3_val.shape)
    print("PyTorch Softmax shape:", softmax_torch_val.shape)
    assert softmax_llama3_val.shape == softmax_torch_val.shape, "Softmax output shapes do not match"

    # Compare results
    assert np.allclose(softmax_llama3_val, softmax_torch_val, atol=1e-6), \
        "Softmax implementations do not match"

if __name__ == "__main__":
    pytest.main([__file__])