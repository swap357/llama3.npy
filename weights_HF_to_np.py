import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import logging
import sys
import torch
# Add project root to Python path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_MODEL_PATH, NP_MODEL_PATH, NP_DTYPE

TORCH_DTYPE = torch.float32

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_export(model_path, output_path):
    """Convert HuggingFace model to NumPy format with careful handling of numerical values"""
    logger.info(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=TORCH_DTYPE,  # Use dtype from config
        device_map="auto"
    )
    model = model.eval()
    
    logger.info(f"Converting parameters to numpy arrays (target dtype: {NP_DTYPE})...")
    dct = {}
    stats = {}
    
    for key, val in tqdm(model.named_parameters(), desc="Processing parameters"):
        # Get original stats
        tensor_stats = {
            "shape": val.shape,
            "orig_dtype": val.dtype,
            "min": float(val.min()),
            "max": float(val.max()),
            "mean": float(val.mean()),
            "std": float(val.std())
        }
        
        # Convert to numpy with configured dtype
        np_val = val.detach().cpu().numpy().astype(NP_DTYPE)
        
        # Check for numerical issues
        if np.any(np.isnan(np_val)) or np.any(np.isinf(np_val)):
            logger.warning(f"Found NaN/Inf values in {key}")
            # Replace NaN/Inf with 0 to prevent propagation
            np_val = np.nan_to_num(np_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get converted stats
        converted_stats = {
            "conv_dtype": np_val.dtype,
            "conv_min": float(np_val.min()),
            "conv_max": float(np_val.max()),
            "conv_mean": float(np_val.mean()),
            "conv_std": float(np_val.std())
        }
        
        # Store parameter and stats
        dct[key] = np_val
        stats[key] = {**tensor_stats, **converted_stats}
        
        # Log conversion details
        logger.info(f"\nParameter: {key}")
        logger.info(f"Shape: {tensor_stats['shape']}")
        logger.info(f"Original: dtype={tensor_stats['orig_dtype']}, range=[{tensor_stats['min']:.4f}, {tensor_stats['max']:.4f}], mean={tensor_stats['mean']:.4f}, std={tensor_stats['std']:.4f}")
        logger.info(f"Converted: dtype={converted_stats['conv_dtype']}, range=[{converted_stats['conv_min']:.4f}, {converted_stats['conv_max']:.4f}], mean={converted_stats['conv_mean']:.4f}, std={converted_stats['conv_std']:.4f}")
    
    # Save model and stats
    logger.info(f"\nSaving model to {output_path}")
    np.savez_compressed(output_path, **dct)
    
    # Save stats to a separate file for analysis
    stats_path = output_path.replace('.npz', '_stats.npz')
    np.savez_compressed(stats_path, stats=stats)
    logger.info(f"Saved conversion statistics to {stats_path}")
    logger.info("Done!")

if __name__ == "__main__":
    if not os.path.exists(HF_MODEL_PATH):
        logger.error(f"Error: HuggingFace model not found at {HF_MODEL_PATH}")
        logger.error("Please download the model first using:")
        logger.error(f"huggingface-cli download {HF_MODEL_PATH}")
        sys.exit(1)
        
    load_and_export(HF_MODEL_PATH, NP_MODEL_PATH)