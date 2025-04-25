import numpy as np
import logging
from config import NP_DTYPE

logger = logging.getLogger(__name__)

def load_parameters(model_path):
    """Load model parameters and convert to the correct dtype."""
    logger.info(f"Loading model from {model_path}")
    
    # Load model weights
    weights = np.load(model_path)
    
    # Log weight statistics before conversion
    # for name, weight in weights.items():
    #     logger.info(f"{name}: shape={weight.shape}, dtype={weight.dtype}, "
    #                f"min={weight.min():.4f}, max={weight.max():.4f}, "
    #                f"mean={weight.mean():.4f}, std={weight.std():.4f}")
    
    # Convert weights to target dtype
    converted = {}
    for name, weight in weights.items():
        # Convert weight to target dtype
        weight = weight.astype(NP_DTYPE)
        
        # Check for NaN or Inf values
        if np.any(np.isnan(weight)) or np.any(np.isinf(weight)):
            logger.warning(f"Weight {name} contains NaN or Inf values")
            weight = np.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
        
        converted[name] = weight
        
        # Log weight statistics after conversion
        # logger.info(f"{name} (converted): dtype={weight.dtype}, "
        #            f"min={weight.min():.4f}, max={weight.max():.4f}, "
        #            f"mean={weight.mean():.4f}, std={weight.std():.4f}")
    
    return converted
