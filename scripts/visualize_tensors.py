import numpy as np
import matplotlib.pyplot as plt
import os

def load_tensors(hf_path, np_path):
    """Load tensors from .npz files."""
    hf_tensors = np.load(hf_path)
    np_tensors = np.load(np_path)
    return hf_tensors, np_tensors

def plot_tensor_comparison(hf_tensor, np_tensor, title, save_path):
    """Plot comparison of two tensors using histograms and scatter plot."""
    plt.figure(figsize=(15, 5))
    
    # Histogram subplot
    plt.subplot(1, 3, 1)
    plt.hist(hf_tensor, bins=50, alpha=0.5, label='HF', density=True)
    plt.hist(np_tensor, bins=50, alpha=0.5, label='NumPy', density=True)
    plt.title(f'{title} - Distribution')
    plt.legend()
    
    # Scatter plot subplot
    plt.subplot(1, 3, 2)
    plt.scatter(hf_tensor, np_tensor, alpha=0.1)
    plt.plot([hf_tensor.min(), hf_tensor.max()], [hf_tensor.min(), hf_tensor.max()], 'r--', label='Identity')
    plt.title(f'{title} - Scatter Plot')
    plt.legend()
    
    # Heatmap of differences
    plt.subplot(1, 3, 3)
    differences = np.abs(hf_tensor - np_tensor)
    plt.hist(differences, bins=50, density=True, color='red', alpha=0.7)
    plt.title(f'{title} - Absolute Differences')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tensor_differences(hf_tensor, np_tensor, title, save_path):
    """Plot absolute differences between tensors with more detailed analysis."""
    plt.figure(figsize=(15, 10))
    
    # Original differences histogram
    plt.subplot(2, 2, 1)
    differences = np.abs(hf_tensor - np_tensor)
    plt.hist(differences, bins=50, density=True, color='red', alpha=0.7)
    plt.title(f'{title} - Absolute Differences')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Density')
    
    # Log-scale differences
    plt.subplot(2, 2, 2)
    plt.hist(np.log1p(differences), bins=50, density=True, color='blue', alpha=0.7)
    plt.title(f'{title} - Log-Scale Differences')
    plt.xlabel('Log(1 + Absolute Difference)')
    plt.ylabel('Density')
    
    # Relative differences
    plt.subplot(2, 2, 3)
    relative_diffs = np.abs(hf_tensor - np_tensor) / (np.abs(hf_tensor) + 1e-10)
    plt.hist(relative_diffs, bins=50, density=True, color='green', alpha=0.7)
    plt.title(f'{title} - Relative Differences')
    plt.xlabel('Relative Difference')
    plt.ylabel('Density')
    
    # Correlation plot
    plt.subplot(2, 2, 4)
    plt.scatter(hf_tensor, np_tensor, alpha=0.1)
    plt.plot([hf_tensor.min(), hf_tensor.max()], [hf_tensor.min(), hf_tensor.max()], 'r--', label='Identity')
    plt.title(f'{title} - Correlation Plot')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_tensor_differences(hf_tensor, np_tensor, title):
    """Print detailed analysis of tensor differences."""
    differences = np.abs(hf_tensor - np_tensor)
    relative_diffs = differences / (np.abs(hf_tensor) + 1e-10)
    
    print(f"\n{title} Analysis:")
    print(f"  Max absolute difference: {np.max(differences):.6f}")
    print(f"  Mean absolute difference: {np.mean(differences):.6f}")
    print(f"  Median absolute difference: {np.median(differences):.6f}")
    print(f"  Std of absolute differences: {np.std(differences):.6f}")
    print(f"  Max relative difference: {np.max(relative_diffs):.6f}")
    print(f"  Mean relative difference: {np.mean(relative_diffs):.6f}")
    print(f"  Correlation coefficient: {np.corrcoef(hf_tensor, np_tensor)[0,1]:.6f}")
    
    # Find indices of largest differences
    top_5_indices = np.argsort(differences)[-5:]
    print("\n  Top 5 largest differences:")
    for idx in top_5_indices:
        print(f"    Index {idx}: HF={hf_tensor[idx]:.6f}, NumPy={np_tensor[idx]:.6f}, Diff={differences[idx]:.6f}")

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hf_path = os.path.join(current_dir, "hf_tensors.npz")
    np_path = os.path.join(current_dir, "np_tensors.npz")
    output_dir = os.path.join(current_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tensors
    hf_tensors, np_tensors = load_tensors(hf_path, np_path)
    
    # Plot comparisons for each tensor
    for key in hf_tensors.keys():
        hf_tensor = hf_tensors[key]
        np_tensor = np_tensors[key]
        
        # Plot comparison
        comparison_path = os.path.join(output_dir, f"{key}_comparison.png")
        plot_tensor_comparison(hf_tensor, np_tensor, key, comparison_path)
        
        # Plot detailed differences
        differences_path = os.path.join(output_dir, f"{key}_differences.png")
        plot_tensor_differences(hf_tensor, np_tensor, key, differences_path)
        
        # Print detailed analysis
        analyze_tensor_differences(hf_tensor, np_tensor, key)

if __name__ == "__main__":
    main() 