import os
import json
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
from tabulate import tabulate  # Install with: pip install tabulate

def load_stats_file(file_path):
    """Load a JSON stats file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_shapes(hf_dir, llama_dir):
    """Compare shapes between HF and Llama3 stats files"""
    hf_dir = Path(hf_dir)
    llama_dir = Path(llama_dir)

    # Get all token files
    hf_files = list(hf_dir.glob('token_*_*.json'))
    llama_files = list(llama_dir.glob('token_*_*.json'))

    # Extract token indices
    hf_tokens = {int(f.stem.split('_')[1]) for f in hf_files}
    llama_tokens = {int(f.stem.split('_')[1]) for f in llama_files}

    # Find common tokens
    common_tokens = sorted(hf_tokens.intersection(llama_tokens))

    # Extract tensor names
    hf_tensor_names = {f.stem.split('_', 2)[2] for f in hf_files}
    llama_tensor_names = {f.stem.split('_', 2)[2] for f in llama_files}
    common_tensor_names = sorted(hf_tensor_names.intersection(llama_tensor_names))

    print(f"Found {len(common_tokens)} common tokens to compare")
    print(f"Found {len(common_tensor_names)} common tensor types\n")

    # Store shape difference details
    shape_differences = []

    # Compare shapes for each token and tensor type
    for token_idx in common_tokens:
        for tensor_name in common_tensor_names:
            hf_file = hf_dir / f"token_{token_idx}_{tensor_name}.json"
            llama_file = llama_dir / f"token_{token_idx}_{tensor_name}.json"

            if not hf_file.exists() or not llama_file.exists():
                continue

            try:
                hf_data = load_stats_file(hf_file)
                llama_data = load_stats_file(llama_file)

                # Compare shapes directly from JSON
                hf_shape = hf_data.get('shape', [])
                llama_shape = llama_data.get('shape', [])

                if hf_shape != llama_shape:
                    shape_differences.append({
                        'token': token_idx,
                        'tensor': tensor_name,
                        'hf_shape': hf_shape,
                        'llama_shape': llama_shape
                    })
            except Exception as e:
                print(f"Error comparing {tensor_name} at token {token_idx}: {e}")

    return shape_differences

def analyze_shape_differences(shape_differences):
    """Analyze and categorize shape differences"""
    if not shape_differences:
        print("No shape differences found! All tensors have matching shapes.")
        return

    # Group by tensor type
    tensor_groups = defaultdict(list)
    for diff in shape_differences:
        tensor_groups[diff['tensor']].append(diff)

    # Print summary
    print(f"Found {len(shape_differences)} shape differences across {len(tensor_groups)} tensor types\n")

    # Print details for each tensor type
    for tensor_name, diffs in tensor_groups.items():
        print(f"\n=== Shape differences for {tensor_name} ===")

        # Get a representative example
        example = diffs[0]
        hf_shape = example['hf_shape']
        llama_shape = example['llama_shape']

        # Compare dimensions
        if len(hf_shape) != len(llama_shape):
            print(f"* Dimension count mismatch: HF has {len(hf_shape)}, Llama has {len(llama_shape)}")
        else:
            for i, (hf_dim, llama_dim) in enumerate(zip(hf_shape, llama_shape)):
                if hf_dim != llama_dim:
                    dim_name = "Batch" if i == 0 else "Sequence Length" if i == 1 else f"Dimension {i}"
                    print(f"* {dim_name} mismatch: HF={hf_dim}, Llama={llama_dim}")

        # Create table of affected tokens
        table_data = []
        for diff in diffs:
            table_data.append([
                diff['token'],
                str(diff['hf_shape']),
                str(diff['llama_shape'])
            ])

        # Print table
        print("\nDetailed comparison by token:")
        headers = ["Token", "HF Shape", "Llama Shape"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser(description='Compare tensor shapes between HF and Llama3 implementations')
    parser.add_argument('--hf-dir', type=str, default='hf_stats', help='Directory containing HF statistics')
    parser.add_argument('--llama-dir', type=str, default='llama3_stats', help='Directory containing Llama3 statistics')
    parser.add_argument('--detailed', action='store_true', help='Show detailed shape comparisons for each token')
    args = parser.parse_args()

    # Compare shapes
    shape_differences = compare_shapes(args.hf_dir, args.llama_dir)

    # Analyze and present results
    analyze_shape_differences(shape_differences)

if __name__ == '__main__':
    main()