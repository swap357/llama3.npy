# llama3.npy

This repository contains numpy implementation of the llama 3 model architecture.

## Outputs

- [NumPy Output Log](./outputs/np_20250331_235520.log)
- [Hugging Face Output Log](./outputs/hf_20250331_235335.log)

The following plots compare the internal activations between this NumPy implementation (`llama3.py`) and a reference HuggingFace implementation (`generate_hf.py`) at various stages within the model, using the same input and weights. 
Note that the NumPy implementation may not perfectly match the reference, and these plots visually highlight potential differences in the activations. [WIP]

## Plots

Figure 0: Llama3 arch

<p align="center">
  <img src="./plots/llama3_forward_pass.png" alt="Llama3 model">
</p>

Figure 1: Embeddings Comparison
![Embeddings Comparison](./plots/embeddings_comparison.png)

Figure 2: First Norm Comparison
![First Norm Comparison](./plots/first_norm_comparison.png)

Figure 3: Attention Output Comparison
![Attention Output Comparison](./plots/attn_output_comparison.png)

Figure 4: Residual 1 Comparison
![Residual 1 Comparison](./plots/residual_1_comparison.png)

Figure 5: Post Attention Norm Comparison
![Post Attention Norm Comparison](./plots/post_attn_norm_comparison.png)

Figure 6: FFN Input Comparison
![FFN Input Comparison](./plots/ffn_input_comparison.png)

Figure 7: FFN Gate Comparison
![FFN Gate Comparison](./plots/ffn_gate_comparison.png)

Figure 8: FFN Up Comparison
![FFN Up Comparison](./plots/ffn_up_comparison.png)

Figure 9: FFN Down Comparison
![FFN Down Comparison](./plots/ffn_down_comparison.png)

Figure 10: FFN Output Comparison
![FFN Output Comparison](./plots/ffn_output_comparison.png)

Figure 11: Layer 0 Output Comparison
![Layer 0 Output Comparison](./plots/layer_0_output_comparison.png)

Figure 12: Final Norm Comparison
![Final Norm Comparison](./plots/final_norm_comparison.png)

Figure 13: Logits Comparison
![Logits Comparison](./plots/logits_comparison.png)
