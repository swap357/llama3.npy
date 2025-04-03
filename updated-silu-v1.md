# llama3.npy

```
--- a/llama3.py
+++ b/llama3.py
@@ -28,5 +28,6 @@
 
 def silu(x):
+    x = np.clip(x, -88.0, 88.0)
     return x * (1.0 / (1.0 + np.exp(-x)))
```     

Figure 1: Embeddings Comparison

![Embeddings Comparison](./plots/np-clip-plots/embeddings_comparison.png)

Figure 2: First Norm Comparison

![First Norm Comparison](./plots/np-clip-plots/first_norm_comparison.png)

Figure 3: Attention Output Comparison

![Attention Output Comparison](./plots/np-clip-plots/attn_output_comparison.png)

Figure 4: Residual 1 Comparison

![Residual 1 Comparison](./plots/np-clip-plots/residual_1_comparison.png)

Figure 5: Post Attention Norm Comparison

![Post Attention Norm Comparison](./plots/np-clip-plots/post_attn_norm_comparison.png)

Figure 6: FFN Input Comparison

![FFN Input Comparison](./plots/np-clip-plots/post_attn_norm_comparison.png)

Figure 7: FFN Gate Comparison

![FFN Gate Comparison](./plots/np-clip-plots/ffn_gate_comparison.png)

Figure 8: FFN Up Comparison

![FFN Up Comparison](./plots/np-clip-plots/ffn_up_comparison.png)

Figure 9: FFN Down Comparison

![FFN Down Comparison](./plots/np-clip-plots/ffn_down_comparison.png)

Figure 10: FFN Output Comparison

![FFN Output Comparison](./plots/np-clip-plots/ffn_output_comparison.png)

Figure 11: Layer 0 Output Comparison
![Layer 0 Output Comparison](./plots/np-clip-plots/layer_0_output_comparison.png)

Figure 12: Final Norm Comparison

![Final Norm Comparison](./plots/np-clip-plots/final_norm_comparison.png)

Figure 13: Logits Comparison

![Logits Comparison](./plots/np-clip-plots/logits_comparison.png)
