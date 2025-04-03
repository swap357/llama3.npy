import os
from graphviz import Digraph

def create_llama3_forward_flow():
    dot = Digraph("Llama3ForwardPass", comment="Llama3 Forward Pass")
    dot.attr(rankdir="TB", size="14,20", fontname="Helvetica")

    # Styles
    styles = {
        "node": {"style": "rounded,filled", "fontname": "Arial", "fontsize": "9"},
        "input": {"fillcolor": "#AEDFF7"},
        "norm": {"fillcolor": "#FFF2AE"},
        "attention": {"fillcolor": "#F8BBD0"},
        "ffn": {"fillcolor": "#D1C4E9"},
        "output": {"fillcolor": "#B9F6CA"},
        "residual": {"style": "dashed", "color": "#FF5252"},
        "cluster": {"style": "rounded,filled", "bgcolor": "#ECEFF1", "fontsize": "12"}
    }

    # Input
    dot.node("input_ids", "Input IDs\n[batch, seq_len]", shape="box", **{**styles["node"], **styles["input"]})
    dot.node("embedding", "Embedding Lookup\ninput_ids @ embed_weight\n[batch, seq_len, dim]", shape="box", **styles["node"])

    dot.edge("input_ids", "embedding")

    # Transformer Block
    with dot.subgraph(name="cluster_transformer") as tf:
        tf.attr(label="Transformer Block", **styles["cluster"])

        tf.node("ln1", "RMSNorm(ln1)\nmean(x²), normalize, scale\n[batch, seq_len, dim]", shape="box", **{**styles["node"], **styles["norm"]})

        with tf.subgraph(name="cluster_attention") as attn:
            attn.attr(label="Attention", **styles["cluster"])
            attn.node("qkv_proj", "QKV Projections\nx @ (q,k,v weights)\n[batch, seq_len, 3*dim]", shape="box", **{**styles["node"], **styles["attention"]})
            attn.node("rotary_emb", "Rotary Embeddings\napply_rotary_emb(q,k)\n[batch, seq_len, heads, head_dim]", shape="box", **{**styles["node"], **styles["attention"]})
            attn.node("attention_scores", "Attention Scores\nsoftmax((q @ k.T) / sqrt(head_dim))\n[batch, heads, seq_len, seq_len]", shape="box", **{**styles["node"], **styles["attention"]})
            attn.node("attention_output", "Attention Output\nattn @ v\n[batch, seq_len, dim]", shape="box", **{**styles["node"], **styles["attention"]})

            attn.edge("qkv_proj", "rotary_emb")
            attn.edge("rotary_emb", "attention_scores")
            attn.edge("attention_scores", "attention_output")

        tf.node("attn_add", "Residual Add\nx + attn\n[batch, seq_len, dim]", shape="ellipse", style="filled", fontname="Arial", fontsize="9")

        tf.edge("embedding", "ln1")
        tf.edge("ln1", "qkv_proj")
        tf.edge("attention_output", "attn_add")
        tf.edge("embedding", "attn_add", **styles["residual"], label="residual")

        tf.node("ln2", "RMSNorm(ln2)\nmean(x²), normalize, scale\n[batch, seq_len, dim]", shape="box", **{**styles["node"], **styles["norm"]})

        with tf.subgraph(name="cluster_ffn") as ffn:
            ffn.attr(label="Feed Forward", **styles["cluster"])
            ffn.node("gate_proj", "Gate Projection\nx @ gate_weight\n[batch, seq_len, hidden_dim]", shape="box", **{**styles["node"], **styles["ffn"]})
            ffn.node("silu", "SiLU Activation\nsilu(gate_proj)\n[batch, seq_len, hidden_dim]", shape="box", **{**styles["node"], **styles["ffn"]})
            ffn.node("up_proj", "Up Projection\nx @ up_weight\n[batch, seq_len, hidden_dim]", shape="box", **{**styles["node"], **styles["ffn"]})
            ffn.node("mul", "Element-wise Multiply\nswish * up_proj\n[batch, seq_len, hidden_dim]", shape="box", **{**styles["node"], **styles["ffn"]})
            ffn.node("down_proj", "Down Projection\nx @ down_weight\n[batch, seq_len, dim]", shape="box", **{**styles["node"], **styles["ffn"]})

            ffn.edge("gate_proj", "silu")
            ffn.edge("silu", "mul")
            ffn.edge("up_proj", "mul")
            ffn.edge("mul", "down_proj")

        tf.node("ffn_add", "Residual Add\nx + ffn\n[batch, seq_len, dim]", shape="ellipse", style="filled", fontname="Arial", fontsize="9")

        tf.edge("attn_add", "ln2")
        tf.edge("ln2", "gate_proj")
        tf.edge("down_proj", "ffn_add")
        tf.edge("attn_add", "ffn_add", **styles["residual"], label="residual")

    dot.node("final_norm", "Final RMSNorm\n[batch, seq_len, dim]", shape="box", **{**styles["node"], **styles["norm"]})
    dot.node("lm_head", "LM Head Projection\nx @ embed_weight.T\n[batch, seq_len, vocab_size]", shape="box", **{**styles["node"], **styles["output"]})
    dot.node("logits", "Logits\n[batch, seq_len, vocab_size]", shape="box", **{**styles["node"], **styles["output"]})

    dot.edge("ffn_add", "final_norm")
    dot.edge("final_norm", "lm_head")
    dot.edge("lm_head", "logits")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    dot.render(os.path.join(output_dir, "llama3_forward_pass"), format="png", cleanup=True)

if __name__ == "__main__":
    create_llama3_forward_flow()
    print("Detailed Llama3 forward pass visualization created in the 'visualizations' directory.")
