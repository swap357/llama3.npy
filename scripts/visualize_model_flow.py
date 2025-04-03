import os
from graphviz import Digraph

def create_llama3_forward_flow():
    dot = Digraph("Llama3ForwardPassSimplified", comment="Simplified Llama3 Forward Pass")
    dot.attr(rankdir="TB", size="12,16", fontname="Helvetica")

    # Styles
    styles = {
        "node": {"style": "rounded,filled", "fontname": "Arial", "fontsize": "10"},
        "input": {"fillcolor": "#AEDFF7"},
        "process": {"fillcolor": "#FFF2AE"},
        "output": {"fillcolor": "#B9F6CA"},
        "residual": {"style": "dashed", "color": "#FF5252"},
        "cluster": {"style": "rounded,filled", "bgcolor": "#ECEFF1", "fontsize": "12"}
    }

    # Input
    dot.node("input_ids", "Input Tokens\n(batch_size, seq_len)", shape="box", **{**styles["node"], **styles["input"]})
    dot.node("embedding", "Token Embedding\n(batch_size, seq_len, dim)", shape="box", **styles["node"])
    dot.edge("input_ids", "embedding")

    # Transformer Block Simplified
    with dot.subgraph(name="cluster_transformer") as tf:
        tf.attr(label="Transformer Block", **styles["cluster"])

        # Attention
        tf.node("attention", "Self-Attention\n(batch_size, seq_len, dim)", shape="box", **{**styles["node"], **styles["process"]})
        tf.node("attn_add", "Add Residual\n(batch_size, seq_len, dim)", shape="ellipse", style="filled", fontname="Arial", fontsize="10")

        tf.edge("embedding", "attention")
        tf.edge("attention", "attn_add")
        tf.edge("embedding", "attn_add", **styles["residual"], label="residual")

        # Feed Forward
        tf.node("ffn", "Feed-Forward Layer\n(batch_size, seq_len, dim)", shape="box", **{**styles["node"], **styles["process"]})
        tf.node("ffn_add", "Add Residual\n(batch_size, seq_len, dim)", shape="ellipse", style="filled", fontname="Arial", fontsize="10")

        tf.edge("attn_add", "ffn")
        tf.edge("ffn", "ffn_add")
        tf.edge("attn_add", "ffn_add", **styles["residual"], label="residual")

    # Output
    dot.node("lm_head", "Output Projection\n(batch_size, seq_len, vocab_size)", shape="box", **{**styles["node"], **styles["output"]})

    dot.edge("ffn_add", "lm_head")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    dot.render(os.path.join(output_dir, "llama3_forward_pass_simplified"), format="png", cleanup=True)

if __name__ == "__main__":
    create_llama3_forward_flow()
    print("Simplified Llama3 forward pass visualization created in the 'visualizations' directory.")
