"""
File: src/utils/visualizations.py
Description: Generation of consensus scatter plots and MPO heatmaps.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(df: pd.DataFrame, out_dir: Path, top_n: int):
    if df.empty: return
    sns.set_theme(style="whitegrid")
    
    # 1. Consensus Scatter Plot
    if all(c in df.columns for c in ["GlideScore", "MMGBSA", "Ultimate_Score"]):
        plt.figure(figsize=(10, 8))
        cnn_col = "Resc_CNNaffinity" if "Resc_CNNaffinity" in df.columns else "Dock_CNNaffinity" if "Dock_CNNaffinity" in df.columns else None
        sizes = df[cnn_col] * 10 if cnn_col else 50
        scatter = plt.scatter(
            x=df["GlideScore"], y=df["MMGBSA"], s=sizes, 
            c=df["Ultimate_Score"], cmap="viridis", alpha=0.8, edgecolor="k"
        )
        native_df = df[df["Is_Native"] == True]
        if not native_df.empty:
            plt.axvline(x=native_df["GlideScore"].iloc[0], color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=native_df["MMGBSA"].iloc[0], color='r', linestyle='--', alpha=0.5)
            plt.scatter(
                native_df["GlideScore"], native_df["MMGBSA"], 
                s=150, facecolors='none', edgecolors='r', linewidth=2, label="Native Baseline"
            )
            plt.legend()
            
        plt.colorbar(scatter, label="Ultimate Score")
        plt.xlabel("GlideScore (kcal/mol)")
        plt.ylabel("Prime MM-GBSA (kcal/mol)")
        plt.title("Thermodynamic Landscape & Consensus Score")
        plt.tight_layout()
        plt.savefig(out_dir / "04_consensus_scatter.png", dpi=300)
        plt.close()

    # 2. MPO Heatmap
    pr_cols = [c for c in ["PR_Glide", "PR_MMGBSA", "PR_CNN_Resc", "PR_GNINA_Dock"] if c in df.columns]
    if pr_cols:
        top_df = df.head(top_n).set_index("Title")[pr_cols]
        plt.figure(figsize=(8, max(4, int(len(top_df) * 0.3))))
        sns.heatmap(top_df, cmap="RdYlGn", vmin=0, vmax=1, annot=True, fmt=".2f", cbar_kws={'label': 'Percentile Rank'})
        plt.title(f"Multiparameter Optimization (Top {len(top_df)})")
        plt.tight_layout()
        plt.savefig(out_dir / "03_mpo_heatmap.png", dpi=300)
        plt.close()