"""
File: src/modules/reporting.py
Description: Master orchestrator for Data Lake aggregation, Consensus Scoring, 
and export delegation (HTML, PyMOL, Plots).
"""
import sys
from pathlib import Path
from loguru import logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

import pandas as pd
import numpy as np

from core.logger import console
from core.config_parser import load_config
from core.state_manager import StateManager
from utils.fs import get_frames_sorted
from utils.ui import print_summary_table

# Import newly separated utility modules
from utils.data_extractors import parse_glide_csv, parse_prime_csv, parse_gnina_sdf, compute_pose_consensus_rmsd
from utils.visualizations import generate_plots
from utils.html_export import generate_html_report
from utils.pymol_export import generate_pymol_session

def build_data_lake(cwd: Path, is_native: bool, progress: Progress, main_task) -> pd.DataFrame:
    """Gathers metrics from CSVs and SDFs into a flat DataFrame."""
    dock_dir = cwd / ("04a_redocking" if is_native else "04b_screening")
    analysis_dir = cwd / ("05a_redock_analysis" if is_native else "05b_screen_analysis")
    grids = get_frames_sorted(cwd / "03_grid" / "grids", "grid_f*.zip")
    rows = []
    
    for _, frame_id in grids:
        progress.update(main_task, description=f"Extracting data from Frame {frame_id} (Native={is_native})...")
        
        glide_data = parse_glide_csv(dock_dir / "tables" / f"glide_dock_f{frame_id}.csv")
        prime_data = parse_prime_csv(analysis_dir / f"mmgbsa_f{frame_id}-out.csv")

        # 1. GNINA Rescore / Minimize (evaluated on Glide geometry)
        gnina_resc_sdf = None
        for pfx in ["gnina_minimize", "gnina_score_only"]:
            f = dock_dir / "export" / f"{pfx}_f{frame_id}_poses.sdf"
            if f.exists():
                gnina_resc_sdf = f
                break
        gnina_resc_data = parse_gnina_sdf(gnina_resc_sdf, prefix="Resc_") if gnina_resc_sdf else {}

        # 2. Independent GNINA Docking (evaluated on independent GNINA geometry)
        gnina_dock_sdf = dock_dir / "export" / f"gnina_dock_f{frame_id}_poses.sdf"
        gnina_dock_data = parse_gnina_sdf(gnina_dock_sdf, prefix="Dock_") if gnina_dock_sdf.exists() else {}

        all_titles = set(list(glide_data.keys()) + list(prime_data.keys()) + list(gnina_resc_data.keys()) + list(gnina_dock_data.keys()))
        for title in all_titles:
            row = {"Title": title, "Frame": f"F{frame_id}", "Is_Native": is_native}
            row.update(glide_data.get(title, {}))
            row.update(prime_data.get(title, {}))
            row.update(gnina_resc_data.get(title, {}))
            row.update(gnina_dock_data.get(title, {}))
            rows.append(row)
            
        progress.advance(main_task)
    return pd.DataFrame(rows)

def calculate_consensus_score(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Percentile Rank consensus score (0-100) using Glide-derived geometry."""
    if "GlideScore" in df.columns: 
        df["PR_Glide"] = df["GlideScore"].rank(ascending=False, pct=True).fillna(0.0)
    if "MMGBSA" in df.columns: 
        df["PR_MMGBSA"] = df["MMGBSA"].rank(ascending=False, pct=True).fillna(0.0)
    if "Resc_CNNaffinity" in df.columns: 
        df["PR_CNN_Resc"] = df["Resc_CNNaffinity"].rank(ascending=True, pct=True).fillna(0.0)
    if "Dock_CNNaffinity" in df.columns: 
        df["PR_GNINA_Dock"] = df["Dock_CNNaffinity"].rank(ascending=True, pct=True).fillna(0.0)
        
    pr_cols = [c for c in ["PR_Glide", "PR_MMGBSA", "PR_CNN_Resc"] if c in df.columns]
    if not pr_cols:
        df["Ultimate_Score"] = 0.0
        return df

    # Ultimate Score: Average percentile * 100
    df["Ultimate_Score"] = (df[pr_cols].sum(axis=1) / len(pr_cols)) * 100.0
    return df

def run(cwd: Path, force: bool = False):
    config = load_config(cwd / "config.yaml")
    settings = config.get("reporting", {})
    top_n_export = settings.get("top_n_export", 50)
    top_n_term = settings.get("top_n_terminal", 10)
    benchmark_titles = settings.get("benchmark_ligands", [])
    
    out_dir = cwd / "06_reporting"
    out_dir.mkdir(parents=True, exist_ok=True)
    hitlist_out = out_dir / "02_master_hitlist.csv"
    
    if StateManager.check_output_exists(hitlist_out, force):
        return

    logger.info("Initializing Data Lake and Conformational Selection Analysis...")
    grids = get_frames_sorted(cwd / "03_grid" / "grids", "grid_f*.zip")
    total_frames = len(grids) * 2

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
        TextColumn("({task.completed}/{task.total} tasks)"), TimeRemainingColumn(), console=console
    ) as progress:
        
        main_task = progress.add_task("[yellow]Building Data Lake...", total=total_frames)
        df_native = build_data_lake(cwd, is_native=True, progress=progress, main_task=main_task)
        df_library = build_data_lake(cwd, is_native=False, progress=progress, main_task=main_task)
        df = pd.concat([df_native, df_library], ignore_index=True)
        
        if df.empty:
            logger.critical("Data Lake is empty. Run docking and analysis first.")
            return
            
        progress.update(main_task, description="[cyan]Calculating Consensus Percentile Ranks...", total=total_frames+2, completed=total_frames)
        df = calculate_consensus_score(df)
        
        progress.update(main_task, description="[magenta]Resolving Conformational Selection...", completed=total_frames+2)
        
        df_resolved = df.sort_values("Ultimate_Score", ascending=False).drop_duplicates(subset="Title")
        df_resolved = compute_pose_consensus_rmsd(df_resolved, cwd)
        df_resolved["Is_Benchmark"] = df_resolved["Title"].isin(benchmark_titles) & (~df_resolved["Is_Native"])
        
        baseline_score = 0.0
        native_row = df_resolved[df_resolved["Is_Native"] == True]
        if not native_row.empty:
            baseline_score = native_row["Ultimate_Score"].iloc[0]
            df_resolved.insert(df_resolved.columns.get_loc("Ultimate_Score") + 1, "Δ_to_Native", df_resolved["Ultimate_Score"] - baseline_score)
            
        df.pivot(index="Title", columns="Frame", values="Ultimate_Score").to_csv(out_dir / "01_cross_frame_matrix.csv")
        df_resolved = df_resolved.sort_values("Ultimate_Score", ascending=False)
        df_resolved.to_csv(hitlist_out, index=False)
        
        progress.update(main_task, description="[green]Generating Reports & Exports...", completed=total_frames+3)
        
        native_df = df_resolved[df_resolved["Is_Native"] == True]
        bench_df = df_resolved[df_resolved["Is_Benchmark"] == True]
        library_df_full = df_resolved[(df_resolved["Is_Native"] == False) & (df_resolved["Is_Benchmark"] == False)].head(top_n_export)
        export_df = pd.concat([library_df_full, native_df, bench_df], ignore_index=True).drop_duplicates(subset="Title")
        export_df = export_df.sort_values("Ultimate_Score", ascending=False)

        generate_plots(df_resolved, out_dir, top_n=top_n_export)
        generate_html_report(export_df, cwd, out_dir)
        generate_pymol_session(export_df, cwd, out_dir)

    logger.success("Reporting module completed successfully.")
    
    summary_cols = ["Title", "Frame", "Ultimate_Score"]
    if "Δ_to_Native" in df_resolved.columns: summary_cols.append("Δ_to_Native")
    if "Pose_RMSD" in df_resolved.columns: summary_cols.append("Pose_RMSD")
    if "PR_Glide" in df_resolved.columns: summary_cols.append("PR_Glide")
    if "PR_MMGBSA" in df_resolved.columns: summary_cols.append("PR_MMGBSA")
    if "PR_CNN_Resc" in df_resolved.columns: summary_cols.append("PR_CNN_Resc")
    if "PR_GNINA_Dock" in df_resolved.columns: summary_cols.append("PR_GNINA_Dock")
    
    library_df_term = df_resolved[(df_resolved["Is_Native"] == False) & (df_resolved["Is_Benchmark"] == False)].head(top_n_term)
    display_df = pd.concat([native_df, bench_df, library_df_term], ignore_index=True).drop_duplicates(subset="Title")
    display_df = display_df.sort_values("Ultimate_Score", ascending=False)
    top_hits = display_df.to_dict("records")
    
    for hit in top_hits:
        if hit.get("Is_Native"): hit["Title"] = "[bold cyan]Native Ligand [REF][/bold cyan]"
        elif hit.get("Is_Benchmark"): hit["Title"] = f"[bold yellow]{hit['Title']} [CTRL][/bold yellow]"
        p_rmsd = hit.get("Pose_RMSD")
        if pd.notna(p_rmsd):
            hit["Pose_RMSD"] = f"[bold green]{p_rmsd:.2f} Å[/bold green]" if p_rmsd <= 2.0 else f"{p_rmsd:.2f} Å"
        else:
            hit["Pose_RMSD"] = "N/A"
            
        for col in ["Ultimate_Score", "Δ_to_Native", "PR_Glide", "PR_MMGBSA", "PR_CNN_Resc", "PR_GNINA_Dock"]:
            if col in hit and pd.notna(hit[col]): hit[col] = f"{hit[col]:.2f}"
            else: hit[col] = "N/A"

    print_summary_table("Conformational Selection Top Hits", summary_cols, top_hits)
    logger.info(f"Full reports, plots, and PyMOL session saved to: {out_dir.absolute()}")