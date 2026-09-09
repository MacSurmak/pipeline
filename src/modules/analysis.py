"""
File: src/modules/analysis.py
Description: Runs Prime MM-GBSA, open-source rescoring (GNINA), and RMSD evaluation.
"""
import os
import csv
import pandas as pd
from pathlib import Path
from loguru import logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from core.logger import console

from core.config_parser import load_config
from core.state_manager import StateManager
from utils.cmd_runner import run_command
from utils.fs import get_frames_sorted
from utils.ui import print_summary_table
from utils.data_extractors import build_funnel_whitelist

def run(cwd: Path, is_native: bool, force: bool = False):
    config = load_config(cwd / "config.yaml")
    settings = config.get("analysis", {})
    funnel_settings = settings.get("funnel", {})
    hw_settings = settings.get("hardware", {})
    
    in_dir = cwd / ("04a_redocking" if is_native else "04b_screening")
    out_dir = cwd / ("05a_redock_analysis" if is_native else "05b_screen_analysis")
    
    # Setup filtered input directory to preserve original docking results
    funnel_inputs_dir = out_dir / "funnel_inputs"
    funnel_inputs_dir.mkdir(parents=True, exist_ok=True)
    whitelist_csv = out_dir / "funnel_whitelist.csv"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    poses_dir = in_dir / "poses" if (in_dir / "poses").exists() else in_dir
    pv_files = get_frames_sorted(poses_dir, "*_pv.mae*")
    
    if not pv_files:
        logger.critical(f"No Pose Viewer files found in {in_dir.name}.")
        return

    schrodinger_path = os.environ.get("SCHRODINGER")
    mem_enabled = config.get("membrane", {}).get("enabled", False)
    mmgbsa_summary = []

    framework_dir = Path(os.environ.get("SBDD_FRAMEWORK_DIR"))
    filter_worker = framework_dir / "src" / "schrodinger_workers" / "filter_pv.py"

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed}/{task.total} tasks)"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        # Execute Funnel Pre-Filter for library
        if not is_native and not StateManager.check_output_exists(whitelist_csv, force):
            funnel_task = progress.add_task("[cyan]Applying Screening Funnel...", total=1)
            k_frames = funnel_settings.get("top_frames_per_ligand", 3)
            cutoff = funnel_settings.get("library_percentile_cutoff", 0.50)
            
            # Combine benchmark ligands from reporting and explicit funnel overrides
            controls = config.get("reporting", {}).get("benchmark_ligands", []) + funnel_settings.get("always_include", [])
            
            if build_funnel_whitelist(in_dir, whitelist_csv, k_frames, cutoff, force_include=controls):
                progress.update(funnel_task, completed=1)
            else:
                logger.warning("Could not build funnel whitelist. Proceeding without filter.")

        # Calculate per-frame pose weights for smooth progress tracking
        frame_weights = {}
        total_evals = len(pv_files)
        
        if not is_native and whitelist_csv.exists():
            df_wl = pd.read_csv(whitelist_csv)
            frame_weights = df_wl["Frame"].astype(str).value_counts().to_dict()
            total_evals = len(df_wl)

        unit_name = "poses" if not is_native else "frames"
        mmgbsa_task = progress.add_task(f"[yellow]Running Prime MM-GBSA...", total=total_evals)

        for pv_file, frame_id in pv_files:
            poses_in_batch = frame_weights.get(str(frame_id), 0) if not is_native else 1
            progress.update(mmgbsa_task, description=f"[yellow]Frame {frame_id}: Prime MM-GBSA ({poses_in_batch} {unit_name})...")
            mmgbsa_out = out_dir / f"mmgbsa_f{frame_id}-out.csv"
            
            if not StateManager.check_output_exists(mmgbsa_out, force):
                # Filter PV file before running
                active_pv_file = pv_file
                if not is_native and whitelist_csv.exists():
                    filtered_pv = funnel_inputs_dir / f"filtered_f{frame_id}_pv.maegz"
                    cmd_filter = [
                        f"{schrodinger_path}/run", "python3", str(filter_worker),
                        "--input", str(pv_file.resolve()),
                        "--output", str(filtered_pv.resolve()),
                        "--whitelist", str(whitelist_csv.resolve()),
                        "--frame", str(frame_id)
                    ]
                    run_command(cmd_filter, out_dir, f"filter_pv_f{frame_id}")
                    if filtered_pv.exists():
                        active_pv_file = filtered_pv

                host = hw_settings.get("host", "localhost")
                max_jobs = str(hw_settings.get("max_jobs", 8))

                cmd_mmgbsa = [
                    f"{schrodinger_path}/prime_mmgbsa",
                    "-job_type", settings.get("prime_job_type", "ENERGY"),
                    "-csv_output", "yes",
                    "-out_type", "PV",
                    "-jobname", f"mmgbsa_f{frame_id}",
                    "-HOST", f"{host}:{max_jobs}",
                    "-NJOBS", max_jobs,
                    "-WAIT"
                ]
                # Enable implicit membrane model if configured
                if mem_enabled:
                    cmd_mmgbsa.append("-membrane")

                cmd_mmgbsa.append(str(active_pv_file.resolve()))
                sub_task = progress.add_task(f"[yellow]Frame {frame_id}: Subjobs...", total=int(max_jobs))
                run_command(cmd_mmgbsa, out_dir, f"mmgbsa_f{frame_id}", progress=progress, task_id=sub_task)
                progress.remove_task(sub_task)
                
            if mmgbsa_out.exists():
                dg_scores = []
                with open(mmgbsa_out, "r") as f_csv:
                    for row in csv.DictReader(f_csv):
                        dg_keys = [k for k in row.keys() if k and "MMGBSA_dG_Bind" in k]
                        if dg_keys:
                            try: dg_scores.append(float(row[dg_keys[0]]))
                            except ValueError: pass
                
                if dg_scores:
                    top_dg, mean_dg = min(dg_scores), sum(dg_scores) / len(dg_scores)
                    logger.info(f"Frame {frame_id} MM-GBSA: Top ΔG_bind = {top_dg:.2f}, Mean = {mean_dg:.2f} kcal/mol")
                    mmgbsa_summary.append({"Frame": frame_id, "Top ΔG_bind": f"{top_dg:.2f}", "Mean ΔG_bind": f"{mean_dg:.2f}"})

            progress.advance(mmgbsa_task, advance=poses_in_batch)

    logger.success("Analysis module complete.")
    if mmgbsa_summary:
        print_summary_table("Prime MM-GBSA Summary", ["Frame", "Top ΔG_bind", "Mean ΔG_bind"], mmgbsa_summary)