"""
File: src/modules/gnina_rescore.py
Description: Rescores or minimizes existing Glide poses using GNINA CNN.
"""
import shutil
from pathlib import Path
from loguru import logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from core.logger import console
from core.config_parser import load_config
from core.state_manager import StateManager
from utils.cmd_runner import run_command
from utils.fs import get_frames_sorted
from utils.ui import print_summary_table
from utils.data_extractors import filter_sdf_by_whitelist

def run(cwd: Path, is_native: bool, mode: str, force: bool = False):
    config = load_config(cwd / "config.yaml")
    gnina_settings = config.get("gnina_rescore", {})
    
    dock_dir = cwd / ("04a_redocking" if is_native else "04b_screening")
    analysis_dir = cwd / ("05a_redock_analysis" if is_native else "05b_screen_analysis")
    whitelist_csv = analysis_dir / "funnel_whitelist.csv"
    
    export_dir = dock_dir / "export"
    poses_dir = dock_dir / "poses"
    funnel_inputs_dir = dock_dir / "funnel_inputs"
    funnel_inputs_dir.mkdir(parents=True, exist_ok=True)
    
    glide_sdfs = get_frames_sorted(export_dir, "glide_dock_f*_poses.sdf")
    if not glide_sdfs:
        logger.critical(f"No Glide poses found in {export_dir.name}. Run 'pipeline dock --glide' first.")
        return

    logger.info(f"Starting GNINA {mode} for {len(glide_sdfs)} frame(s).")
    summary = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed}/{task.total} frames)"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task_gnina = progress.add_task(f"[magenta]GNINA {mode}...", total=len(glide_sdfs))

        for sdf_in, frame_id in glide_sdfs:
            progress.update(task_gnina, description=f"[magenta]Frame {frame_id}: GNINA {mode}...")
            
            rec_pdb = cwd / "01_prep_receptor" / "export" / f"receptor_prepared_f{frame_id}.pdb"
            ref_lig = cwd / "01_prep_receptor" / "export" / f"ref_native_f{frame_id}.sdf"
            
            StateManager.require_file(rec_pdb, f"gnina {mode} (receptor)")
            StateManager.require_file(ref_lig, f"gnina {mode} (autobox ligand)")
            
            job_name = f"gnina_{mode}_f{frame_id}"
            expected_out = poses_dir / f"{job_name}.sdf"
            
            if not StateManager.check_output_exists(expected_out, force):
                active_sdf_in = sdf_in
                # Apply funnel whitelist to input SDF if available
                if not is_native and whitelist_csv.exists():
                    filtered_sdf = funnel_inputs_dir / f"filtered_f{frame_id}_poses.sdf"
                    filter_sdf_by_whitelist(sdf_in, filtered_sdf, frame_id, whitelist_csv)
                    if filtered_sdf.exists():
                        active_sdf_in = filtered_sdf
                        
                cmd = [
                    "gnina", "-r", str(rec_pdb.resolve()), "-l", str(active_sdf_in.resolve()),
                    "--autobox_ligand", str(ref_lig.resolve()), "-o", str(expected_out.resolve()),
                    "--cnn", str(gnina_settings.get("cnn", "crossdock_default2018"))
                ]
                if mode == "score_only": cmd.append("--score_only")
                elif mode == "minimize": cmd.append("--minimize")
                    
                run_command(cmd, dock_dir, job_name)
            
            if expected_out.exists():
                sdf_out = export_dir / f"{job_name}_poses.sdf"
                shutil.copy(expected_out, sdf_out)
                
                affinities, scores = [], []
                with open(sdf_out, "r") as f_sdf:
                    lines = f_sdf.readlines()
                    for i, line in enumerate(lines):
                        if "Affinity>" in line or "affinity>" in line:
                            try: affinities.append(float(lines[i+1].strip()))
                            except ValueError: pass
                        elif "CNNscore>" in line:
                            try: scores.append(float(lines[i+1].strip()))
                            except ValueError: pass
                
                if affinities and scores:
                    top_aff, top_score = max(affinities), max(scores)
                    logger.info(f"Frame {frame_id} GNINA {mode}: Top CNN_affinity = {top_aff:.2f}, Top CNN_score = {top_score:.3f}")
                    summary.append({"Frame": frame_id, "Top CNN_affinity": f"{top_aff:.2f}", "Top CNN_score": f"{top_score:.3f}"})
            
            progress.advance(task_gnina)
            
    if summary:
        print_summary_table(f"GNINA Rescoring Summary ({mode})", ["Frame", "Top CNN_affinity", "Top CNN_score"], summary)