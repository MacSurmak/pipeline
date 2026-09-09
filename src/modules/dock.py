"""
File: src/modules/dock.py
Description: Glide ensemble docking. Docks ligands into all available receptor frames.
"""
import os
from pathlib import Path
from loguru import logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from core.logger import console
from core.config_parser import load_config
from core.state_manager import StateManager
from utils.count_structures import get_structure_count
from utils.fs import get_frames_sorted
from utils.ui import print_summary_table
from utils.schrodinger import run_rmsd_worker
from engines import glide, gnina

def run(cwd: Path, is_native: bool, engine_name: str = "glide", force: bool = False):
    config = load_config(cwd / "config.yaml")
    
    if engine_name == "glide":
        in_ligands = cwd / "02_prep_ligands" / "structures" / ("native_prepared.maegz" if is_native else "library_prepared.maegz")
    else:
        in_ligands = cwd / "02_prep_ligands" / "export" / ("native_prepared.sdf" if is_native else "library_prepared.sdf")
        
    grid_dir = cwd / "03_grid" / "grids"
    out_dir = cwd / ("04a_redocking" if is_native else "04b_screening")
    
    StateManager.require_file(in_ligands, "docking (ligands)")
    
    grids = get_frames_sorted(grid_dir, "grid_f*.zip")
    if not grids:
        logger.critical("No Glide grids found in 03_grid/. Run 'grid' module first.")
        return

    engine_mod = glide if engine_name == "glide" else gnina
    total_ligands = get_structure_count(in_ligands)
    logger.info(f"Detected {total_ligands} ligand state(s) in {in_ligands.name}")

    redock_summary = []
    library_summary = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console  
    ) as progress:

        task_ensemble = progress.add_task("[green]Ensemble Docking", total=len(grids))
        task_frame = progress.add_task("[cyan]Current Frame", total=total_ligands)
        
        export_dir = out_dir / "export"
        tables_dir = out_dir / "tables"
        poses_dir = out_dir / "poses"
        inputs_dir = out_dir / "inputs"
        for d in [export_dir, tables_dir, poses_dir, inputs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for grid_zip, frame_id in grids:
            expected_out = poses_dir / (f"{engine_name}_dock_f{frame_id}_pv.maegz" if engine_name == "glide" else f"{engine_name}_dock_f{frame_id}.sdf")
            
            if StateManager.check_output_exists(expected_out, force):
                progress.advance(task_ensemble)
                continue

            progress.reset(task_frame, description=f"[cyan]Frame {frame_id}", total=total_ligands)
            logger.info(f"Setting up {engine_name.upper()} docking for Frame {frame_id}")
            
            # Delegate to engine
            expected_out, scores = engine_mod.run_docking(
                cwd, config, in_ligands, grid_zip, frame_id, out_dir, force, progress, task_frame
            )
            
            if expected_out and expected_out.exists():
                # RMSD Calculation (Native Redocking)
                if is_native:
                    ref_ligand = cwd / "01_prep_receptor" / "structures" / f"ref_native_f{frame_id}.mae"
                    rmsd_out = tables_dir / f"{engine_name}_rmsd_f{frame_id}.csv"
                    best_rmsd, best_score = run_rmsd_worker(ref_ligand, expected_out, rmsd_out, out_dir)
                    
                    if best_rmsd is not None:
                        status = "PASS" if best_rmsd < 2.0 else "FAIL"
                        logger.info(f"Frame {frame_id} {engine_name.upper()} Redock: Best RMSD={best_rmsd:.3f} Å, Score={best_score:.2f} ({status})")
                        redock_summary.append({"Frame": frame_id, "RMSD (Å)": f"{best_rmsd:.3f}", "Score": f"{best_score:.2f}", "Status": f"[bold {'green' if status=='PASS' else 'red'}]{status}[/]"})
                
                # Library Screening Summary
                elif scores:
                    top_score = max(scores) if engine_name == "gnina" else min(scores)
                    mean_score = sum(scores) / len(scores)
                    logger.info(f"Frame {frame_id} Screening: {len(scores)} poses. Top Score = {top_score:.2f}, Mean = {mean_score:.2f}")
                    library_summary.append({"Frame": frame_id, "Poses": str(len(scores)), "Top Score": f"{top_score:.2f}", "Mean Score": f"{mean_score:.2f}"})

            progress.advance(task_ensemble)

    logger.success(f"Docking complete for {len(grids)} frames.")

    if is_native and redock_summary:
        print_summary_table(f"Redocking Validation Summary ({engine_name.upper()})", ["Frame", "RMSD (Å)", "Score", "Status"], redock_summary)

    if not is_native and library_summary:
        print_summary_table(f"Library Screening Summary ({engine_name.upper()})", ["Frame", "Poses", "Top Score", "Mean Score"], library_summary)