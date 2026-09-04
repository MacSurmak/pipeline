"""
File: src/modules/dock.py
Description: Glide ensemble docking. Docks ligands into all available receptor frames.
"""
import os
import re
import csv
import shutil
from pathlib import Path
from loguru import logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from core.logger import console
from core.config_parser import load_config
from core.state_manager import StateManager
from utils.cmd_runner import run_command
from utils.count_structures import get_structure_count

def run(cwd: Path, is_native: bool, force: bool = False):
    config = load_config(cwd / "config.yaml")
    
    # 1. Setup paths based on native/library mode
    in_ligands = cwd / "02_prep_ligands" / "structures" / ("native_prepared.maegz" if is_native else "library_prepared.maegz")
    grid_dir = cwd / "03_grid" / "grids"
    out_dir = cwd / ("04a_redocking" if is_native else "04b_screening")
    
    StateManager.require_file(in_ligands, "docking (ligands)")
    
    # grids = list(grid_dir.glob("grid_f*.zip"))
    grids = sorted(grid_dir.glob("grid_f*.zip"), key=lambda p: int(re.search(r'_f(\d+)', p.stem).group(1)))
    if not grids:
        logger.critical("No Glide grids found in 03_grid/. Run 'grid' module first.")
        return

    schrodinger_path = os.environ.get("SCHRODINGER")
    glide_bin = f"{schrodinger_path}/glide"
    framework_dir = Path(os.environ.get("SBDD_FRAMEWORK_DIR"))
    
    total_ligands = get_structure_count(in_ligands)
    logger.info(f"Detected {total_ligands} ligand state(s) in {in_ligands.name}")

    redock_summary = []

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
        export_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        poses_dir.mkdir(parents=True, exist_ok=True)
        inputs_dir.mkdir(parents=True, exist_ok=True)

        # 2. Iterate over all available grids (Ensemble Docking)
        for grid_zip in grids:

            frame_id = grid_zip.stem.split("_f")[-1]
            job_name = f"dock_f{frame_id}"
            expected_out = poses_dir / f"{job_name}_pv.maegz"
            
            if StateManager.check_output_exists(expected_out, force):
                progress.advance(task_ensemble)
                continue

            progress.reset(task_frame, description=f"[cyan]Frame {frame_id}", total=total_ligands)
                
            logger.info(f"Setting up docking for Frame {frame_id}")
            
            dock_in = out_dir / f"dock_f{frame_id}.in"
            shutil.copy(cwd / "docking.in", dock_in)
            
            # Append specific parameters to template
            with open(dock_in, "a") as f:
                f.write(f"\nGRIDFILE {grid_zip.resolve()}\n")
                f.write(f"LIGANDFILE {in_ligands.resolve()}\n")
                f.write(f"PRECISION {config.get('docking', {}).get('precision', 'SP')}\n")
                
            logger.info(f"Executing Glide docking for Frame {frame_id}...")
            cmd = [glide_bin, dock_in.name, "-OVERWRITE", "-WAIT"]
            
            run_command(cmd, out_dir, f"glide_dock_f{frame_id}", progress=progress, task_id=task_frame)

            raw_pv = out_dir / f"{job_name}_pv.maegz"
            if raw_pv.exists():
                shutil.move(str(raw_pv), str(expected_out))
            if dock_in.exists():
                shutil.move(str(dock_in), str(inputs_dir / dock_in.name))
                
            if expected_out.exists():
                sdf_out = export_dir / f"dock_f{frame_id}_poses.sdf"
                export_worker = framework_dir / "src" / "schrodinger_workers" / "pv_to_sdf.py"
                cmd_sdf = [
                    f"{schrodinger_path}/run", "python3", str(export_worker),
                    "--pv", str(expected_out.resolve()),
                    "--sdf", str(sdf_out.resolve())
                ]
                run_command(cmd_sdf, out_dir, f"export_sdf_f{frame_id}")
                logger.info(f"Frame {frame_id}: Exported sorted poses to {sdf_out.name}")

            for csv_file in [out_dir / f"dock_f{frame_id}.csv", out_dir / f"dock_f{frame_id}_skip.csv"]:
                if csv_file.exists():
                    shutil.move(str(csv_file), str(tables_dir / csv_file.name))

            # RMSD for redocking analysis
            if is_native and expected_out.exists():
                ref_ligand = cwd / "01_prep_receptor" / "structures" / f"ref_native_f{frame_id}.mae"
                rmsd_out = tables_dir / f"rmsd_f{frame_id}.csv"
                worker_script = framework_dir / "src" / "schrodinger_workers" / "run_rmsd.py"
                
                cmd_rmsd = [
                    f"{schrodinger_path}/run", "python3", str(worker_script),
                    "--reference", str(ref_ligand.resolve()),
                    "--poses", str(expected_out.resolve()),
                    "--output", str(rmsd_out.resolve())
                ]
                run_command(cmd_rmsd, out_dir, f"rmsd_f{frame_id}")
                
                if rmsd_out.exists():
                    with open(rmsd_out, "r") as f_rmsd:
                        rows = list(csv.DictReader(f_rmsd))
                    if rows:
                        best_pose = min(rows, key=lambda r: float(r["RMSD_Heavy"]))
                        redock_summary.append({
                            "frame": frame_id,
                            "rmsd": float(best_pose["RMSD_Heavy"]),
                            "gscore": float(best_pose.get("GlideScore", 0.0))
                        })

            progress.advance(task_ensemble)

    logger.success(f"Docking complete for {len(grids)} frames.")

    if is_native and redock_summary:
        console.print("\n[bold cyan]══════════════════════ REDOCKING VALIDATION SUMMARY ══════════════════════[/bold cyan]")
        for item in redock_summary:
            status = "[bold green][PASS][/bold green]" if item['rmsd'] < 2.00 else "[bold red][FAIL][/bold red]"
            console.print(f"  Frame {item['frame']:>2} │ RMSD: [bold]{item['rmsd']:>6.3f} Å[/bold] │ GlideScore: [bold]{item['gscore']:>7.2f}[/bold] │ {status}")
        console.print("[bold cyan]══════════════════════════════════════════════════════════════════════════[/bold cyan]\n")