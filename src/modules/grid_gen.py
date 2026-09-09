"""
Generates Glide grids for all prepared receptor frames using native ligand as center.
"""
import os
import shutil
from pathlib import Path
from loguru import logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from core.logger import console
from core.config_parser import load_config
from core.state_manager import StateManager
from utils.cmd_runner import run_command
from utils.fs import get_frames_sorted

def run(cwd: Path, force: bool = False):
    config = load_config(cwd / "config.yaml")
    prep_dir = cwd / "01_prep_receptor" / "structures"
    
    out_dir = cwd / "03_grid"
    grids_dir = out_dir / "grids"
    inputs_dir = out_dir / "inputs"
    grids_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    
    prepared_receptors = get_frames_sorted(prep_dir, "receptor_prepared_f*.mae")

    if not prepared_receptors:
        logger.critical("No prepared receptors found. Run 'prep_rec' first.")
        return

    schrodinger_path = os.environ.get("SCHRODINGER")
    glide_bin = f"{schrodinger_path}/glide"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total} frames)"),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        grid_task = progress.add_task("[yellow]Generating Grids...", total=len(prepared_receptors))
        for rec_path, frame_id in prepared_receptors:
            grid_zip = grids_dir / f"grid_f{frame_id}.zip"
            
            if StateManager.check_output_exists(grid_zip, force):
                progress.advance(grid_task)
                continue
                
            native_lig = prep_dir / f"ref_native_f{frame_id}.mae"
            StateManager.require_file(native_lig, "grid_gen (native ligand 3D reference)")

            grid_cfg = config.get("docking", {}).get("grid", {})
            inner = [int(round(float(x))) for x in grid_cfg.get("inner_box", [10, 10, 10])]
            outer = [int(round(float(x))) for x in grid_cfg.get("outer_box", [26, 26, 26])]
            ff = grid_cfg.get("forcefield", "OPLS_2005")

            grid_in = inputs_dir / f"gridgen_f{frame_id}.in"
            grid_content = (
                f"GRIDFILE {grid_zip.name}\n"
                f"RECEP_FILE {rec_path.resolve()}\n"
                f"REF_LIGAND_FILE {native_lig.resolve()}\n"
                f"INNERBOX {inner[0]}, {inner[1]}, {inner[2]}\n"
                f"OUTERBOX {outer[0]}, {outer[1]}, {outer[2]}\n"
                f"FORCEFIELD {ff}\n"
            )
            grid_in.write_text(grid_content)
            inner_box, outer_box = f"{inner[0]}, {inner[1]}, {inner[2]}", f"{outer[0]}, {outer[1]}, {outer[2]}"
            
            progress.update(grid_task, description=f"[yellow]Generating grid for frame {frame_id}...")
            cmd = [glide_bin, str(grid_in.resolve()), "-WAIT"]
            
            if not run_command(cmd, inputs_dir, f"glide_grid_f{frame_id}"):
                logger.critical(f"Grid generation failed for frame {frame_id}. Aborting.")
                return

            raw_grid = inputs_dir / f"grid_f{frame_id}.zip"
            if raw_grid.exists():
                shutil.move(str(raw_grid), str(grid_zip))
                logger.info(f"Frame {frame_id} Grid created successfully. InnerBox = [{inner_box}], OuterBox = [{outer_box}]")
        
            progress.advance(grid_task)

    logger.success("Grid generation complete for all frames.")