"""
Generates Glide grids for all prepared receptor frames using native ligand as center.
"""
import os
import re
import shutil
from pathlib import Path
from loguru import logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from core.logger import console
from core.config_parser import load_config
from core.state_manager import StateManager
from utils.cmd_runner import run_command

def run(cwd: Path, force: bool = False):
    prep_dir = cwd / "01_prep_receptor" / "structures"
    
    out_dir = cwd / "03_grid"
    grids_dir = out_dir / "grids"
    inputs_dir = out_dir / "inputs"
    grids_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    
    prepared_receptors = sorted(prep_dir.glob("receptor_prepared_f*.mae"), key=lambda p: int(re.search(r'_f(\d+)', p.stem).group(1)))

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
        for rec_path in prepared_receptors:
            frame_id = rec_path.stem.split("_f")[-1]
            grid_zip = grids_dir / f"grid_f{frame_id}.zip"
            
            if StateManager.check_output_exists(grid_zip, force):
                progress.advance(grid_task)
                continue
                
            native_lig = prep_dir / f"ref_native_f{frame_id}.mae"
            StateManager.require_file(native_lig, "grid_gen (native ligand 3D reference)")

            grid_in = out_dir / f"gridgen_f{frame_id}.in"
            shutil.copy(cwd / "gridgen.in", grid_in)

            with open(grid_in, "a") as f:
                f.write(f"\nGRIDFILE grid_f{frame_id}.zip\n")
                f.write(f"RECEP_FILE {rec_path.resolve()}\n")
                f.write(f"REF_LIGAND_FILE {native_lig.resolve()}\n")

            logger.info(f"Generating grid for frame {frame_id}...")
            cmd = [glide_bin, grid_in.name, "-WAIT"]
            
            run_command(cmd, out_dir, f"glide_grid_f{frame_id}")

            raw_grid = out_dir / f"grid_f{frame_id}.zip"
            if raw_grid.exists():
                shutil.move(str(raw_grid), str(grid_zip))
            if grid_in.exists():
                shutil.move(str(grid_in), str(inputs_dir / grid_in.name))
        
            progress.advance(grid_task)

    logger.success("Grid generation complete for all frames.")