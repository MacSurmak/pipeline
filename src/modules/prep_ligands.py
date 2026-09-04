"""
Ligand preparation module. DRY implementation for both native ligand and library.
"""
import os
from pathlib import Path
from loguru import logger

from core.config_parser import load_config
from core.state_manager import StateManager
from utils.cmd_runner import run_command
from utils.count_structures import get_structure_count
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from core.logger import console

def run(cwd: Path, is_native: bool, force: bool = False):
    config = load_config(cwd / "config.yaml")

    out_dir = cwd / "02_prep_ligands"
    struct_dir = out_dir / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)
    
    if is_native:
        # For native, we use frame 1's extracted 3D reference as the 2D template generator
        input_file = cwd / "01_prep_receptor" / "structures" / "ref_native_f1.mae"
        final_output = struct_dir / "native_prepared.maegz"
        log_name = "01_ligprep_native"
    else:
        lig_config = config.get("ligand_library", {})
        input_file = cwd / "00_input" / lig_config.get("input_file", "library.sdf")
        final_output = struct_dir / "library_prepared.maegz"
        log_name = "01_ligprep_library"
        
    StateManager.require_file(input_file, f"prep_lig (Native={is_native})")
    
    if StateManager.check_output_exists(final_output, force):
        return

    schrodinger_path = os.environ.get("SCHRODINGER")
    logger.info(f"Starting LigPrep for {'native ligand' if is_native else 'library'}.")
    
    ligprep_bin = f"{schrodinger_path}/ligprep"
    lp_settings = config.get("ligand_library", {}).get("ligprep", {})
    
    ext = input_file.suffix.lower()
    in_flag = "-isd" if ext in [".sdf", ".sd"] else "-ismi" if ext in [".smi", ".csv"] else "-imae"
    
    cmd = [
        ligprep_bin,
        in_flag, str(input_file),
        "-omae", str(final_output),
        "-epik",
        "-ph", str(lp_settings.get("target_ph", 7.4)),
        "-pht", str(lp_settings.get("ph_tolerance", 2.0)),
        "-s", str(lp_settings.get("max_states", 32)),
        "-bff", str(lp_settings.get("force_field", 14)),
        "-g", # Preserve native stereoisomer
        "-WAIT"
    ]

    total_input = get_structure_count(input_file)
    logger.info(f"Input library contains {total_input} molecule(s).")

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[magenta]LigPrep: processing {total_input} input molecules (Epik pH {lp_settings.get('target_ph', 7.4)})..."),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        progress.add_task("ligprep", total=None)  # total=None делает спиннер бесконечно крутящимся
        
        success = run_command(cmd, out_dir, log_name)

    if final_output.exists():
        export_dir = out_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)

        final_sdf = export_dir / final_output.name.replace(".maegz", ".sdf").replace(".mae", ".sdf")
        run_command([f"{schrodinger_path}/run", "structconvert.py", str(final_output), str(final_sdf)], out_dir, f"conv_sdf_{log_name}")

    if success:
        total_out = get_structure_count(final_output)
        logger.success(f"Ligand preparation complete: {total_input} inputs -> {total_out} states generated.")