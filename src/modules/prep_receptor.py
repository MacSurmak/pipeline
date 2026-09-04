"""
Receptor preparation module. Orchestrates structure splitting and PrepWizard over ensembles.
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
    config = load_config(cwd / "config.yaml")
    rec_config = config.get("receptor", {})
    
    input_pdb = cwd / "00_input" / rec_config.get("input_file", "complex.pdb")
    out_dir = cwd / "01_prep_receptor"
    StateManager.require_file(input_pdb, "prep_rec")

    schrodinger_path = os.environ.get("SCHRODINGER")
    framework_dir = Path(os.environ.get("SBDD_FRAMEWORK_DIR"))
    
    # 1. Split Receptor
    logger.info("Step 1: Splitting complex and retaining cofactors (Ensemble support)")
    worker_script = framework_dir / "src" / "schrodinger_workers" / "split_receptor.py"
    
    raw_dir = out_dir / "raw"
    struct_dir = out_dir / "structures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)

    split_cmd = [
        f"{schrodinger_path}/run", "python3", str(worker_script),
        "--input", str(input_pdb),
        "--outdir", str(raw_dir),
        "--ligand", rec_config.get("native_ligand_resname", "LIG")
    ]
    
    cofactors = rec_config.get("keep_cofactors", [])
    if cofactors:
        split_cmd.extend(["--cofactors"] + cofactors)
        
    if not run_command(split_cmd, out_dir, "01_split_receptor"):
        return

    for ref_file in raw_dir.glob("ref_native_f*.mae"):
        shutil.move(str(ref_file), str(struct_dir / ref_file.name))

    # 2. PrepWizard
    logger.info("Step 2: Running Protein Preparation Wizard on all frames")
    prepwizard_bin = f"{schrodinger_path}/utilities/prepwizard"
    pw_settings = rec_config.get("prep_wizard", {})
    
    # raw_receptors = list(out_dir.glob("receptor_raw_f*.mae"))
    raw_receptors = sorted(raw_dir.glob("receptor_raw_f*.mae"), key=lambda p: int(re.search(r'_f(\d+)', p.stem).group(1)))

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed}/{task.total} frames)"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        prep_task = progress.add_task("[magenta]PrepWizard...", total=len(raw_receptors))
        for raw_rec in raw_receptors:
            frame_id = raw_rec.stem.split("_f")[-1]
            final_output = struct_dir / f"receptor_prepared_f{frame_id}.mae"
            
            if StateManager.check_output_exists(final_output, force):
                progress.advance(prep_task)
                continue
                
            logger.info(f"Preparing frame {frame_id}...")
            prep_cmd = [
                prepwizard_bin,
                "-fix", "-fillsidechains",
                "-propka_pH", str(pw_settings.get("target_ph", 7.4)),
                "-epik_pH", str(pw_settings.get("target_ph", 7.4)),
                "-epik_pHt", str(pw_settings.get("ph_tolerance", 1.0)),
                "-rmsd", str(pw_settings.get("minimize_rmsd", 0.3)),
                "-preserve_st_titles", "-WAIT"
            ]

            if pw_settings.get("cap_termini", False):
                prep_cmd.append("-captermini")

            prep_cmd.extend([str(raw_rec), str(final_output)])
            
            progress.update(prep_task, description=f"[magenta]PrepWizard Frame {frame_id}...")
            run_command(prep_cmd, out_dir, f"02_prepwizard_f{frame_id}")

            export_dir = out_dir / "export"
            export_dir.mkdir(parents=True, exist_ok=True)

            # Convert protein to .pdb
            final_pdb = export_dir / f"receptor_prepared_f{frame_id}.pdb"
            run_command([f"{schrodinger_path}/run", "structconvert.py", str(final_output), str(final_pdb)], out_dir, f"conv_pdb_f{frame_id}")
            
            # Convert native ligand to .sdf
            ref_lig = struct_dir / f"ref_native_f{frame_id}.mae"
            if ref_lig.exists():
                ref_sdf = export_dir / f"ref_native_f{frame_id}.sdf"
                run_command([f"{schrodinger_path}/run", "structconvert.py", str(ref_lig), str(ref_sdf)], out_dir, f"conv_ref_sdf_f{frame_id}")

            progress.advance(prep_task)
    
    logger.success("Receptor preparation complete for all frames.")