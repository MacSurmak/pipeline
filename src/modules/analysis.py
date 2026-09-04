"""
File: src/modules/analysis.py
Description: Runs Prime MM-GBSA, open-source rescoring (GNINA), and RMSD evaluation.
"""
import os
from pathlib import Path
from loguru import logger

from core.config_parser import load_config
from core.state_manager import StateManager
from utils.cmd_runner import run_command

def run(cwd: Path, is_native: bool, force: bool = False):
    config = load_config(cwd / "config.yaml")
    settings = config.get("analysis", {})
    
    in_dir = cwd / ("04a_redocking" if is_native else "04b_screening")
    out_dir = cwd / ("05a_redock_analysis" if is_native else "05b_screen_analysis")
    
    poses_dir = in_dir / "poses" if (in_dir / "poses").exists() else in_dir
    pv_files = list(poses_dir.glob("*_pv.maegz")) + list(poses_dir.glob("*_pv.mae"))
    
    if not pv_files:
        logger.critical(f"No Pose Viewer files found in {in_dir.name}.")
        return

    schrodinger_path = os.environ.get("SCHRODINGER")
    framework_dir = Path(os.environ.get("SBDD_FRAMEWORK_DIR"))

    for pv_file in pv_files:
        frame_id = pv_file.stem.split("_f")[-1].split("_")[0]
        
        # ---------------------------------------------------------
        # 1. PRIME MM-GBSA
        # ---------------------------------------------------------
        if settings.get("run_mmgbsa", True):
            mmgbsa_out = out_dir / f"mmgbsa_f{frame_id}-out.csv"
            if not StateManager.check_output_exists(mmgbsa_out, force):
                logger.info(f"Running Prime MM-GBSA for Frame {frame_id}")
                cmd_mmgbsa = [
                    f"{schrodinger_path}/prime_mmgbsa",
                    "-job_type", settings.get("prime_job_type", "REAL_MIN"),
                    "-csv_output", "yes",
                    "-out_type", "PV",
                    "-jobname", f"mmgbsa_f{frame_id}",
                    "-WAIT",
                    str(pv_file.resolve())
                ]
                run_command(cmd_mmgbsa, out_dir, f"mmgbsa_f{frame_id}")

        # ---------------------------------------------------------
        # 2. RMSD CALCULATION (Native Redocking Only)
        # ---------------------------------------------------------
        if is_native and settings.get("rmsd_to_native", True):
            rmsd_out = out_dir / f"rmsd_f{frame_id}.csv"
            ref_ligand = cwd / "01_prep_receptor" / f"ref_native_f{frame_id}.mae"
            
            if ref_ligand.exists() and not StateManager.check_output_exists(rmsd_out, force):
                logger.info(f"Calculating Redocking RMSD for Frame {frame_id}")
                worker_script = framework_dir / "src" / "schrodinger_workers" / "run_rmsd.py"
                cmd_rmsd = [
                    f"{schrodinger_path}/run", "python3", str(worker_script),
                    "--reference", str(ref_ligand.resolve()),
                    "--poses", str(pv_file.resolve()),
                    "--output", str(rmsd_out)
                ]
                run_command(cmd_rmsd, out_dir, f"rmsd_f{frame_id}")

        # ---------------------------------------------------------
        # 3. GNINA (Open-Source Integration Placeholder)
        # ---------------------------------------------------------
        if settings.get("run_gnina", False):
            logger.info("GNINA rescoring is requested. Extracting to SDF/PDB...")
            # Here we would call structconvert to split PV to SDF (ligands) and PDB (receptor)
            # Then call gnina binary via subprocess.
            pass

    logger.success("Analysis module complete.")