"""
File: src/engines/gnina.py
Description: Execution logic for GNINA docking.
"""
import shutil
from pathlib import Path
from core.state_manager import StateManager
from utils.cmd_runner import run_command

def run_docking(cwd, config, in_ligands, grid_zip, frame_id, out_dir, force, progress, task_id):
    job_name = f"gnina_dock_f{frame_id}"
    poses_dir = out_dir / "poses"
    export_dir = out_dir / "export"
    expected_out = poses_dir / f"{job_name}.sdf"
    
    gnina_settings = config.get("docking", {}).get("gnina", {})
    rec_pdb = cwd / "01_prep_receptor" / "export" / f"receptor_prepared_f{frame_id}.pdb"
    ref_lig = cwd / "01_prep_receptor" / "export" / f"ref_native_f{frame_id}.sdf"
    
    StateManager.require_file(rec_pdb, "gnina docking (receptor)")
    StateManager.require_file(ref_lig, "gnina docking (autobox ligand)")
    
    cmd = [
        "gnina", "-r", str(rec_pdb.resolve()), "-l", str(in_ligands.resolve()),
        "--autobox_ligand", str(ref_lig.resolve()), "-o", str(expected_out.resolve()),
        "--exhaustiveness", str(gnina_settings.get("exhaustiveness", 8)),
        "--cnn_scoring", str(gnina_settings.get("cnn_scoring", "rescore")),
        "--num_modes", str(gnina_settings.get("num_modes", 3))
    ]
    
    run_command(cmd, out_dir, job_name, progress=progress, task_id=task_id)
    
    scores = []
    if expected_out.exists():
        sdf_out = export_dir / f"{job_name}_poses.sdf"
        shutil.copy(expected_out, sdf_out)
        
        with open(sdf_out, "r") as f_sdf:
            lines = f_sdf.readlines()
            for i, line in enumerate(lines):
                if "<CNNaffinity>" in line:
                    try: scores.append(float(lines[i+1].strip()))
                    except ValueError: pass
                    
    return expected_out, scores