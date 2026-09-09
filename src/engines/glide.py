"""
File: src/engines/glide.py
Description: Execution logic for Glide docking.
"""
import os
import csv
import shutil
from pathlib import Path
from utils.cmd_runner import run_command

def run_docking(cwd, config, in_ligands, grid_zip, frame_id, out_dir, force, progress, task_id):
    schrodinger_path = os.environ.get("SCHRODINGER")
    glide_bin = f"{schrodinger_path}/glide"
    framework_dir = Path(os.environ.get("SBDD_FRAMEWORK_DIR"))
    
    job_name = f"glide_dock_f{frame_id}"
    poses_dir = out_dir / "poses"
    export_dir = out_dir / "export"
    tables_dir = out_dir / "tables"
    inputs_dir = out_dir / "inputs"
    expected_out = poses_dir / f"{job_name}_pv.maegz"
    
    glide_cfg = config.get("docking", {}).get("glide", {})
    dock_in = inputs_dir / f"{job_name}.in"
    
    postdock = "yes" if glide_cfg.get("postdock", True) else "no"
    epik_penalties = "yes" if glide_cfg.get("epik_penalties", True) else "no"
    sample_rings = "yes" if glide_cfg.get("sample_rings", True) else "no"
    write_csv = "yes" if glide_cfg.get("write_csv", True) else "no"
    
    dock_content = (
        f"GRIDFILE {grid_zip.resolve()}\n"
        f"LIGANDFILE {in_ligands.resolve()}\n"
        f"PRECISION {glide_cfg.get('precision', 'SP')}\n"
        f"POSES_PER_LIG {glide_cfg.get('poses_per_lig', 3)}\n"
        f"POSTDOCK {postdock}\n"
        f"EPIK_PENALTIES {epik_penalties}\n"
        f"SAMPLE_RINGS {sample_rings}\n"
        f"WRITE_CSV {write_csv}\n"
        f"FORCEFIELD {glide_cfg.get('forcefield', 'OPLS_2005')}\n"
    )
    dock_in.write_text(dock_content)
        
    cmd = [glide_bin, str(dock_in.resolve()), "-OVERWRITE", "-WAIT"]
    run_command(cmd, inputs_dir, job_name, progress=progress, task_id=task_id)

    # Move results to target directories
    raw_pv = inputs_dir / f"{job_name}_pv.maegz"
    if raw_pv.exists():
        shutil.move(str(raw_pv), str(expected_out))
    raw_csv = inputs_dir / f"{job_name}.csv"
    if raw_csv.exists():
        shutil.copy(str(raw_csv), str(out_dir / f"{job_name}.csv"))
        
    if expected_out.exists():
        sdf_out = export_dir / f"glide_dock_f{frame_id}_poses.sdf"
        export_worker = framework_dir / "src" / "schrodinger_workers" / "pv_to_sdf.py"
        cmd_sdf = [
            f"{schrodinger_path}/run", "python3", str(export_worker),
            "--pv", str(expected_out.resolve()), "--sdf", str(sdf_out.resolve())
        ]
        run_command(cmd_sdf, out_dir, f"export_sdf_f{frame_id}")

    scores = []
    csv_path = out_dir / f"{job_name}.csv"
    if csv_path.exists():
        with open(csv_path, "r") as f_csv:
            for row in csv.DictReader(f_csv):
                score_keys = [k for k in row.keys() if k and ("gscore" in k.lower() or "docking_score" in k.lower())]
                if score_keys:
                    try: scores.append(float(row[score_keys[0]]))
                    except ValueError: pass
        shutil.move(str(csv_path), str(tables_dir / csv_path.name))
        
    return expected_out, scores