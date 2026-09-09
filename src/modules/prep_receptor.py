"""
Receptor preparation module. Orchestrates structure splitting and PrepWizard over ensembles.
"""
import os
import re
import shutil
import json
from pathlib import Path
from loguru import logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from core.logger import console
from core.config_parser import load_config
from core.state_manager import StateManager
from utils.cmd_runner import run_command
from utils.count_structures import get_structure_count
from utils.fs import get_frames_sorted
from utils.ui import print_summary_table

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
    
    total_detected = get_structure_count(input_pdb)
    selected_model = int(rec_config.get("model_index", 0))
    
    if selected_model > 0 and selected_model <= total_detected:
        frames_to_process = [selected_model]
        logger.info(f"Processing single specified NMR model: {selected_model} (out of {total_detected})")
    else:
        frames_to_process = list(range(1, total_detected + 1))
        logger.info(f"Processing entire ensemble: {total_detected} frame(s) in {input_pdb.name}")

    membrane_cfg = config.get("membrane", {})
    membrane_summary = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        split_task = progress.add_task("Splitting & Orienting (PPM 3.0)...", total=len(frames_to_process))
        
        for i in frames_to_process:
            progress.update(split_task, description=f"[bold cyan]Frame {i}: Splitting & Orienting...")
            
            if StateManager.check_output_exists(raw_dir / f"receptor_raw_f{i}.mae", force):
                progress.advance(split_task)
                continue

            split_cmd = [
                f"{schrodinger_path}/run", "python3", str(worker_script),
                "--input", str(input_pdb),
                "--outdir", str(raw_dir),
                "--ligand", rec_config.get("native_ligand_resname", "LIG"),
                "--frame_index", str(i)
            ]
            
            cofactors = rec_config.get("keep_cofactors", [])
            if cofactors:
                split_cmd.extend(["--cofactors"] + cofactors)
                
            if membrane_cfg.get("enabled", False):
                chain_str = ",".join(membrane_cfg.get("chains", ["A"])) if isinstance(membrane_cfg.get("chains"), list) else str(membrane_cfg.get("chains", "A"))
                split_cmd.extend([
                    "--membrane_type", membrane_cfg.get("type", "MOM"),
                    "--membrane_topo", membrane_cfg.get("topology", "in"),
                    "--membrane_chains", chain_str
                ])

            if not run_command(split_cmd, out_dir, f"01_split_receptor_f{i}"):
                return
            
            if membrane_cfg.get("enabled", False):
                json_file = raw_dir / f"membrane_info_f{i}.json"
                if json_file.exists():
                    with open(json_file, "r") as jf:
                        m_data = json.load(jf)
                    logger.info(
                        f"Frame {i} PPM 3.0: Thickness={m_data['thickness']:.1f} Å, "
                        f"Tilt={m_data['tilt']:.0f}°, ΔG_transfer={m_data['dg']:.1f} kcal/mol"
                    )
                    membrane_summary.append({"Frame": str(i), "Thickness (Å)": f"{m_data['thickness']:.1f}", 
                                             "Tilt Angle": f"{m_data['tilt']:.0f}°", "ΔG_transfer (kcal/mol)": f"{m_data['dg']:.1f}"})

            ref_file = raw_dir / f"ref_native_f{i}.mae"
            if ref_file.exists():
                shutil.move(str(ref_file), str(struct_dir / ref_file.name))

            progress.advance(split_task)

    # 2. PrepWizard
    logger.info("Step 2: Running Protein Preparation Wizard on all frames")
    prepwizard_bin = f"{schrodinger_path}/utilities/prepwizard"
    pw_settings = rec_config.get("prep_wizard", {})
    
    raw_receptors = get_frames_sorted(raw_dir, "receptor_raw_f*.mae")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed}/{task.total} frames)"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        prep_task = progress.add_task("[magenta]PrepWizard...", total=len(raw_receptors))
        for raw_rec, frame_id in raw_receptors:
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

            if membrane_cfg.get("enabled", False):
                progress.update(prep_task, description=f"[magenta]Applying Membrane to Frame {frame_id}...")
                mem_worker = framework_dir / "src" / "schrodinger_workers" / "apply_membrane.py"
                tmp_out = struct_dir / f"receptor_prepared_f{frame_id}_mem.mae"
                info_file = raw_dir / f"membrane_info_f{frame_id}.txt"
                
                mem_cmd = [
                    f"{schrodinger_path}/run", "python3", "-u", str(mem_worker),
                    "--input", str(final_output),
                    "--output", str(tmp_out),
                    "--info", str(info_file)
                ]

                if not run_command(mem_cmd, out_dir, f"03_membrane_f{frame_id}") or not tmp_out.exists():
                    logger.critical(f"Membrane application failed for frame {frame_id}. Aborting.")
                    return
                shutil.move(str(tmp_out), str(final_output))

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

    if membrane_cfg.get("enabled", False) and membrane_summary:
        print_summary_table("Membrane Orientation Summary (PPM 3.0)", 
                            ["Frame", "Thickness (Å)", "Tilt Angle", "ΔG_transfer (kcal/mol)"], 
                            membrane_summary)
