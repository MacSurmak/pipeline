"""
File: src/modules/md_setup.py
Description: Generates topologies for GROMACS. Executes steps sequentially.
Clean UX: fast steps logged cleanly, progress bars only for heavy compute tasks.
"""
import os
import json
import pandas as pd
from pathlib import Path
from loguru import logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from core.logger import console
from core.config_parser import load_config
from core.state_manager import StateManager
from utils.cmd_runner import run_command
from utils.topology_prep import generate_tleap_bonds
from utils.gromacs_utils import inject_ligand_posre, add_thermostat_groups_by_name, fix_posre_indices

def _prepare_mdp_templates(target_dir: Path, config: dict):
    """Dynamically generates specific MDP files from 4 base templates based on system logic."""
    framework_dir = Path(os.environ.get("SBDD_FRAMEWORK_DIR"))
    tpl_dir = framework_dir / "templates" / "mdp"
    
    mem_cfg = config.get("membrane", {})
    is_mem = mem_cfg.get("enabled", False)
    
    md_cfg = config.get("md_simulation", {})
    sys_cfg = md_cfg.get("system_build", {})
    eq_cfg = md_cfg.get("equilibration", {})
    prod_cfg = md_cfg.get("production", {})

    temp = str(prod_cfg.get("temperature", 303.15))
    tc_grps = "PROT_LIG SOL_ION"
    pcoupl_type = "semiisotropic" if is_mem else "isotropic"
    ref_p = "1.0 1.0" if is_mem else "1.0"
    compressibility = "4.5e-5 4.5e-5" if is_mem else "4.5e-5"
    
    # Helper to calculate steps from picoseconds
    def dt_to_steps(ps: float, dt: float = 0.002) -> str:
        return str(int(ps / dt))

    # Base dictionary for all templates
    base_replacements = {
        "{TEMP}": temp,
        "{TC_GRPS}": tc_grps,
        "{PCOUPL_TYPE}": pcoupl_type,
        "{REF_P}": ref_p,
        "{COMPRESSIBILITY}": compressibility
    }

    # Define dynamic workflow stages
    if is_mem:
        stages = [
            {"name": "emA", "tpl": "minim.mdp", "integrator": "steep", "macros": {"{DEFINE}": "-DPOSRES -DPOSRES_LIG", "{EMSTEP}": "0.005", "{CONSTRAINTS}": "h-bonds"}},
            {"name": "emB", "tpl": "minim.mdp", "integrator": "steep", "macros": {"{DEFINE}": "", "{EMSTEP}": "0.001", "{CONSTRAINTS}": "h-bonds"}},
            {"name": "emC", "tpl": "minim.mdp", "integrator": "steep", "macros": {"{DEFINE}": "", "{EMSTEP}": "0.0001", "{CONSTRAINTS}": "none"}},
            {"name": "nvt_short", "tpl": "nvt.mdp", "integrator": "md", "macros": {"{DEFINE}": "-DPOSRES -DPOSRES_LIG", "{NSTEPS}": dt_to_steps(eq_cfg.get("nvt_short_ps", 50), 0.001), "{DT}": "0.001", "{GEN_VEL}": "yes", "{CONTINUATION}": "no"}},
            {"name": "nvt_posres_1000", "tpl": "nvt.mdp", "integrator": "md", "macros": {"{DEFINE}": "-DPOSRES -DPOSRES_LIG", "{NSTEPS}": dt_to_steps(eq_cfg.get("nvt_long_ps", 500)), "{DT}": "0.002", "{GEN_VEL}": "no", "{CONTINUATION}": "yes"}},
            {"name": "npt_berendsen_1000", "tpl": "npt.mdp", "integrator": "md", "macros": {"{DEFINE}": "-DPOSRES -DPOSRES_LIG", "{NSTEPS}": dt_to_steps(eq_cfg.get("npt_berendsen_ps", 1000)), "{PCOUPL}": "C-rescale", "{TAU_P}": "5.0"}},
            {"name": "npt_pr_250", "tpl": "npt.mdp", "integrator": "md", "macros": {"{DEFINE}": "-DPOSRES_250 -DPOSRES_LIG_250", "{NSTEPS}": dt_to_steps(eq_cfg.get("npt_pr_ps", 1000)), "{PCOUPL}": "Parrinello-Rahman", "{TAU_P}": "5.0"}},
            {"name": "npt_pr_50", "tpl": "npt.mdp", "integrator": "md", "macros": {"{DEFINE}": "-DPOSRES_50 -DPOSRES_LIG_50", "{NSTEPS}": dt_to_steps(eq_cfg.get("npt_pr_ps", 1000)), "{PCOUPL}": "Parrinello-Rahman", "{TAU_P}": "5.0"}},
            {"name": "npt_pr_unrestrained", "tpl": "npt.mdp", "integrator": "md", "macros": {"{DEFINE}": "", "{NSTEPS}": dt_to_steps(eq_cfg.get("npt_pr_ps", 1000)), "{PCOUPL}": "Parrinello-Rahman", "{TAU_P}": "5.0"}}
        ]
    else:
        stages = [
            {"name": "emA", "tpl": "minim.mdp", "integrator": "steep", "macros": {"{DEFINE}": "-DPOSRES -DPOSRES_LIG", "{EMSTEP}": "0.01", "{CONSTRAINTS}": "h-bonds"}},
            {"name": "nvt_short", "tpl": "nvt.mdp", "integrator": "md", "macros": {"{DEFINE}": "-DPOSRES -DPOSRES_LIG", "{NSTEPS}": dt_to_steps(eq_cfg.get("nvt_short_ps", 50), 0.001), "{DT}": "0.001", "{GEN_VEL}": "yes", "{CONTINUATION}": "no"}},
            {"name": "npt_pr_1000", "tpl": "npt.mdp", "integrator": "md", "macros": {"{DEFINE}": "-DPOSRES -DPOSRES_LIG", "{NSTEPS}": dt_to_steps(eq_cfg.get("npt_pr_ps", 1000)), "{PCOUPL}": "Parrinello-Rahman", "{TAU_P}": "5.0"}},
            {"name": "npt_pr_unrestrained", "tpl": "npt.mdp", "integrator": "md", "macros": {"{DEFINE}": "", "{NSTEPS}": dt_to_steps(eq_cfg.get("npt_pr_ps", 1000)), "{PCOUPL}": "Parrinello-Rahman", "{TAU_P}": "5.0"}}
        ]

    # Generate MDPs for all workflow stages
    for stage in stages:
        src = tpl_dir / stage["tpl"]
        if not src.exists():
            logger.critical(f"Missing base template: {src.name}")
            continue
        
        content = src.read_text()
        # Apply base replacements (Temperature, Groups)
        for k, v in base_replacements.items():
            content = content.replace(k, v)
        # Apply stage-specific macros
        for k, v in stage["macros"].items():
            content = content.replace(k, v)
            
        (target_dir / f"{stage['name']}.mdp").write_text(content)

    # Generate Production MD template
    prod_nsteps = dt_to_steps(prod_cfg.get("chunk_size_ns", 10) * 1000, 0.002)
    src_md = tpl_dir / "md.mdp"
    content_md = src_md.read_text()
    for k, v in base_replacements.items():
        content_md = content_md.replace(k, v)
    content_md = content_md.replace("{NSTEPS}", prod_nsteps)
    (target_dir / "md.mdp").write_text(content_md)

    # Dump execution workflow for md_run.py
    run_manifest = [{"name": s["name"], "integrator": s["integrator"]} for s in stages]
    (target_dir / "md_stages.json").write_text(json.dumps(run_manifest, indent=2))


def run(cwd: Path, targets: list, force: bool = False):
    config = load_config(cwd / "config.yaml")
    if not config.get("md_simulation", {}).get("enabled", False):
        logger.error("MD Simulation is disabled in config.yaml")
        return

    out_dir = cwd / "07_md"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    hitlist = pd.read_csv(cwd / "06_reporting" / "02_master_hitlist.csv")
    schrodinger_path = os.environ.get("SCHRODINGER")
    framework_dir = Path(os.environ.get("SBDD_FRAMEWORK_DIR"))
    worker_script = framework_dir / "src" / "schrodinger_workers" / "extract_md_files.py"
    gmx_bin = config.get("md_simulation", {}).get("gmx_path", "gmx_mpi")

    logger.info(f"Setting up MD for targets: {', '.join(targets)} using {gmx_bin}")
    processed_count = 0

    for target in targets:
        target_str = str(target).strip()
        
        # Target resolution
        if target_str.lower() in ["native", "ref", "reference"]:
            hit_row = hitlist[hitlist["Is_Native"] == True]
            folder_name = "native"
        elif target_str.isdigit():
            rank_idx = int(target_str) - 1
            library_hits = hitlist[hitlist["Is_Native"] == False]
            if 0 <= rank_idx < len(library_hits):
                hit_row = library_hits.iloc[[rank_idx]]
                folder_name = f"rank_{target_str}_{hit_row['Title'].values[0]}"
            else:
                logger.error(f"Rank '{target_str}' out of range. Skipping.")
                continue
        else:
            hit_row = hitlist[hitlist["Title"].str.lower() == target_str.lower()]
            folder_name = target_str

        if hit_row.empty:
            logger.error(f"Target '{target}' not found in hitlist. Skipping.")
            continue

        target_dir = out_dir / folder_name
        if StateManager.check_output_exists(target_dir / "topol.top", force):
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        
        frame_id = str(hit_row["Frame"].values[0]).replace("F", "")
        is_native = bool(hit_row["Is_Native"].values[0])
        actual_title = str(hit_row["Title"].values[0])

        logger.info(f"=== Initializing System Setup for: {folder_name} ({actual_title}) ===")

        # Step 0: Fast extraction (support both Glide PV and GNINA SDF inputs)
        dock_poses_dir = cwd / ("04a_redocking" if is_native else "04b_screening") / "poses"
        lig_source = dock_poses_dir / f"glide_dock_f{frame_id}_pv.maegz"
        
        if not lig_source.exists():
            gnina_candidate = dock_poses_dir / f"gnina_dock_f{frame_id}.sdf"
            if gnina_candidate.exists():
                lig_source = gnina_candidate

        rec_source = cwd / "01_prep_receptor" / "structures" / f"receptor_prepared_f{frame_id}.mae"
        
        if not run_command([
            f"{schrodinger_path}/run", "python3", str(worker_script),
            "--lig_mae", str(lig_source), "--lig_title", actual_title,
            "--rec_mae", str(rec_source), "--out_dir", str(target_dir)
        ], target_dir, "00_extract"):
            logger.error("Extraction failed. Aborting target.")
            continue

        charge = int((target_dir / "lig_charge.txt").read_text().strip()) if (target_dir / "lig_charge.txt").exists() else 0
        _prepare_mdp_templates(target_dir, config)

        # Step 1: Fast PDB Fix
        if not run_command(["python3", str(framework_dir / "src" / "utils" / "amber_prep_parmed.py"), "receptor.pdb", "receptor_fixed.pdb"], target_dir, "01_amber_prep"):
            continue

        # Step 2: Heavy Antechamber calculation (cached if already computed)
        gaff_mol2 = target_dir / "UNK_gaff2.mol2"
        frcmod_file = target_dir / "UNK.frcmod"
        
        if gaff_mol2.exists() and frcmod_file.exists() and not force:
            logger.info("Existing GAFF2/AM1-BCC parameters found. Skipping Antechamber.")
        else:
            with Progress(SpinnerColumn(), TextColumn("[bold yellow]{task.description}"), TimeElapsedColumn(), console=console) as prog_ante:
                t2 = prog_ante.add_task("Antechamber: Calculating AM1-BCC Charges & GAFF2...", total=None)
                ok_ante = run_command([
                    "antechamber", "-i", "ligand.mol2", "-fi", "mol2", 
                    "-o", "UNK_gaff2.mol2", "-fo", "mol2", "-at", "gaff2", 
                    "-c", "bcc", "-nc", str(charge), "-rn", "UNK"
                ], target_dir, "02_antechamber")
                if not ok_ante:
                    logger.error("Antechamber failed. Check logs/02_antechamber.log.")
                    continue
                run_command(["parmchk2", "-i", "UNK_gaff2.mol2", "-f", "mol2", "-o", "UNK.frcmod", "-s", "2"], target_dir, "03_parmchk")

        # Step 3: Fast Combine
        if not run_command(["python3", str(framework_dir / "src" / "utils" / "merge_complex_parmed.py"), "receptor_fixed.pdb", "UNK_gaff2.mol2", "complex.pdb"], target_dir, "04_combine"):
            continue

        # Step 4: Heavy Packmol calculation (Full progress bar)
        mem_cfg = config.get("membrane", {})
        sys_cfg = config.get("md_simulation", {}).get("system_build", {})
        
        if mem_cfg.get("enabled", False):
            lipids_cfg = mem_cfg.get("lipids", {})
            upper_lipids = lipids_cfg.get("upper_leaflet", {}).get("composition", "POPC:POPE:CHL1")
            upper_ratio = lipids_cfg.get("upper_leaflet", {}).get("ratio", "5:4:1")
            lower_lipids = lipids_cfg.get("lower_leaflet", {}).get("composition", "POPC:POPE:POPS:CHL1")
            lower_ratio = lipids_cfg.get("lower_leaflet", {}).get("ratio", "4:4:1:1")

            mem_complex_file = target_dir / "mem_complex.pdb"
            # Ensure file exists and is complete (finished packmol PDBs are > 1 MB)
            if mem_complex_file.exists() and mem_complex_file.stat().st_size > 1000000 and not force:
                logger.info("Complete mem_complex.pdb detected. Skipping Packmol-Memgen.")
            else:
                with Progress(
                    SpinnerColumn(), TextColumn("[bold magenta]{task.description}"), 
                    BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
                    TimeRemainingColumn(), console=console
                ) as prog_pack:
                    t4 = prog_pack.add_task("Packmol: Initializing Membrane Assembly...", total=360)
                    lipid_pad = str(sys_cfg.get('lipid_padding', 10.0))
                    wat_pad = str(sys_cfg.get('box_padding', 15.0))
                    pack_cmd = [
                        "packmol-memgen", "--pdb", "complex.pdb", 
                        "--lipids", f"{upper_lipids}//{lower_lipids}",
                        "--ratio", f"{upper_ratio}//{lower_ratio}",
                        "--preoriented", "--dist", lipid_pad, "--dist_wat", wat_pad,
                        "--salt", "--salt_c", "K+", "--saltcon", str(sys_cfg.get('salt_conc', 0.15)),
                        "--notprotonate", "-o", "mem_complex"
                    ]
                    if not run_command(pack_cmd, target_dir, "05_packmol", prog_pack, t4):
                        logger.error("Packmol failed. Check logs/05_packmol.log.")
                        continue
        else:
            (target_dir / "complex.pdb").rename(target_dir / "mem_complex.pdb")

        # Step 5: Fast tLEaP
        # Remove artificial TER inserted by packmol around caps (after ACE and before NME)
        mem_pdb = target_dir / "mem_complex.pdb"
        pdb_lines = mem_pdb.read_text().splitlines()
        fixed_pdb_lines = []
        last_chain = "A"
        for i, line in enumerate(pdb_lines):
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 22:
                ch = line[21].strip()
                if ch:
                    last_chain = ch

            # Strip artificial TER right after ACE or right before NME
            if line.startswith("TER"):
                if i > 0 and "ACE" in pdb_lines[i-1]:
                    continue
                if i + 1 < len(pdb_lines) and "NME" in pdb_lines[i+1]:
                    continue

            # Ensure both ACE and NME caps strictly share the protein chain ID
            if ("ACE" in line or "NME" in line) and (line.startswith("ATOM") or line.startswith("HETATM")):
                line = line[:21] + last_chain + line[22:]

            # Automatic AMBER protonation state mapping on the fly
            if "ASP" in line and ("HD2" in line or "2HD" in line):
                line = line[:17] + "ASH" + line[20:]
            elif "GLU" in line and ("HE2" in line or "2HE" in line):
                line = line[:17] + "GLH" + line[20:]

            fixed_pdb_lines.append(line)
        mem_pdb.write_text("\n".join(fixed_pdb_lines) + "\n")

        bonds_str = generate_tleap_bonds(mem_pdb)

        tleap_in = f"""source leaprc.protein.ff19SB
source leaprc.gaff2
source leaprc.lipid21
source leaprc.water.opc
loadAmberParams frcmod.ionslm_126_opc
UNK = loadmol2 UNK_gaff2.mol2
loadAmberParams UNK.frcmod
complex = loadPdb mem_complex.pdb
set default PBRadii mbondi3
addPdbResMap {{ {{"Zn2+" "ZN"}} {{"Ca2+" "CA"}} {{"Mg2+" "MG"}} {{"Na+" "NA"}} {{"K+" "K"}} {{"Cl-" "CL"}} }}
{bonds_str}
{"solvateBox complex OPCBOX " + str(sys_cfg.get("box_padding", 15.0)) if not mem_cfg.get("enabled", False) else ""}
# Neutralize system by replacing random water with counter-ions
addIonsRand complex Cl- 0
addIonsRand complex K+ 0
charge complex
saveAmberParm complex complex.prmtop complex.inpcrd
savepdb complex complex_tleap.pdb
quit
"""
        (target_dir / "tleap.in").write_text(tleap_in)
        if not run_command(["tleap", "-f", "tleap.in"], target_dir, "06_tleap"):
            logger.error("tLEaP failed. Check logs/06_tleap.log.")
            continue

        # Step 6: Fast ParmEd
        if not run_command(["python3", "-c", "import parmed as p; s=p.load_file('complex.prmtop','complex.inpcrd'); s.save('topol.top', format='gromacs'); s.save('step3_input.gro')"], target_dir, "07_parmed"):
            continue

        # Step 7: Declash via original script
        with Progress(
            SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), 
            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), console=console
        ) as prog_declash:
            t7 = prog_declash.add_task("Declash: Relieving Intermolecular Overlaps...", total=120)
            declash_cmd = [
                "python3", "-u", str(framework_dir / "software" / "declash_gmx_molecules_v2.py"),
                "-f", "step3_input.gro", "-p", "topol.top", "-o", "step3_declashed.gro",
                "--fix-moltype", "system1", "--fix-moltype", "UNK",
                "--cutoff", "0.10", "--niter", "120", "--max-move", "0.10", "-v"
            ]
            if not run_command(declash_cmd, target_dir, "08_declash", prog_declash, t7):
                logger.error("Declash failed. Check logs/08_declash.log.")
                continue
        
        # Step 8: Generate clean index and construct PROT_LIG / SOL_ION strictly by name
        q_script = target_dir / "make_ndx.in"
        q_script.write_text("q\n")
        # Ensure we use declashed coordinates to avoid geometry mismatches
        base_gro = "step3_declashed.gro" if (target_dir / "step3_declashed.gro").exists() else "step3_input.gro"
        
        if not run_command([gmx_bin, "make_ndx", "-f", base_gro, "-o", "index.ndx"], target_dir, "08_make_ndx", stdin=q_script):
            continue

        try:
            add_thermostat_groups_by_name(target_dir / "index.ndx")
            logger.info("Generated [ PROT_LIG ] and [ SOL_ION ] strictly by name.")
        except Exception as e:
            logger.error(f"Failed to generate thermostat groups by name: {e}")
            continue

        # Step 9: Position Restraints (Multi-level scaling)
        posre_p_in = target_dir / "posre_p.in"
        posre_p_in.write_text("Protein\n")
        posre_l_in = target_dir / "posre_l.in"
        posre_l_in.write_text("UNK\n")
        
        fc_levels = [("1000", ""), ("250", "_250"), ("50", "_50")]
        
        for fc, suffix in fc_levels:
            run_command([gmx_bin, "genrestr", "-f", base_gro, "-n", "index.ndx", "-o", f"posre{suffix}.itp", "-fc", fc, fc, fc], target_dir, f"09_posre_prot{suffix}", stdin=posre_p_in)
            run_command([gmx_bin, "genrestr", "-f", base_gro, "-n", "index.ndx", "-o", f"posre_lig{suffix}.itp", "-fc", fc, fc, fc], target_dir, f"10_posre_lig{suffix}", stdin=posre_l_in)
            # Reindex global -> local (1..N) to match inside [ moleculetype ] safely
            fix_posre_indices(target_dir / f"posre{suffix}.itp")
            fix_posre_indices(target_dir / f"posre_lig{suffix}.itp")
            
        inject_ligand_posre(target_dir / "topol.top")
        processed_count += 1
        logger.success(f"System fully prepared and validated for {folder_name}.")

    if processed_count > 0:
        logger.success(f"MD Setup completed! {processed_count} system(s) ready in {out_dir.name}/")
    else:
        logger.error("MD Setup failed for all requested targets.")