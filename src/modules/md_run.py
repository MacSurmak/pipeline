"""
File: src/modules/md_run.py
Description: Executes MD via GROMACS. Handles EM, NVT, NPT, and Production chunking.
"""
import json
from pathlib import Path
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from core.logger import console
from core.config_parser import load_config
from core.state_manager import StateManager
from utils.cmd_runner import run_command

def _get_safe_hw_args(hw_args: list, integrator: str, stage_name: str) -> list:
    """
    Safely strips incompatible GPU offloading arguments.
    Steepest descent and Berendsen/C-rescale often crash with `-update gpu` or `-pme gpu`.
    """
    safe_args = []
    skip_next = False
    for arg in hw_args:
        if skip_next:
            skip_next = False
            continue
        if integrator == "steep" and arg in ["-pme", "-update"]:
            skip_next = True
            continue
        if "berendsen" in stage_name.lower() and arg == "-update":
            skip_next = True
            continue
        safe_args.append(arg)
    return safe_args

def _get_nsteps_from_mdp(mdp_path: Path) -> int:
    """Parses nsteps from an MDP file to provide accurate progress bar totals."""
    if not mdp_path.exists(): return 1
    for line in mdp_path.read_text().splitlines():
        if line.strip().startswith("nsteps"):
            try: return int(line.split("=")[1].strip())
            except ValueError: pass
    return 1

def _run_stage_safely(gmx_bin: str, tdir: Path, name: str, prev_name: str, hw_args: list, integrator: str, 
                      progress, step_task_id, maxwarn: str = "2", step_offset: int = 0) -> bool:
    """Executes a single MD stage strictly ensuring idempotency and failure cleanup."""
    out_tpr = tdir / f"{name}.tpr"
    out_gro = tdir / f"{name}.gro"
    is_steep = integrator == "steep"
    total_steps = _get_nsteps_from_mdp(tdir / f"{name}.mdp") if not is_steep else None
    
    if out_gro.exists():
        if progress and step_task_id:
            progress.update(step_task_id, completed=total_steps or 1)
        return True

    if name == "emA":
        input_gro = "step3_declashed.gro" if (tdir / "step3_declashed.gro").exists() else "step3_input.gro"
    else:
        input_gro = f"{prev_name}.gro"
    
    cmd_grompp = [
        gmx_bin, "grompp", "-f", f"{name}.mdp", "-o", str(out_tpr.name), 
        "-c", input_gro, "-r", input_gro, "-p", "topol.top", "-n", "index.ndx", "-maxwarn", maxwarn
    ]
    
    if integrator != "steep" and (tdir / f"{prev_name}.cpt").exists():
        cmd_grompp.extend(["-t", f"{prev_name}.cpt"])

    if not run_command(cmd_grompp, tdir, f"gmx_grompp_{name}"):
        out_tpr.unlink(missing_ok=True)
        return False

    safe_hw = _get_safe_hw_args(hw_args, integrator, name)
    cmd_mdrun = [gmx_bin, "mdrun", "-deffnm", name] + safe_hw

    if not run_command(cmd_mdrun, tdir, f"gmx_mdrun_{name}", progress=progress, task_id=step_task_id, 
                       total_override=total_steps, step_offset=step_offset):
        logger.error(f"Stage '{name}' failed. Cleaning up corrupted outputs to protect idempotency.")
        out_gro.unlink(missing_ok=True)
        (tdir / f"{name}.cpt").unlink(missing_ok=True)
        return False
        
    return True

def run(cwd: Path, targets: list):
    md_dir = cwd / "07_md"
    config = load_config(cwd / "config.yaml")
    hw_args = config["md_simulation"]["production"]["hardware"]["mdrun_args"].split()
    gmx_bin = config.get("md_simulation", {}).get("gmx_path", "gmx_mpi")
    
    total_time_ns = config["md_simulation"]["production"]["total_time_ns"]
    chunk_ns = config["md_simulation"]["production"]["chunk_size_ns"]
    total_chunks = int(total_time_ns / chunk_ns)

    if not md_dir.exists():
        logger.error("No MD directory found. Run 'md_setup' first.")
        return

    if not targets:
        targets = [p.name for p in md_dir.iterdir() if p.is_dir() and (p / "topol.top").exists()]

    logger.info(f"Starting GROMACS Pipeline for: {', '.join(targets)} using {gmx_bin}")

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        # Level 1: System / Target progress
        sys_task = progress.add_task("[bold yellow]Systems/Targets", total=len(targets))
        
        for target in targets:
            tdir = md_dir / target
            StateManager.require_file(tdir / "topol.top", f"md_run for {target}")
            
            stages_json = tdir / "md_stages.json"
            if not stages_json.exists():
                logger.error(f"Missing md_stages.json in {tdir.name}. Run md_setup with --force.")
                progress.advance(sys_task)
                continue
                
            stages = json.loads(stages_json.read_text())
            total_stages = len(stages) + total_chunks
            
            # Level 2: Workflow stage progress (Equilibration stages + Production chunks)
            stage_task = progress.add_task(f"[bold cyan]{target} Workflow Stages", total=total_stages)
            
            prev_stage = "step3_input"
            eq_success = True
            
            # Equilibration loop
            for stage in stages:
                s_name = stage["name"]
                is_steep = stage["integrator"] == "steep"
                total_steps = _get_nsteps_from_mdp(tdir / f"{s_name}.mdp") if not is_steep else None
                
                # Level 3: Dynamic convergence spinner for EM vs Step progress bar for MD
                task_desc = f"[bold green]{s_name}: Relaxing (EM)..." if is_steep else f"[bold green]{s_name}: Steps"
                step_task = progress.add_task(task_desc, total=total_steps)
                
                ok = _run_stage_safely(gmx_bin, tdir, s_name, prev_stage, hw_args, stage["integrator"], 
                                       progress, step_task)
                progress.remove_task(step_task)
                
                if not ok:
                    eq_success = False
                    break
                    
                prev_stage = s_name
                progress.advance(stage_task)
                
            if not eq_success:
                logger.error(f"Pipeline aborted for {target} during equilibration.")
                progress.remove_task(stage_task)
                progress.advance(sys_task)
                continue

            # Production MD chunks loop
            prod_success = True
            chunk_steps = _get_nsteps_from_mdp(tdir / "md.mdp")

            for chunk in range(1, total_chunks + 1):
                chunk_name = f"md_{chunk}"
                prev_name = f"md_{chunk-1}" if chunk > 1 else prev_stage
                
                step_task = progress.add_task(f"[bold green]{chunk_name}: Steps", total=chunk_steps)
                
                if (tdir / f"{chunk_name}.gro").exists():
                    progress.update(step_task, completed=chunk_steps)
                    progress.remove_task(step_task)
                    progress.advance(stage_task)
                    continue
                    
                if not (tdir / f"{chunk_name}.tpr").exists():
                    if chunk == 1:
                        cmd_grompp = [gmx_bin, "grompp", "-f", "md.mdp", "-o", f"{chunk_name}.tpr", 
                                      "-c", f"{prev_name}.gro", "-t", f"{prev_name}.cpt", 
                                      "-p", "topol.top", "-n", "index.ndx", "-maxwarn", "2"]
                        if not run_command(cmd_grompp, tdir, f"gmx_md_tpr_{chunk}"):
                            prod_success = False
                            progress.remove_task(step_task)
                            break
                    else:
                        extension_ps = int(chunk_ns * 1000)
                        if not run_command([gmx_bin, "convert-tpr", "-s", f"{prev_name}.tpr", 
                                            "-extend", str(extension_ps), "-o", f"{chunk_name}.tpr"], 
                                           tdir, f"gmx_extend_{chunk}"):
                            prod_success = False
                            progress.remove_task(step_task)
                            break

                md_cmd = [gmx_bin, "mdrun", "-deffnm", chunk_name] + hw_args
                if (tdir / f"{chunk_name}.cpt").exists():
                    md_cmd.extend(["-cpi", f"{chunk_name}.cpt", "-append"])
                elif chunk > 1 and (tdir / f"{prev_name}.cpt").exists():
                    md_cmd.extend(["-cpi", f"{prev_name}.cpt", "-noappend"])
                    
                # GROMACS accumulates step count in extended runs; calculate offset to start from 0
                step_offset = (chunk - 1) * chunk_steps
                ok = run_command(md_cmd, tdir, f"gmx_md_run_{chunk}", progress=progress, task_id=step_task, 
                                 total_override=chunk_steps, step_offset=step_offset)
                                 
                progress.remove_task(step_task)

                if not ok:
                    logger.error(f"[{target}] Chunk {chunk} failed. Rolling back corrupted files.")
                    (tdir / f"{chunk_name}.gro").unlink(missing_ok=True)
                    (tdir / f"{chunk_name}.cpt").unlink(missing_ok=True)
                    prod_success = False
                    break
                    
                progress.advance(stage_task)
            
            progress.remove_task(stage_task)

            if prod_success:
                logger.success(f"Production MD finished for {target}.")
                
            progress.advance(sys_task)

    logger.success("MD execution orchestrator finished.")