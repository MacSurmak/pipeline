"""
File: src/cli.py
Description: Main CLI entry point. Routes commands to specific pipeline modules.
"""
import argparse
import sys
from pathlib import Path
from loguru import logger

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.logger import setup_logger
from core.state_manager import StateManager

def main():
    parser = argparse.ArgumentParser(description="Modular SBDD Pipeline")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available modules")

    # INIT, PREP_REC, GRID (as before)
    subparsers.add_parser("init", help="Initialize workspace")
    p_rec = subparsers.add_parser("prep_rec", help="Prepare receptor")
    p_rec.add_argument("--force", action="store_true")
    p_grid = subparsers.add_parser("grid", help="Generate grids")
    p_grid.add_argument("--force", action="store_true")

    # PREP_LIG
    p_lig = subparsers.add_parser("prep_lig", help="Prepare ligands")
    group = p_lig.add_mutually_exclusive_group(required=True)
    group.add_argument("--native", action="store_true")
    group.add_argument("--library", action="store_true")
    p_lig.add_argument("--force", action="store_true")

    # DOCKING
    p_dock = subparsers.add_parser("dock", help="Run docking (Glide or GNINA)")
    dock_target = p_dock.add_mutually_exclusive_group(required=True)
    dock_target.add_argument("--native", action="store_true", help="Redock native ligand")
    dock_target.add_argument("--library", action="store_true", help="Dock library")
    dock_engine = p_dock.add_mutually_exclusive_group(required=True)
    dock_engine.add_argument("--glide", action="store_true", help="Use Glide")
    dock_engine.add_argument("--gnina", action="store_true", help="Use GNINA")
    p_dock.add_argument("--force", action="store_true")

    # GNINA RESCORE/MINIMIZE
    p_gnina = subparsers.add_parser("gnina", help="Rescore or minimize Glide poses using GNINA")
    gnina_target = p_gnina.add_mutually_exclusive_group(required=True)
    gnina_target.add_argument("--native", action="store_true")
    gnina_target.add_argument("--library", action="store_true")
    gnina_mode = p_gnina.add_mutually_exclusive_group(required=True)
    gnina_mode.add_argument("--score_only", action="store_true")
    gnina_mode.add_argument("--minimize", action="store_true")
    p_gnina.add_argument("--force", action="store_true")

    # ANALYSIS
    p_analyze = subparsers.add_parser("analyze", help="Run MM-GBSA, GNINA, RMSD")
    analyze_group = p_analyze.add_mutually_exclusive_group(required=True)
    analyze_group.add_argument("--native", action="store_true")
    analyze_group.add_argument("--library", action="store_true")
    p_analyze.add_argument("--force", action="store_true")

    # REPORTING
    p_report = subparsers.add_parser("report", help="Aggregate data, compare with native, and select hits")
    p_report.add_argument("--force", action="store_true", help="Force recalculation and overwrite reports")

    # MD SETUP
    p_md_setup = subparsers.add_parser("md_setup", help="Generate bash scripts and topologies for MD")
    p_md_setup.add_argument("--hits", type=str, required=True, help="Comma-separated hit titles (e.g., 'native,ZINC123')")
    p_md_setup.add_argument("--force", action="store_true", help="Overwrite existing MD setups")

    # MD RUN
    p_md_run = subparsers.add_parser("md_run", help="Execute generated MD bash scripts chunk by chunk")
    p_md_run.add_argument("--targets", type=str, help="Comma-separated targets to run (runs all setups if omitted)")

    # ADMET PROFILING
    p_admet = subparsers.add_parser("admet", help="Run ADMET-AI profiling and traffic-light safety assessment")
    admet_target = p_admet.add_mutually_exclusive_group(required=False)
    admet_target.add_argument("--hits", action="store_true", help="Profile top hits from reporting (default)")
    admet_target.add_argument("--library", action="store_true", help="Profile full library from 00_input")
    p_admet.add_argument("--input", type=str, help="Path to custom .smi or .sdf file")
    p_admet.add_argument("--top_n", type=int, default=20, help="Number of top hits to profile (default: 20)")
    p_admet.add_argument("--force", action="store_true", help="Force recalculation")

    args = parser.parse_args()

    cwd = Path.cwd()
    log_file = cwd / "pipeline.log" if (cwd / "config.yaml").exists() else None
    setup_logger(log_file=log_file, verbose=args.verbose)
    framework_dir = SRC_DIR.parent

    if args.command == "init":
        from modules.workspace_init import init_workspace
        init_workspace(framework_dir, cwd)
    elif args.command == "prep_rec":
        StateManager.require_file(cwd / "config.yaml", "prep_rec")
        from modules import prep_receptor; prep_receptor.run(cwd, force=args.force)
    elif args.command == "prep_lig":
        StateManager.require_file(cwd / "config.yaml", "prep_lig")
        from modules import prep_ligands; prep_ligands.run(cwd, is_native=args.native, force=args.force)
    elif args.command == "grid":
        StateManager.require_file(cwd / "config.yaml", "grid")
        from modules import grid_gen; grid_gen.run(cwd, force=args.force)
    elif args.command == "dock":
        StateManager.require_file(cwd / "config.yaml", "dock")
        engine = "glide" if args.glide else "gnina"
        from modules import dock; dock.run(cwd, is_native=args.native, engine_name=engine, force=args.force)
    elif args.command == "gnina":
        StateManager.require_file(cwd / "config.yaml", "gnina")
        mode = "score_only" if args.score_only else "minimize"
        from modules import gnina_rescore; gnina_rescore.run(cwd, is_native=args.native, mode=mode, force=args.force)
    elif args.command == "analyze":
        StateManager.require_file(cwd / "config.yaml", "analyze")
        from modules import analysis; analysis.run(cwd, is_native=args.native, force=args.force)
    elif args.command == "report":
        from modules import reporting
        reporting.run(cwd, force=args.force)
    elif args.command == "md_setup":
        StateManager.require_file(cwd / "config.yaml", "md_setup")
        StateManager.require_file(cwd / "06_reporting" / "02_master_hitlist.csv", "md_setup")
        from modules import md_setup
        targets = [t.strip() for t in args.hits.split(",")]
        md_setup.run(cwd, targets, force=args.force)
    elif args.command == "md_run":
        StateManager.require_file(cwd / "config.yaml", "md_run")
        from modules import md_run
        targets = [t.strip() for t in args.targets.split(",")] if args.targets else []
        md_run.run(cwd, targets)
    elif args.command == "admet":
        StateManager.require_file(cwd / "config.yaml", "admet")
        from modules import admet
        target_mode = "library" if args.library else "hits"
        admet.run(cwd, mode=target_mode, custom_input=args.input, top_n=args.top_n, force=args.force)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()