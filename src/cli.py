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
    p_dock = subparsers.add_parser("dock", help="Run Glide docking")
    dock_group = p_dock.add_mutually_exclusive_group(required=True)
    dock_group.add_argument("--native", action="store_true", help="Redock native ligand")
    dock_group.add_argument("--library", action="store_true", help="Dock library")
    p_dock.add_argument("--force", action="store_true")

    # ANALYSIS
    p_analyze = subparsers.add_parser("analyze", help="Run MM-GBSA, GNINA, RMSD")
    analyze_group = p_analyze.add_mutually_exclusive_group(required=True)
    analyze_group.add_argument("--native", action="store_true")
    analyze_group.add_argument("--library", action="store_true")
    p_analyze.add_argument("--force", action="store_true")

    # REPORTING
    p_report = subparsers.add_parser("report", help="Aggregate data and select hits")
    report_group = p_report.add_mutually_exclusive_group(required=True)
    report_group.add_argument("--native", action="store_true")
    report_group.add_argument("--library", action="store_true")

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
        from modules import dock; dock.run(cwd, is_native=args.native, force=args.force)
    elif args.command == "analyze":
        StateManager.require_file(cwd / "config.yaml", "analyze")
        from modules import analysis; analysis.run(cwd, is_native=args.native, force=args.force)
    elif args.command == "report":
        from modules import reporting; reporting.run(cwd, is_native=args.native)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()