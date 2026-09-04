"""
File: src/modules/workspace_init.py
Description: Initializes the current working directory with pipeline templates.
"""
import shutil
from pathlib import Path
from loguru import logger

def init_workspace(framework_dir: Path, target_dir: Path) -> None:
    templates_dir = framework_dir / "templates"
    files_to_copy = {
        "config_template.yaml": "config.yaml",
        "gridgen_template.in": "gridgen.in",
        "docking_template.in": "docking.in"
    }

    logger.info(f"Initializing SBDD workspace in {target_dir.absolute()}")

    for src_name, dst_name in files_to_copy.items():
        src_path = templates_dir / src_name
        dst_path = target_dir / dst_name

        if not src_path.exists():
            logger.error(f"Template not found: {src_path}")
            continue

        if dst_path.exists():
            logger.warning(f"File {dst_name} already exists. Skipping.")
            continue

        shutil.copy(src_path, dst_path)
        logger.success(f"Generated template: {dst_name}")

    # Explicit separation of redocking and screening flows
    directories = [
        "00_input",
        "01_prep_receptor",
        "02_prep_ligands",
        "03_grid",
        "04a_redocking",
        "05a_redock_analysis",
        "04b_screening",
        "05b_screen_analysis"
    ]
    
    for directory in directories:
        dir_path = target_dir / directory
        dir_path.mkdir(exist_ok=True)
        
    logger.info("Workspace ready! Place input files into '00_input/' and edit 'config.yaml'.")