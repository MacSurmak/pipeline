"""
File: src/utils/pymol_export.py
Description: Generates standalone PyMOL sessions with top docked poses.
"""
import shutil
import pandas as pd
from pathlib import Path
from loguru import logger

def _extract_sdf_block(filepath: Path, target_title: str) -> str:
    if not filepath.exists(): return ""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    for block in content.split("$$$$"):
        lines = block.strip().splitlines()
        if not lines: continue
        title = lines[0].strip()
        if title == target_title:
            return block.strip() + "\n$$$$\n"
    return ""

def _build_multistate_sdf(source_sdf: Path, out_sdf: Path, ranked_titles: list, engine_label: str):
    """Assembles a multi-state SDF for a specific engine, ordered strictly by hit rank."""
    if not source_sdf.exists(): return False
    blocks = {}
    with open(source_sdf, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    for block in content.split("$$$$"):
        lines = block.strip().splitlines()
        if not lines: continue
        t = lines[0].strip()
        if t in ranked_titles and t not in blocks:
            blocks[t] = block.strip() + "\n$$$$\n"
            
    assembled = []
    for rank, t in enumerate(ranked_titles, start=1):
        if t in blocks:
            # Stamp rank into the molecule title line so PyMOL displays it
            b_lines = blocks[t].splitlines()
            b_lines[0] = f"Rank_{rank:02d}_{t} [{engine_label}]"
            assembled.append("\n".join(b_lines) + "\n")
            
    if assembled:
        with open(out_sdf, "w", encoding="utf-8") as f_out:
            f_out.write("$$$$\n".join(b.rstrip("$$$$\n") for b in assembled) + "\n$$$$\n")
        return True
    return False

def generate_pymol_session(df: pd.DataFrame, cwd: Path, out_dir: Path):
    if df.empty: return
    pymol_dir = out_dir / "pymol_session"
    pymol_dir.mkdir(parents=True, exist_ok=True)
    pml_file = pymol_dir / "view_hits.pml"
    
    unique_frames = sorted(list(set(str(f).replace("F", "") for f in df["Frame"])), key=lambda x: int(x))
    
    with open(pml_file, "w", encoding="utf-8") as pml:
        pml.write("# PyMOL Multi-Engine SBDD Session\nset bg_rgb, white\n")
        pml.write("set static_singletons, on\n\n") # Keeps receptor & native visible across all states!
        
        for frame_id in unique_frames:
            f_df = df[df["Frame"].isin([f"F{frame_id}", frame_id])]
            ranked_titles = f_df["Title"].tolist()
            
            group_members = []
            
            # 1. Receptor PDB
            rec_pdb = cwd / "01_prep_receptor" / "export" / f"receptor_prepared_f{frame_id}.pdb"
            if rec_pdb.exists():
                shutil.copy(rec_pdb, pymol_dir / rec_pdb.name)
                rec_obj = f"rec_f{frame_id}"
                pml.write(f"load {rec_pdb.name}, {rec_obj}\ncolor gray80, {rec_obj}\n")
                group_members.append(rec_obj)
                
            # 2. ALWAYS Load Native Reference Ligand (Green)
            ref_sdf = cwd / "01_prep_receptor" / "export" / f"ref_native_f{frame_id}.sdf"
            if ref_sdf.exists():
                shutil.copy(ref_sdf, pymol_dir / ref_sdf.name)
                native_obj = f"native_ref_f{frame_id}"
                pml.write(f"load {ref_sdf.name}, {native_obj}\ncolor green, {native_obj} and elem c\nshow sticks, {native_obj}\n")
                group_members.append(native_obj)
                
            dock_export = cwd / "04b_screening" / "export"
            
            # 3. Glide Multi-State (Cyan)
            glide_sdf = dock_export / f"glide_dock_f{frame_id}_poses.sdf"
            out_glide = pymol_dir / f"glide_f{frame_id}.sdf"
            if _build_multistate_sdf(glide_sdf, out_glide, ranked_titles, "Glide"):
                pml.write(f"load {out_glide.name}, glide_f{frame_id}\ncolor cyan, glide_f{frame_id} and elem c\n")
                group_members.append(f"glide_f{frame_id}")
                
            # 4. GNINA Dock Multi-State (Magenta)
            gnina_dock_sdf = dock_export / f"gnina_dock_f{frame_id}_poses.sdf"
            out_gnina = pymol_dir / f"gnina_dock_f{frame_id}.sdf"
            if _build_multistate_sdf(gnina_dock_sdf, out_gnina, ranked_titles, "GNINA_Dock"):
                pml.write(f"load {out_gnina.name}, gnina_dock_f{frame_id}\ncolor magenta, gnina_dock_f{frame_id} and elem c\n")
                group_members.append(f"gnina_dock_f{frame_id}")

            # 5. GNINA Minimized Multi-State (Orange)
            gnina_min_sdf = dock_export / f"gnina_minimize_f{frame_id}_poses.sdf"
            out_min = pymol_dir / f"gnina_min_f{frame_id}.sdf"
            if _build_multistate_sdf(gnina_min_sdf, out_min, ranked_titles, "GNINA_Min"):
                pml.write(f"load {out_min.name}, gnina_min_f{frame_id}\ncolor orange, gnina_min_f{frame_id} and elem c\n")
                group_members.append(f"gnina_min_f{frame_id}")
                
            if group_members:
                pml.write(f"group Frame_{frame_id}, {' '.join(group_members)}\n\n")
                
        pml.write("disable all\n")
        if unique_frames:
            pml.write(f"enable Frame_{unique_frames[0]}\nzoom Frame_{unique_frames[0]}\n")
            
    logger.success(f"PyMOL Multi-Engine Session exported to: {pymol_dir.absolute()}")