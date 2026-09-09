"""
File: src/utils/data_extractors.py
Description: Parsers for extracting scoring metrics from CSV and SDF files.
"""
import csv
from pathlib import Path
import pandas as pd
from loguru import logger

def parse_glide_csv(csv_path: Path) -> dict:
    """Strict parser for Glide CSV tables, prioritizing docking_score with Epik penalties."""
    data = {}
    if not csv_path.exists(): return data
    with open(csv_path, "r", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title") or row.get("Title") or row.get("s_m_title")
            if not title or title in data: continue
            
            score = None
            # Strict match for docking score (with Epik penalties)
            for k in ["r_i_docking_score", "docking_score"]:
                if k in row and row[k]:
                    try: score = float(row[k]); break
                    except ValueError: pass
            # Fallback to gscore
            if score is None:
                for k in ["r_i_glide_gscore", "gscore"]:
                    if k in row and row[k]:
                        try: score = float(row[k]); break
                        except ValueError: pass
            if score is not None:
                data[title] = {"GlideScore": score}
    return data

def build_funnel_whitelist(dock_dir: Path, out_csv: Path, top_k_frames: int, percentile_cutoff: float, force_include: list = None) -> bool:
    """Aggregates Glide CSVs to identify top K frames per ligand for top N% + forced controls."""
    csv_files = list((dock_dir / "tables").glob("glide_dock_f*.csv"))
    if not csv_files:
        return False
        
    all_data = []
    for f in csv_files:
        frame_id = f.stem.split("_f")[-1]
        data = parse_glide_csv(f)
        for title, scores in data.items():
            all_data.append({"Title": title, "Frame": frame_id, "GlideScore": scores["GlideScore"]})
            
    df = pd.DataFrame(all_data)
    if df.empty:
        return False
        
    # Determine the global best score per ligand
    best_scores = df.groupby("Title")["GlideScore"].min()
    
    # Calculate cutoff threshold (GlideScore is negative, lower is better, so we want the bottom X percentile)
    threshold = best_scores.quantile(percentile_cutoff)
    
    # Keep ligands passing threshold OR explicitly forced (controls/benchmarks)
    forced_set = set(force_include or [])
    passing_score = set(best_scores[best_scores <= threshold].index)
    kept_ligands = passing_score | (forced_set & set(df["Title"]))
    
    df_filtered = df[df["Title"].isin(kept_ligands)]
    
    # Keep only Top K frames for each surviving ligand
    whitelist = df_filtered.sort_values("GlideScore").groupby("Title").head(top_k_frames)
    
    whitelist[["Title", "Frame"]].to_csv(out_csv, index=False)
    logger.info(f"Screening Funnel: Retained {len(kept_ligands)}/{len(best_scores)} ligands. Generated {len(whitelist)} MM-GBSA tasks.")
    return True

def filter_sdf_by_whitelist(in_sdf: Path, out_sdf: Path, frame_id: str, whitelist_csv: Path):
    """Filters a multi-conformer SDF based on a funnel whitelist CSV."""
    if not in_sdf.exists() or not whitelist_csv.exists():
        return
        
    df_wl = pd.read_csv(whitelist_csv)
    # Get allowed titles for THIS specific frame
    allowed_titles = set(df_wl[df_wl["Frame"].astype(str) == str(frame_id)]["Title"])
    
    blocks = []
    with open(in_sdf, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        
    for block in content.split("$$$$"):
        lines = block.strip().splitlines()
        if not lines: continue
        title = lines[0].strip()
        if title in allowed_titles:
            blocks.append(block.strip() + "\n$$$$\n")
            
    with open(out_sdf, "w", encoding="utf-8") as f_out:
        f_out.writelines(blocks)

def parse_prime_csv(csv_path: Path) -> dict:
    """Strict parser for Prime MM-GBSA CSV tables, ignoring (NS) and energy breakdown components."""
    data = {}
    if not csv_path.exists(): return data
    with open(csv_path, "r", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title") or row.get("Title") or row.get("s_m_title")
            if not title or title in data: continue
            props = {}
            
            # 1. Exact match for total dG Bind (strictly ignore (NS) and Coulomb/vdW parts)
            for k, v in row.items():
                if not k or not v: continue
                k_clean = k.strip()
                if k_clean == "r_psp_MMGBSA_dG_Bind" or k_clean == "MMGBSA dG Bind":
                    try: props["MMGBSA"] = float(v); break
                    except ValueError: pass
                    
            # 2. Match Ligand Strain
            for k, v in row.items():
                if not k or not v: continue
                if k.strip() == "r_psp_Lig_Strain_Energy" or k.strip() == "Lig Strain Energy":
                    try: props["Lig_Strain"] = float(v); break
                    except ValueError: pass
                    
            if props: data[title] = props
    return data

def _extract_sdf_block(filepath: Path, target_title: str) -> str:
    """Extracts a specific molecule's 3D SDF block as raw text."""
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

def compute_pose_consensus_rmsd(df, cwd: Path):
    """Calculates in-place heavy-atom RMSD between Glide and independent GNINA dock poses."""
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign
    import numpy as np
    
    dock_export = cwd / "04b_screening" / "export"
    rmsd_list = []
    
    for _, row in df.iterrows():
        if row.get("Is_Native", False):
            rmsd_list.append(np.nan)
            continue
            
        frame_id = str(row["Frame"]).replace("F", "")
        title = row["Title"]
        
        glide_sdf = dock_export / f"glide_dock_f{frame_id}_poses.sdf"
        gnina_sdf = dock_export / f"gnina_dock_f{frame_id}_poses.sdf"
        
        block_g = _extract_sdf_block(glide_sdf, title)
        block_n = _extract_sdf_block(gnina_sdf, title)
        
        if block_g and block_n:
            try:
                mg = Chem.RemoveHs(Chem.MolFromMolBlock(block_g))
                mn = Chem.RemoveHs(Chem.MolFromMolBlock(block_n))
                if mg and mn and mg.GetNumAtoms() == mn.GetNumAtoms():
                    val = rdMolAlign.GetBestRMS(mg, mn)
                    rmsd_list.append(round(val, 2))
                else:
                    rmsd_list.append(np.nan)
            except Exception:
                rmsd_list.append(np.nan)
        else:
            rmsd_list.append(np.nan)
            
    df["Pose_RMSD"] = rmsd_list
    return df

def parse_gnina_sdf(sdf_path: Path, prefix: str = "") -> dict:
    data = {}
    if not sdf_path or not sdf_path.exists(): return data
    with open(sdf_path, "r", errors="ignore") as f:
        content = f.read()
    for entry in content.split("$$$$"):
        lines = entry.strip().splitlines()
        if not lines: continue
        title = lines[0].strip()
        if not title: continue
        props = {}
        for i, line in enumerate(lines):
            line_str = line.strip()
            if line_str.startswith("> <") and line_str.endswith(">"):
                tag_clean = line_str[3:-1].strip().lower()
                if i + 1 < len(lines):
                    try:
                        val = float(lines[i + 1].strip())
                        if tag_clean == "cnnscore": 
                            props[f"{prefix}CNNscore"] = val
                        elif tag_clean == "cnnaffinity": 
                            props[f"{prefix}CNNaffinity"] = val
                        elif tag_clean in ["minimizedaffinity", "affinity"]: 
                            props[f"{prefix}VinaAff"] = val
                    except ValueError: pass
        if title not in data and props: data[title] = props
    return data