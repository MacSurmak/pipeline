"""
File: src/modules/admet.py
Description: Evaluates ADMET-AI properties with DrugBank percentiles and traffic-light color coding.
Supports inputs from SMILES (.smi), SDF (.sdf), or top hits from master_hitlist.csv.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from loguru import logger
from rdkit import Chem
from rich.table import Table

from core.logger import console
from core.config_parser import load_config
from core.state_manager import StateManager

# Key pharmacokinetics & toxicity rules: property -> (rule_type, good_thresh, bad_thresh, label)
ADMET_RULES: Dict[str, Dict[str, Any]] = {
    "hERG": {"type": "lower", "good": 0.30, "bad": 0.60, "name": "hERG (Cardiotox)"},
    "AMES": {"type": "lower", "good": 0.30, "bad": 0.60, "name": "AMES (Mutagen)"},
    "DILI": {"type": "lower", "good": 0.40, "bad": 0.70, "name": "DILI (Liver Inj)"},
    "ClinTox": {"type": "lower", "good": 0.20, "bad": 0.50, "name": "Clinical Toxicity"},
    "HIA_Hou": {"type": "higher", "good": 0.70, "bad": 0.40, "name": "HIA (GI Absorpt)"},
    "Bioavailability_Ma": {"type": "higher", "good": 0.60, "bad": 0.40, "name": "Bioavailability"},
    "BBB_Martins": {"type": "higher", "good": 0.60, "bad": 0.40, "name": "BBB Penetration"},
    "CYP3A4_Veith": {"type": "lower", "good": 0.35, "bad": 0.65, "name": "CYP3A4 Inhibit"},
    "CYP2D6_Veith": {"type": "lower", "good": 0.35, "bad": 0.65, "name": "CYP2D6 Inhibit"},
    "Caco2_Wang": {"type": "higher", "good": -5.15, "bad": -5.70, "name": "Caco-2 Perm"},
    "Solubility_AqSolDB": {"type": "higher", "good": -4.0, "bad": -6.0, "name": "LogS (Solubility)"},
    "Lipophilicity_AstraZeneca": {"type": "range", "min_good": 1.0, "max_good": 3.5, "min_bad": 0.0, "max_bad": 5.0, "name": "LogP (Lipophil)"}
}

def _parse_smi_file(path: Path) -> Tuple[List[str], List[str]]:
    """Reads SMILES and names from .smi or .csv."""
    smiles_list, mol_ids = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles = parts[0]
            name = parts[1] if len(parts) > 1 else f"Mol_{idx}"
            smiles_list.append(smiles)
            mol_ids.append(name)
    return smiles_list, mol_ids

def _parse_sdf_file(path: Path, max_mols: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Extracts SMILES and Titles from multi-molecule SDF using RDKit."""
    smiles_list, mol_ids = [], []
    suppl = Chem.SDMolSupplier(str(path), removeHs=True)
    for idx, mol in enumerate(suppl, start=1):
        if mol is None:
            continue
        title = mol.GetProp("_Name") if mol.HasProp("_Name") else f"Mol_{idx}"
        smi = Chem.MolToSmiles(mol)
        smiles_list.append(smi)
        mol_ids.append(title)
        if max_mols and len(smiles_list) >= max_mols:
            break
    return smiles_list, mol_ids

def _load_structures(cwd: Path, mode: str, custom_input: Optional[str], top_n: int) -> Tuple[List[str], List[str]]:
    """Resolves input molecules based on execution mode."""
    if custom_input:
        in_path = Path(custom_input)
        StateManager.require_file(in_path, "admet (custom input)")
        if in_path.suffix.lower() in [".smi", ".txt", ".csv"]:
            return _parse_smi_file(in_path)
        return _parse_sdf_file(in_path, max_mols=top_n if mode == "hits" else None)

    # Mode: hits (default, reads top_n from hitlist)
    hitlist_path = cwd / "06_reporting" / "02_master_hitlist.csv"
    if mode == "hits" and hitlist_path.exists():
        df_hits = pd.read_csv(hitlist_path).head(top_n)
        hit_titles = set(df_hits["Title"])
        export_dir = cwd / "04b_screening" / "export"
        # Find molecules in exported docking SDFs
        found_mols = {}
        for sdf_file in export_dir.glob("*.sdf"):
            suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=True)
            for mol in suppl:
                if mol is None: continue
                name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                if name in hit_titles and name not in found_mols:
                    found_mols[name] = Chem.MolToSmiles(mol)
            if len(found_mols) >= len(hit_titles):
                break
        
        # Maintain original hitlist ranking order
        ordered_smiles, ordered_names = [], []
        for name in df_hits["Title"]:
            if name in found_mols:
                ordered_names.append(name)
                ordered_smiles.append(found_mols[name])
        if ordered_smiles:
            return ordered_smiles, ordered_names

    # Fallback to library input from config
    config = load_config(cwd / "config.yaml")
    lib_file = cwd / "00_input" / config.get("ligand_library", {}).get("input_file", "library.sdf")
    StateManager.require_file(lib_file, "admet (library input)")
    if lib_file.suffix.lower() in [".smi", ".txt", ".csv"]:
        return _parse_smi_file(lib_file)
    return _parse_sdf_file(lib_file, max_mols=top_n if mode == "hits" else None)

def _format_traffic_light(prop: str, val: float, perc: Optional[float] = None) -> str:
    """Formats numeric ADMET values with Rich traffic-light styling."""
    rule = ADMET_RULES.get(prop)
    perc_str = f" ({round(perc)}%)" if perc is not None and pd.notna(perc) else ""
    val_str = f"{val:.2f}"

    if not rule:
        return f"{val_str}{perc_str}"

    r_type = rule["type"]
    color = "white"

    if r_type == "lower":
        if val <= rule["good"]: color = "green"
        elif val >= rule["bad"]: color = "red"
        else: color = "yellow"
    elif r_type == "higher":
        if val >= rule["good"]: color = "green"
        elif val <= rule["bad"]: color = "red"
        else: color = "yellow"
    elif r_type == "range":
        if rule["min_good"] <= val <= rule["max_good"]: color = "green"
        elif val < rule["min_bad"] or val > rule["max_bad"]: color = "red"
        else: color = "yellow"

    return f"[{color}]{val_str}{perc_str}[/{color}]"

def run(cwd: Path, mode: str = "hits", custom_input: Optional[str] = None, top_n: int = 20, force: bool = False):
    try:
        from admet_ai import ADMETModel
    except ImportError:
        logger.critical("Package 'admet_ai' is not installed in the active environment.")
        logger.error("Please run: pip install admet-ai")
        return

    out_dir = cwd / "08_admet"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / f"admet_{mode}_summary.csv"
    transposed_csv = out_dir / f"admet_{mode}_transposed.csv"

    if StateManager.check_output_exists(summary_csv, force):
        return

    smiles_list, mol_ids = _load_structures(cwd, mode, custom_input, top_n)
    if not smiles_list:
        logger.error("No valid structures found for ADMET profiling.")
        return

    logger.info(f"Loaded {len(smiles_list)} molecule(s). Initializing ADMET-AI model inference...")
    model = ADMETModel()
    preds_df = model.predict(smiles=smiles_list).reset_index(drop=True)

    # 1. Export Raw Full Predictions
    preds_df.insert(0, "Title", mol_ids)
    preds_df.to_csv(out_dir / f"admet_{mode}_full_raw.csv", index=False)

    # 2. Build Transposed Representation (val (perc%))
    perc_suffix = "_drugbank_approved_percentile"
    base_props = [c for c in preds_df.columns if not c.endswith(perc_suffix) and f"{c}{perc_suffix}" in preds_df.columns]
    
    transposed_dict = {}
    for idx, name in enumerate(mol_ids):
        row = preds_df.iloc[idx]
        mol_series = {}
        for p in base_props:
            v = row[p]
            pct = row[f"{p}{perc_suffix}"]
            if pd.notna(v) and pd.notna(pct):
                mol_series[p] = f"{round(float(v), 2)} ({round(float(pct))}%)"
            elif pd.notna(v):
                mol_series[p] = f"{round(float(v), 2)}"
            else:
                mol_series[p] = "N/A"
        transposed_dict[name] = mol_series

    df_transposed = pd.DataFrame(transposed_dict)
    df_transposed.index.name = "Property"
    df_transposed.to_csv(transposed_csv)

    # 3. Build Rich Terminal Traffic-Light Table
    table = Table(title=f"ADMET-AI Traffic-Light Profiling (Mode: {mode.upper()})", header_style="bold cyan")
    table.add_column("Compound ID", style="bold white", justify="left")

    display_endpoints = [
        "hERG", "AMES", "DILI", "HIA_Hou", "Bioavailability_Ma", 
        "BBB_Martins", "CYP3A4_Veith", "Caco2_Wang", "Solubility_AqSolDB", "Lipophilicity_AstraZeneca"
    ]

    for ep in display_endpoints:
        col_title = ADMET_RULES[ep]["name"]
        table.add_column(col_title, justify="center")

    summary_rows = []
    for idx, name in enumerate(mol_ids):
        row = preds_df.iloc[idx]
        display_cells = [name]
        clean_row_data = {"Title": name}

        for ep in display_endpoints:
            val = float(row[ep]) if pd.notna(row.get(ep)) else None
            perc = float(row[f"{ep}{perc_suffix}"]) if f"{ep}{perc_suffix}" in row and pd.notna(row[f"{ep}{perc_suffix}"]) else None
            
            if val is not None:
                styled_cell = _format_traffic_light(ep, val, perc)
                display_cells.append(styled_cell)
                clean_row_data[ep] = f"{val:.2f}" + (f" ({round(perc)}%)" if perc is not None else "")
            else:
                display_cells.append("[gray]N/A[/gray]")
                clean_row_data[ep] = "N/A"

        table.add_row(*display_cells)
        summary_rows.append(clean_row_data)

    console.print("")
    console.print(table)
    console.print("")

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    logger.success(f"ADMET profiling complete! Reports saved in: {out_dir.name}/")
    logger.info(f"  - Full table:       {out_dir.name}/admet_{mode}_full_raw.csv")
    logger.info(f"  - Transposed view:  {transposed_csv.name}")
    logger.info(f"  - Summary report:   {summary_csv.name}")