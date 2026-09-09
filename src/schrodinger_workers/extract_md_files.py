"""
Worker script executed via $SCHRODINGER/run python3.
Extracts receptor and ligand structures, and determines formal charge for antechamber.
"""
import argparse
import sys
from pathlib import Path
from schrodinger import structure

def main():
    parser = argparse.ArgumentParser(description="Extracts files for MD")
    parser.add_argument("--lig_mae", required=True, help="Input docked pose MAE/SDF")
    parser.add_argument("--lig_title", required=True, help="Title of target hit")
    parser.add_argument("--rec_mae", required=True, help="Input prepared receptor MAE")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save receptor PDB
    rec_st = structure.StructureReader.read(args.rec_mae)
    rec_out = out_dir / "receptor.pdb"
    structure.StructureWriter.write(rec_st, str(rec_out))

    # 2. Find ligand and save Mol2/SDF + calculate formal charge
    charge = 0
    lig_found = False
    all_ligs = []
    
    with structure.StructureReader(args.lig_mae) as reader:
        for i, st in enumerate(reader):
            # Skip receptor (first entry in PV files)
            if i == 0 and (st.property.get("b_glide_receptor", 0) == 1 or st.atom_total > 500):
                continue
            all_ligs.append(st)
            title = (st.property.get("s_m_title") or st.title or "").strip()
            if title == args.lig_title.strip() or title.lower() == args.lig_title.strip().lower():
                lig_mol2 = out_dir / "ligand.mol2"
                structure.StructureWriter.write(st, str(lig_mol2))
                charge = st.formal_charge
                lig_found = True
                break

    # Fallback: if only one ligand pose exists in PV file (e.g. redocking), extract it directly
    if not lig_found and len(all_ligs) == 1:
        st = all_ligs[0]
        lig_mol2 = out_dir / "ligand.mol2"
        structure.StructureWriter.write(st, str(lig_mol2))
        charge = st.formal_charge
        lig_found = True

    if not lig_found:
        print(f"ERROR: Ligand '{args.lig_title}' not found in {args.lig_mae}", file=sys.stderr)
        sys.exit(1)

    # 3. Write charge to file for Bash script parsing
    with open(out_dir / "lig_charge.txt", "w") as f:
        f.write(str(charge))
        
    print(f"SUCCESS: Receptor and ligand extracted. Formal charge: {charge}")

if __name__ == "__main__":
    main()