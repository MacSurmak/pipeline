"""
Worker script executed via $SCHRODINGER/run python3.
Extracts native ligand and retains specific cofactors across multiple frames.
"""
import argparse
import sys
from pathlib import Path
from schrodinger import structure
from schrodinger.structutils import analyze

def main():
    parser = argparse.ArgumentParser(description="Split receptor complex (Multi-frame)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--ligand", required=True)
    parser.add_argument("--cofactors", nargs="*", default=[])
    args = parser.parse_args()

    input_file = Path(args.input)
    out_dir = Path(args.outdir)
    
    print(f"Reading structures from: {input_file}")
    
    for i, st in enumerate(structure.StructureReader(str(input_file)), start=1):
        print(f"--- Processing Frame {i} ---")
        
        # 1. Extract Native Ligand (Reference for RMSD and Grid Center)
        ligand_asl = f"res.ptype '{args.ligand}'"
        ligand_atoms = analyze.evaluate_asl(st, ligand_asl)
        
        if not ligand_atoms:
            print(f"WARNING: Native ligand '{args.ligand}' not found in frame {i}.")
        else:
            native_lig_st = st.extract(ligand_atoms)
            # Prefix 'ref_' to clearly mark it as the 3D reference crystal pose
            lig_out = out_dir / f"ref_native_f{i}.mae"
            native_lig_st.write(str(lig_out))
            print(f"Saved 3D reference native ligand to {lig_out.name}")

        # 2. Extract Receptor & Cofactors
        cofactors_asl = " OR ".join([f"res.ptype '{c}'" for c in args.cofactors])
        keep_asl = f"(protein) OR ({cofactors_asl})" if cofactors_asl else "(protein)"
            
        keep_atoms = analyze.evaluate_asl(st, keep_asl)
        if not keep_atoms:
            print(f"ERROR: No protein atoms found in frame {i}!")
            continue
            
        rec_st = st.extract(keep_atoms)
        rec_out = out_dir / f"receptor_raw_f{i}.mae"
        rec_st.write(str(rec_out))
        print(f"Saved raw receptor to {rec_out.name}")

if __name__ == "__main__":
    main()