"""
Worker script to calculate symmetry-aware heavy-atom RMSD between reference and docked poses.
"""
import argparse
import csv
from pathlib import Path
from schrodinger import structure
from schrodinger.structutils import rmsd

def main():
    parser = argparse.ArgumentParser(description="Calculate Conformer RMSD")
    parser.add_argument("--reference", required=True, help="Path to reference ligand MAE")
    parser.add_argument("--poses", required=True, help="Path to Glide PV or Ligand MAE")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()

    ref_st = structure.StructureReader.read(args.reference)
    
    results = []
    
    with structure.StructureReader(args.poses) as reader:
        for i, st in enumerate(reader):
            # Skip the receptor
            if i == 0 and st.atom_total > 500:
                continue
                
            try:
                conf_eval = rmsd.ConformerRmsd(ref_st, st, asl_expr='NOT atom.element H', in_place=True)
                conf_eval.use_symmetry = True
                rmsd_val = conf_eval.calculate()
                
                gscore = st.property.get('r_i_glide_gscore', '')
                dscore = st.property.get('r_i_glide_docking_score', '')
                title = st.title
                
                results.append({
                    "Title": title,
                    "Pose_Index": i,
                    "GlideScore": gscore,
                    "DockingScore": dscore,
                    "RMSD_Heavy": round(rmsd_val, 3)
                })
            except Exception as e:
                print(f"Error processing pose {i}: {e}")

    keys = ["Title", "Pose_Index", "GlideScore", "DockingScore", "RMSD_Heavy"]
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Successfully calculated RMSD for {len(results)} poses. Saved to {args.output}")

if __name__ == "__main__":
    main()