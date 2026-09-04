"""
Extracts docked ligand poses from Pose Viewer MAE to a multi-conformation SDF.
"""
import argparse
from schrodinger import structure

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pv", required=True, help="Input _pv.maegz file")
    parser.add_argument("--sdf", required=True, help="Output .sdf file")
    args = parser.parse_args()

    with structure.StructureReader(args.pv) as reader, structure.StructureWriter(args.sdf) as writer:
        for i, st in enumerate(reader):
            # Skip the receptor (first record in Pose Viewer files)
            if i == 0 and (st.property.get("b_glide_receptor", 0) == 1 or st.atom_total > 500):
                continue
            writer.append(st)

if __name__ == "__main__":
    main()