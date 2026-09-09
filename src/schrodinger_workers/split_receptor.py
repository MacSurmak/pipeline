"""
Worker script executed via $SCHRODINGER/run python3.
Extracts native ligand and retains specific cofactors across multiple frames.
"""
import argparse
import sys
import os
import subprocess
import tempfile
import logging
import json
from pathlib import Path
from schrodinger import structure
from schrodinger.structutils import analyze, rmsd, transform

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

def main():
    parser = argparse.ArgumentParser(description="Split receptor complex (Multi-frame)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--ligand", required=True)
    parser.add_argument("--cofactors", nargs="*", default=[])
    parser.add_argument("--membrane_type", type=str)
    parser.add_argument("--membrane_topo", type=str)
    parser.add_argument("--membrane_chains", type=str)
    parser.add_argument("--frame_index", type=int, required=True)
    args = parser.parse_args()

    input_file = Path(args.input)
    out_dir = Path(args.outdir)
    
    logging.info(f"Reading structures from: {input_file}")
    
    for _ in [1]:
        i = args.frame_index
        st = structure.StructureReader.read(str(input_file), index=i)
        logging.info(f"--- Processing Frame {i} ---")
        
        if args.membrane_type:
            logging.info(f"Orienting frame {i} via PPM 3.0 (immers)...")
            immers_bin = Path(os.environ.get("SBDD_FRAMEWORK_DIR")) / "software" / "immers"
            res_lib = Path(os.environ.get("SBDD_FRAMEWORK_DIR")) / "software" / "res.lib"
            
            if not immers_bin.exists() or not res_lib.exists():
                logging.error("PPM 3.0 (immers) or res.lib not found in software/ directory.")
                sys.exit(1)
                
            prot_st = st.extract(analyze.evaluate_asl(st, "protein"))
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                prot_pdb = tmpdir / "prot.pdb"
                prot_st.write(str(prot_pdb))
                
                inp_file = tmpdir / "immers.inp"
                with open(inp_file, "w") as f:
                    f.write("2\nno\nprot.pdb\n1\n")
                    f.write(f"{args.membrane_type}\nplanar\n{args.membrane_topo}\n{args.membrane_chains}\n")
                
                os.symlink(res_lib, tmpdir / "res.lib")
                
                with open(inp_file, "r") as f_in, open(tmpdir / "immers.log", "w") as f_out:
                    subprocess.run([str(immers_bin)], stdin=f_in, stdout=f_out, cwd=tmpdir, check=True)
                
                out_pdb = tmpdir / "protout.pdb"
                if not out_pdb.exists():
                    logging.error("immers failed to produce protout.pdb.")
                    sys.exit(1)
                    
                # Read MODEL 1 (protein)
                oriented_prot = structure.StructureReader.read(str(out_pdb), index=1)
                
                # Parse calculated hydrophobic parameters from datapar1
                mem_data = {"thickness": 28.0, "tilt": 0.0, "dg": 0.0}
                datapar_file = tmpdir / "datapar1"
                if datapar_file.exists():
                    try:
                        with open(datapar_file) as df:
                            parts = [p.strip() for p in df.readline().split(";")]
                            mem_data["thickness"] = float(parts[1])
                            mem_data["tilt"] = float(parts[3])
                            mem_data["dg"] = float(parts[5])
                    except Exception as e:
                        logging.warning(f"Could not fully parse datapar1: {e}")

                with open(out_dir / f"membrane_info_f{i}.json", "w") as f_json:
                    json.dump(mem_data, f_json, indent=2)

                with open(out_dir / f"membrane_info_f{i}.txt", "w") as f_info:
                    f_info.write(str(mem_data["thickness"]))

                # Align via C-alpha atoms to get the transformation matrix
                calpha_asl = "atom.ptype ' CA '"
                at_list_orig = analyze.evaluate_asl(prot_st, calpha_asl)
                at_list_orient = analyze.evaluate_asl(oriented_prot, calpha_asl)
                
            if len(at_list_orig) == len(at_list_orient) and len(at_list_orig) >= 3:
                matrix = rmsd.get_super_transformation_matrix(oriented_prot, at_list_orient, prot_st, at_list_orig)
                transform.transform_structure(st, matrix)
                logging.info("Successfully applied PPM 3.0 transformation to the entire complex.")
            else:
                logging.error("CA atoms mismatch between original and immers output.")
                sys.exit(1)

        # 1. Extract Native Ligand (Reference for RMSD and Grid Center)
        ligand_asl = f"res.ptype '{args.ligand}'"
        ligand_atoms = analyze.evaluate_asl(st, ligand_asl)
        
        if not ligand_atoms:
            logging.warning(f"Native ligand '{args.ligand}' not found in frame {i}.")
        else:
            native_lig_st = st.extract(ligand_atoms)
            # Prefix 'ref_' to clearly mark it as the 3D reference crystal pose
            lig_out = out_dir / f"ref_native_f{i}.mae"
            native_lig_st.write(str(lig_out))
            logging.info(f"Saved 3D reference native ligand to {lig_out.name}")

        # 2. Extract Receptor & Cofactors
        cofactors_asl = " OR ".join([f"res.ptype '{c}'" for c in args.cofactors])
        keep_asl = f"(protein) OR ({cofactors_asl})" if cofactors_asl else "(protein)"
            
        keep_atoms = analyze.evaluate_asl(st, keep_asl)
        if not keep_atoms:
            logging.error(f"No protein atoms found in frame {i}!")
            continue
            
        rec_st = st.extract(keep_atoms)
            
        rec_out = out_dir / f"receptor_raw_f{i}.mae"
        rec_st.write(str(rec_out))
        logging.info(f"Saved raw receptor to {rec_out.name}")

if __name__ == "__main__":
    main()