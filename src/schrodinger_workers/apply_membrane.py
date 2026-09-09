import argparse
import sys
from pathlib import Path
from schrodinger import structure

def main():
    parser = argparse.ArgumentParser(description="Inject Implicit Membrane Properties")
    parser.add_argument("--input", required=True, help="Input prepared receptor MAE")
    parser.add_argument("--info", required=True, help="Input membrane thickness text file")
    parser.add_argument("--output", required=True, help="Output receptor MAE with implicit membrane")
    args = parser.parse_args()

    print("--- Starting Direct Membrane Property Injection ---", flush=True)
    
    try:
        st = structure.StructureReader.read(args.input)
        thickness = 28.0
        
        if Path(args.info).exists():
            with open(args.info, "r") as f:
                thickness = float(f.read().strip())
            print(f"Read thickness from immers (datapar1): {thickness} A", flush=True)
        else:
            print(f"WARNING: info file not found, using default thickness: {thickness} A", flush=True)

        half_t = thickness / 2.0
        # Set Prime membrane slab boundary points along the Z-axis normal
        st.property['r_psp_Memb1_x'] = 0.0
        st.property['r_psp_Memb1_y'] = 0.0
        st.property['r_psp_Memb1_z'] = -half_t

        st.property['r_psp_Memb2_x'] = 0.0
        st.property['r_psp_Memb2_y'] = 0.0
        st.property['r_psp_Memb2_z'] = half_t

        # Descriptive metadata properties
        st.property['r_psp_Prime_Membrane_Thickness'] = thickness
        st.property['r_psp_Prime_Membrane_Buffer'] = 2.5
        
        print(f"SUCCESS: Prime implicit membrane properties written (Z = {-half_t:.2f} to {half_t:.2f} A).", flush=True)
        
        st.write(args.output)
        print(f"Successfully saved to {Path(args.output).name}", flush=True)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()