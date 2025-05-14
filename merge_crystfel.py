import subprocess
import os
import tempfile
import re
import math
import shutil  # Moved to top
import argparse # For CLI arguments

def run_indexamajig(cbf_file, geom_file, output_stream, cell_file=None, crystfel_path="", indexamajig_args=None):
    """
    Runs indexamajig on the provided CBF file.

    Args:
        cbf_file (str): Path to the input CBF image file.
        geom_file (str): Path to the CrystFEL geometry file (.geom).
        output_stream (str): Path for the output .stream file.
        cell_file (str, optional): Path to the unit cell file (.cell or .pdb). Defaults to None.
        crystfel_path (str, optional): Path to CrystFEL executables if not in system PATH. Defaults to "".
        indexamajig_args (list, optional): Additional arguments for indexamajig. Defaults to None.


    Returns:
        bool: True if successful, False otherwise.
    """
    if not os.path.exists(cbf_file):
        print(f"Error: CBF file '{cbf_file}' not found.")
        return False
    if not os.path.exists(geom_file):
        print(f"Error: Geometry file '{geom_file}' not found.")
        return False
    if cell_file and not os.path.exists(cell_file):
        print(f"Error: Cell file '{cell_file}' not found.")
        return False

    input_list_name = ""
    try:
        # indexamajig expects a list of files, even for a single image
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".lst") as tmp_list_file:
            tmp_list_file.write(os.path.abspath(cbf_file) + "\n")
            input_list_name = tmp_list_file.name

        indexamajig_exe_name = "indexamajig"
        if crystfel_path:
            indexamajig_exe = os.path.join(crystfel_path, indexamajig_exe_name)
        else:
            indexamajig_exe = shutil.which(indexamajig_exe_name)

        if not indexamajig_exe:
            print(f"Error: '{indexamajig_exe_name}' not found. Is CrystFEL in your PATH or crystfel_path set correctly?")
            return False

        cmd = [
            indexamajig_exe,
            "-i", input_list_name,
            "-g", os.path.abspath(geom_file),
            "-o", os.path.abspath(output_stream),
        ]

        # Add default or user-provided indexamajig arguments
        if indexamajig_args:
            cmd.extend(indexamajig_args)
        else:
            # Default arguments if none are provided
            cmd.extend([
                "--peaks=zaef",
                "--indexing=dirax", # Good for testing, consider others for real data
                # Example for real data:
                # "--indexing=mosflm,xds,dirax",
                # "--peakfinder=8", # Or another peakfinder like hrooint
                # "--threshold=150",
                # "--min-snr=4.0",
                # "--min-pix-count=3",
            ])


        if cell_file:
            cmd.extend(["-p", os.path.abspath(cell_file)])

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False) # check=False to handle errors manually

        print("\n--- indexamajig STDOUT ---")
        print(result.stdout if result.stdout else "<No STDOUT>")
        print("--- END indexamajig STDOUT ---\n")

        if result.stderr:
            print("--- indexamajig STDERR ---")
            print(result.stderr)
            print("--- END indexamajig STDERR ---\n")

        if result.returncode != 0:
            print(f"Error: indexamajig exited with return code {result.returncode}")
            return False
        
        return True

    except subprocess.CalledProcessError as e: # Should not be reached if check=False
        print(f"Error running indexamajig (CalledProcessError):")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError: # Should be caught by shutil.which check now
        print(f"Error: '{indexamajig_exe_name}' not found. Is CrystFEL in your PATH or crystfel_path set correctly?")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while trying to run indexamajig: {e}")
        return False
    finally:
        if input_list_name and os.path.exists(input_list_name):
            os.remove(input_list_name)

def parse_crystfel_stream(stream_file):
    """
    Parses a CrystFEL .stream file to extract crystal information and reflections.

    Args:
        stream_file (str): Path to the .stream file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a crystal
              and contains its reciprocal lattice vectors and a list of reflections
              with their calculated scattering vectors.
              Example:
              [
                  {
                      'filename': 'event_filename',
                      'event': 'event_id',
                      'astar': [ax, ay, az], # in Å^-1 (Angstrom^-1)
                      'bstar': [bx, by, bz], # in Å^-1
                      'cstar': [cx, cy, cz], # in Å^-1
                      'reflections': [
                          {'h':h, 'k':k, 'l':l, 'q_vec': [qx,qy,qz], 'q_mag': |q| (in Å^-1)}, ...
                      ]
                  }, ...
              ]
    """
    crystals_data = []
    current_crystal = None
    in_reflections_section = False
    reflection_header_found = False

    if not os.path.exists(stream_file):
        print(f"Error: Stream file '{stream_file}' not found for parsing.")
        return crystals_data

    with open(stream_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: # Skip empty lines
                continue

            if line == "----- BEGIN CRYSTAL -----":
                current_crystal = {
                    'astar': None, 'bstar': None, 'cstar': None,
                    'reflections': [], 'filename': 'Unknown', 'event': '',
                    'cell_a': None, 'cell_b': None, 'cell_c': None,
                    'cell_alpha': None, 'cell_beta': None, 'cell_gamma': None,
                    'diffraction_resolution_limit': None
                }
                in_reflections_section = False
                reflection_header_found = False
            elif line == "----- END CRYSTAL -----":
                if current_crystal:
                    crystals_data.append(current_crystal)
                current_crystal = None
                in_reflections_section = False
                reflection_header_found = False
            elif current_crystal:
                if line.startswith("Image filename:"):
                    current_crystal['filename'] = line.split(":", 1)[1].strip()
                elif line.startswith("Event:"):
                     current_crystal['event'] = line.split(":", 1)[1].strip()
                elif line.startswith("Cell parameters:"): # Example: Cell parameters: 78.000 78.000 37.000  90.00 90.00 90.00
                    parts = line.split()
                    try:
                        current_crystal['cell_a'] = float(parts[2])
                        current_crystal['cell_b'] = float(parts[3])
                        current_crystal['cell_c'] = float(parts[4])
                        current_crystal['cell_alpha'] = float(parts[5])
                        current_crystal['cell_beta'] = float(parts[6])
                        current_crystal['cell_gamma'] = float(parts[7])
                    except (IndexError, ValueError) as e:
                        print(f"Warning: Could not parse cell parameters from line: {line} (Error: {e})")
                elif line.startswith("astar ="): # Units are Å^-1
                    parts = line.split()
                    current_crystal['astar'] = [float(parts[2]), float(parts[3]), float(parts[4])]
                elif line.startswith("bstar ="):
                    parts = line.split()
                    current_crystal['bstar'] = [float(parts[2]), float(parts[3]), float(parts[4])]
                elif line.startswith("cstar ="):
                    parts = line.split()
                    current_crystal['cstar'] = [float(parts[2]), float(parts[3]), float(parts[4])]
                elif line.startswith("diffraction_resolution_limit ="):
                    try:
                        current_crystal['diffraction_resolution_limit'] = float(line.split('=')[1].strip().split()[0])
                    except (IndexError, ValueError):
                         print(f"Warning: Could not parse diffraction_resolution_limit from line: {line}")

                # More robust reflection header check
                elif (not reflection_header_found and
                      line.startswith("   h    k    l") or line.startswith("h    k    l")): # Common variations
                    reflection_header_found = True
                    in_reflections_section = True
                elif in_reflections_section and reflection_header_found:
                    # Check if it's a reflection data line (starts with numbers)
                    if re.match(r"^\s*[-+]?\d+\s+[-+]?\d+\s+[-+]?\d+", line):
                        try:
                            parts = line.split()
                            h, k, l = int(parts[0]), int(parts[1]), int(parts[2])

                            if current_crystal['astar'] and current_crystal['bstar'] and current_crystal['cstar']:
                                ax, ay, az = current_crystal['astar']
                                bx, by, bz = current_crystal['bstar']
                                cx, cy, cz = current_crystal['cstar']

                                # Calculate scattering vector q = h*a* + k*b* + l*c*
                                # These a*, b*, c* components are in Å^-1
                                q_x = h * ax + k * bx + l * cx
                                q_y = h * ay + k * by + l * cy
                                q_z = h * az + k * bz + l * cz
                                q_vec = [q_x, q_y, q_z]
                                q_mag = math.sqrt(q_x**2 + q_y**2 + q_z**2) # Magnitude in Å^-1

                                current_crystal['reflections'].append({
                                    'h': h, 'k': k, 'l': l,
                                    'q_vec': q_vec, # in Å^-1
                                    'q_mag': q_mag  # in Å^-1
                                })
                            else:
                                print(f"Warning: Missing reciprocal lattice vectors for crystal in {current_crystal.get('filename', '')} (line {line_num})")
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Could not parse reflection line: '{line}' (Error: {e}) (line {line_num})")
                            # Potentially end of reflections or a misformatted line
                            in_reflections_section = False # Assume end of section on error
                    else:
                        # If line doesn't look like a reflection data line, assume reflections section ended
                        in_reflections_section = False
    return crystals_data

def main():
    parser = argparse.ArgumentParser(description="Run CrystFEL's indexamajig on a CBF image and parse the output stream.")
    parser.add_argument("cbf_file", help="Path to the input CBF image file.")
    parser.add_argument("geom_file", help="Path to the CrystFEL geometry file (.geom).")
    parser.add_argument("output_stream", help="Path for the output .stream file from indexamajig.")
    parser.add_argument("-p", "--cell_file", help="Path to the unit cell file (.cell or .pdb). Optional.", default=None)
    parser.add_argument("--crystfel_path", help="Path to CrystFEL executables if not in system PATH. Optional.", default="")
    parser.add_argument("--extra_args", nargs='*', help="Additional arguments to pass to indexamajig (e.g., --peaks=...) Optional. Use as --extra_args --peaks=myfinder --threshold=100", default=None)


    args = parser.parse_args()

    # 1. Run indexamajig
    print(f"Attempting to index {args.cbf_file}...")
    print(f"  Geometry file: {args.geom_file}")
    if args.cell_file:
        print(f"  Unit cell file: {args.cell_file}")
    if args.extra_args:
        print(f"  Extra indexamajig args: {' '.join(args.extra_args)}")


    success = run_indexamajig(args.cbf_file, args.geom_file, args.output_stream,
                              args.cell_file, args.crystfel_path, args.extra_args)

    if not success:
        print("indexamajig run failed or encountered an error. Exiting.")
        return

    if not os.path.exists(args.output_stream) or os.path.getsize(args.output_stream) == 0:
        print(f"Output stream file '{args.output_stream}' was not created or is empty. "
              "Indexing might have failed or produced no hits.")
        return

    # 2. Parse the stream and calculate scattering vectors
    print(f"\nParsing stream file {args.output_stream}...")
    indexed_crystals = parse_crystfel_stream(args.output_stream)

    if not indexed_crystals:
        print("No crystals found in the stream file.")
    else:
        print(f"\nFound {len(indexed_crystals)} indexed crystal(s).")
        for i, crystal_data in enumerate(indexed_crystals):
            print(f"\n--- Crystal {i+1} from {crystal_data['filename']} (Event: {crystal_data['event']}) ---")
            if crystal_data['cell_a']:
                print(f"  Cell: a={crystal_data['cell_a']:.3f} Å, b={crystal_data['cell_b']:.3f} Å, c={crystal_data['cell_c']:.3f} Å")
                print(f"        α={crystal_data['cell_alpha']:.2f}°, β={crystal_data['cell_beta']:.2f}°, γ={crystal_data['cell_gamma']:.2f}°")
            if crystal_data['diffraction_resolution_limit']:
                 print(f"  Diffraction resolution limit: {crystal_data['diffraction_resolution_limit']:.3f} Å")
            if crystal_data['astar']: # Check if astar was successfully parsed
                print(f"  a*: {['{:.6f}'.format(x) for x in crystal_data['astar']]} Å⁻¹")
                print(f"  b*: {['{:.6f}'.format(x) for x in crystal_data['bstar']]} Å⁻¹")
                print(f"  c*: {['{:.6f}'.format(x) for x in crystal_data['cstar']]} Å⁻¹")
            else:
                print("  Reciprocal lattice vectors not found in stream for this crystal.")

            print(f"  Found {len(crystal_data['reflections'])} reflections.")
            if crystal_data['reflections']:
                print("  h   k   l   |q| (Å⁻¹)     q_x (Å⁻¹)     q_y (Å⁻¹)     q_z (Å⁻¹)")
                for refl_idx, refl in enumerate(crystal_data['reflections']):
                    if refl_idx < 5 or refl_idx > len(crystal_data['reflections']) - 6 : # Print first 5 and last 5
                        q_vec = refl['q_vec']
                        print(f"  {refl['h']:3d} {refl['k']:3d} {refl['l']:3d}   "
                              f"{refl['q_mag']:.4e}   " # Use scientific notation for better precision view
                              f"{q_vec[0]:.4e}   {q_vec[1]:.4e}   {q_vec[2]:.4e}")
                    elif refl_idx == 5 and len(crystal_data['reflections']) > 10:
                         print("  ... many reflections omitted ...")

    # 3. Cleanup (optional - output_stream is a required CLI arg, so user might want to keep it)
    # if os.path.exists(args.output_stream):
    #     os.remove(args.output_stream)

if __name__ == "__main__":
    main()
