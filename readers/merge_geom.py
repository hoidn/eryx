import re
import os
import argparse

# Conversion factor for Angstrom to eV (hc/e)
HC_OVER_E_ANGSTROM_EV = 12398.419843320026

def parse_cbf_header(cbf_file_path):
    header_data = {}
    # print(f"DEBUG: Attempting to open and read CBF: {cbf_file_path}") # Keep for deep debugging if needed
    try:
        with open(cbf_file_path, 'r', errors='ignore') as f:
            for line_num, raw_line in enumerate(f):
                if "--CIF-BINARY-FORMAT-SECTION--" in raw_line:
                    # print(f"DEBUG: Reached binary section at line {line_num + 1}.")
                    break

                line = raw_line.strip()

                if line.startswith('#'):
                    content = line[1:].strip()
                    key_str, value_str = "", ""
                    if ':' in content:
                        parts_colon = content.split(':', 1)
                        if len(parts_colon) == 2:
                            potential_key = parts_colon[0].strip()
                            if potential_key in ["Detector", "Threshold_setting", "Gain_setting",
                                                 "Excluded_pixels", "Flat_field", "Trim_file",
                                                 "Image_path", "Ratecorr_lut_directory", "Retrigger_mode"]:
                                key_str, value_str = potential_key, parts_colon[1].strip()
                    if not key_str:
                        parts_space = content.split(None, 1)
                        if len(parts_space) == 2:
                            key_str, value_str = parts_space[0].strip(), parts_space[1].strip()

                    if key_str:
                        if key_str == "Wavelength":
                            match = re.search(r"([0-9.]+)\s*A", value_str, re.IGNORECASE)
                            if match: header_data['wavelength_A'] = float(match.group(1))
                        elif key_str == "Detector_distance":
                            match = re.search(r"([0-9.]+)\s*m", value_str, re.IGNORECASE)
                            if match: header_data['detector_distance_m'] = float(match.group(1))
                        elif key_str == "Beam_xy":
                            match = re.search(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)\s*pixels", value_str, re.IGNORECASE)
                            if match: header_data['beam_xy_pix'] = (float(match.group(1)), float(match.group(2)))
                        elif key_str == "Count_cutoff":
                            match = re.search(r"([0-9]+)\s*counts", value_str, re.IGNORECASE)
                            if match: header_data['count_cutoff'] = int(match.group(1))
                        elif key_str == "Pixel_size":
                            match = re.search(r"([0-9.eE\-]+)\s*m", value_str, re.IGNORECASE)
                            if match: header_data['pixel_size_m'] = float(match.group(1))
    except FileNotFoundError: print(f"Error: CBF file '{cbf_file_path}' not found."); return None
    except Exception as e: print(f"Error parsing CBF file '{cbf_file_path}': {e}"); return None
    return header_data if header_data else None


def merge_geom_with_cbf_header(geom_template_path, cbf_header_data, output_geom_path):
    if not cbf_header_data:
        cbf_header_data = {} 
        print("No CBF header data parsed. Using template mostly as is.")

    geom_lines_read = []
    try:
        with open(geom_template_path, 'r') as f_template:
            geom_lines_read = f_template.readlines()
    except FileNotFoundError: print(f"Error: Geometry template file '{geom_template_path}' not found."); return
    except Exception as e: print(f"Error reading geometry template file '{geom_template_path}': {e}"); return

    output_lines = []
    existing_template_keys = set() # For adding genuinely new keys later
    
    # Populate existing_template_keys first
    for line in geom_lines_read:
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith(';'):
            parts_check = stripped_line.split('=', 1)
            if len(parts_check) == 2:
                existing_template_keys.add(parts_check[0].strip())

    for original_line in geom_lines_read:
        line_to_append = original_line 
        stripped_line = original_line.strip()

        if stripped_line and not stripped_line.startswith(';'):
            parts = stripped_line.split('=', 1)
            if len(parts) == 2:
                key = parts[0].strip()

                # --- Parameter Overriding & Commenting Logic ---
                # Check for panel/corner_x or panel/corner_y first
                if (re.match(r"\S+/corner_x$", key) or re.match(r"\S+/corner_y$", key)) and 'beam_xy_pix' in cbf_header_data:
                    line_to_append = f";{stripped_line} ; Commented out: CBF Beam_xy will be used for global beam center\n"
                    print(f"  Commenting out template line: '{stripped_line}' (using global beam center from CBF)")
                elif key == "photon_energy" and 'wavelength_A' in cbf_header_data:
                    wavelength_A = cbf_header_data['wavelength_A']
                    photon_energy_eV = HC_OVER_E_ANGSTROM_EV / wavelength_A
                    line_to_append = f"photon_energy = {photon_energy_eV:.2f} ; From CBF Wavelength {wavelength_A:.4f} A\n"
                    print(f"  Overriding '{key}' with CBF value (Photon Energy: {photon_energy_eV:.2f} eV)")
                elif key == "clen" and 'detector_distance_m' in cbf_header_data:
                    dist_m = cbf_header_data['detector_distance_m']
                    line_to_append = f"clen = {dist_m:.6f} ; From CBF Detector_distance\n"
                    print(f"  Overriding '{key}' with CBF value (Detector Distance: {dist_m:.6f} m)")
                elif key == "pixel_saturation_adu" and 'count_cutoff' in cbf_header_data:
                    cutoff = cbf_header_data['count_cutoff']
                    line_to_append = f"pixel_saturation_adu = {cutoff} ; From CBF Count_cutoff\n"
                    print(f"  Overriding '{key}' with CBF value (Count Cutoff: {cutoff})")
                elif key == "res" and 'pixel_size_m' in cbf_header_data:
                    pixel_s_m = cbf_header_data['pixel_size_m']
                    if pixel_s_m > 1e-9:
                        res_val = 1.0 / pixel_s_m
                        line_to_append = f"res = {res_val:.1f} ; From CBF Pixel_size {pixel_s_m*1e6:.0f} um\n"
                        print(f"  Overriding '{key}' with CBF value (res: {res_val:.1f})")
                elif key == "pixel_size" and 'pixel_size_m' in cbf_header_data:
                    pixel_s_m = cbf_header_data['pixel_size_m']
                    line_to_append = f"pixel_size = {pixel_s_m:.8f} ; From CBF Pixel_size\n"
                    print(f"  Overriding '{key}' with CBF value (Pixel Size: {pixel_s_m:.8f} m)")
                elif key == "beam_center_x" and 'beam_xy_pix' in cbf_header_data:
                    beam_x_pix = cbf_header_data['beam_xy_pix'][0]
                    line_to_append = f"beam_center_x = {beam_x_pix:.2f} ; From CBF Beam_xy\n"
                    print(f"  Overriding '{key}' with CBF value (Beam X: {beam_x_pix:.2f} pix)")
                elif key == "beam_center_y" and 'beam_xy_pix' in cbf_header_data:
                    beam_y_pix = cbf_header_data['beam_xy_pix'][1]
                    line_to_append = f"beam_center_y = {beam_y_pix:.2f} ; From CBF Beam_xy\n"
                    print(f"  Overriding '{key}' with CBF value (Beam Y: {beam_y_pix:.2f} pix)")
        
        output_lines.append(line_to_append)

    # --- Add parameters from CBF header if they were not in the template at all ---
    if 'wavelength_A' in cbf_header_data and "photon_energy" not in existing_template_keys:
        wavelength_A = cbf_header_data['wavelength_A']
        photon_energy_eV = HC_OVER_E_ANGSTROM_EV / wavelength_A
        output_lines.append(f"photon_energy = {photon_energy_eV:.2f} ; Added from CBF Wavelength {wavelength_A:.4f} A\n")
        print(f"  Adding 'photon_energy' from CBF (Photon Energy: {photon_energy_eV:.2f} eV)")
    if 'detector_distance_m' in cbf_header_data and "clen" not in existing_template_keys:
        dist_m = cbf_header_data['detector_distance_m']
        output_lines.append(f"clen = {dist_m:.6f} ; Added from CBF Detector_distance\n")
        print(f"  Adding 'clen' from CBF (Detector Distance: {dist_m:.6f} m)")
    if 'count_cutoff' in cbf_header_data and "pixel_saturation_adu" not in existing_template_keys:
        cutoff = cbf_header_data['count_cutoff']
        output_lines.append(f"pixel_saturation_adu = {cutoff} ; Added from CBF Count_cutoff\n")
        print(f"  Adding 'pixel_saturation_adu' from CBF (Count Cutoff: {cutoff})")
    if 'pixel_size_m' in cbf_header_data and "res" not in existing_template_keys and "pixel_size" not in existing_template_keys:
        pixel_s_m = cbf_header_data['pixel_size_m']
        if pixel_s_m > 1e-9:
            res_val = 1.0 / pixel_s_m
            output_lines.append(f"res = {res_val:.1f} ; Added from CBF Pixel_size {pixel_s_m*1e6:.0f} um\n")
            print(f"  Adding 'res' from CBF (res: {res_val:.1f})")
    
    if 'beam_xy_pix' in cbf_header_data:
        if "beam_center_x" not in existing_template_keys:
            beam_x_pix = cbf_header_data['beam_xy_pix'][0]
            output_lines.append(f"beam_center_x = {beam_x_pix:.2f} ; Added from CBF Beam_xy\n")
            print(f"  Adding 'beam_center_x' from CBF (Beam X: {beam_x_pix:.2f} pix)")
        if "beam_center_y" not in existing_template_keys:
            beam_y_pix = cbf_header_data['beam_xy_pix'][1]
            output_lines.append(f"beam_center_y = {beam_y_pix:.2f} ; Added from CBF Beam_xy\n")
            print(f"  Adding 'beam_center_y' from CBF (Beam Y: {beam_y_pix:.2f} pix)")

    try:
        with open(output_geom_path, 'w') as f_out:
            f_out.writelines(output_lines)
        print(f"Successfully merged geometry to: {output_geom_path}")
    except Exception as e:
        print(f"Error writing merged geometry file '{output_geom_path}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CBF header information into a CrystFEL .geom template.")
    parser.add_argument("geom_template", help="Path to the input .geom template file.")
    parser.add_argument("cbf_file", help="Path to the input CBF file.")
    parser.add_argument("output_geom", help="Path for the new, merged .geom file.")
    args = parser.parse_args()
    print(f"Parsing CBF header from: {args.cbf_file}")
    cbf_data = parse_cbf_header(args.cbf_file)
    if cbf_data:
        print("Successfully parsed CBF header data:")
        for key, value in cbf_data.items(): print(f"  {key}: {value}")
    else:
        print("Could not parse CBF header. Using template mostly as is.")
    print(f"\nMerging with template '{args.geom_template}' into '{args.output_geom}'...")
    merge_geom_with_cbf_header(args.geom_template, cbf_data, args.output_geom)
