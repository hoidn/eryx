```markdown
# CBF to CrystFEL Geometry Merger (`merge_geom.py`)

## Overview

This Python script (`merge_geom.py`) automates the process of creating a CrystFEL `.geom` file by merging metadata extracted from a Crystallographic Binary Format (CBF) image file header with a base CrystFEL `.geom` template file.

The primary goal is to prioritize experimental parameters found in the CBF header (like wavelength, detector distance, and beam center) when generating the final `.geom` file for CrystFEL processing.

## Features

*   Parses common metadata from CBF image headers, specifically those formatted with lines starting with `# Key Value` or `# Key: Value` (typical for PILATUS detector outputs).
*   Extracts:
    *   Wavelength (and converts to photon energy in eV)
    *   Detector distance
    *   Beam X, Y coordinates (in pixels)
    *   Pixel size (and can calculate `res` or update `pixel_size`)
    *   Count cutoff (for `pixel_saturation_adu`)
*   Merges extracted CBF data with a user-provided `.geom` template file:
    *   **Overrides:** If a parameter (e.g., `photon_energy`, `clen`, `res`, global `beam_center_x/y`) exists in both the CBF header and the template, the CBF value takes precedence.
    *   **Adds:** If a parameter is found in the CBF header but not in the template, it is added to the output `.geom` file.
    *   **Beam Center Prioritization:**
        *   If the CBF header contains `Beam_xy` data, the script will use these values to define global `beam_center_x` and `beam_center_y`.
        *   If the template file contains `panel/corner_x` or `panel/corner_y` definitions (an alternative way to define beam center by panel offset), these lines will be **commented out** in the output `.geom` file to prevent conflicts and ensure the CBF-derived global beam center is used.
    *   **Comments:** The script adds comments to modified or added lines indicating their origin (from CBF data).

## Prerequisites

*   Python 3.x
*   No external Python libraries are required beyond the standard library (`re`, `os`, `argparse`).

## Usage

The script is run from the command line and takes three arguments:

1.  Path to the input `.geom` template file.
2.  Path to the input CBF image file from which to extract header metadata.
3.  Path for the output (newly created or overwritten) merged `.geom` file.

```bash
python merge_geom.py <path_to_template.geom> <path_to_image.cbf> <path_to_output.geom>
```

**Example:**

```bash
python merge_geom.py ./pilatus_base.geom ./my_experiment/image_001.cbf ./my_experiment/processed.geom
```

### Arguments:

*   `<path_to_template.geom>`: This should be a valid CrystFEL geometry file that serves as a base. It might contain detector panel definitions, bad regions, fixed offsets, etc. Values in this template will be overridden by corresponding values from the CBF header if found.
*   `<path_to_image.cbf>`: The CBF image file whose header will be parsed for metadata. The script expects metadata in comment lines (e.g., `# Wavelength 0.9768 A`).
*   `<path_to_output.geom>`: The file path where the resulting merged geometry will be written. If this file already exists, it will be overwritten.

## CBF Header Parsing Details

The script currently looks for the following keys in the CBF header comments (lines starting with `#`):

*   `Wavelength` (expects value in Ångströms, e.g., `# Wavelength 0.9768 A`)
*   `Detector_distance` (expects value in meters, e.g., `# Detector_distance 0.22982 m`)
*   `Beam_xy` (expects value in pixels, e.g., `# Beam_xy (1264.48, 1242.52) pixels`)
*   `Count_cutoff` (expects integer value, e.g., `# Count_cutoff 1009797 counts`)
*   `Pixel_size` (expects value in meters, e.g., `# Pixel_size 172e-6 m x 172e-6 m`)

The parser attempts to handle both colon-separated (`Key: Value`) and space-separated (`Key Value`) formats within these comment lines.

## Output `.geom` File

The output `.geom` file will contain:

*   All original content from the template file.
*   Lines from the template that were overridden by CBF values will be replaced.
*   New lines for parameters found in the CBF but not in the template will be appended.
*   `panel/corner_x` and `panel/corner_y` lines from the template will be commented out if `Beam_xy` was successfully parsed from the CBF.
*   Comments indicating the source of changes or additions (e.g., ` ; From CBF Wavelength ...`).

## Important Considerations

*   **Template File:** The quality and completeness of your input `.geom` template are important. This script primarily focuses on merging scalar experimental parameters. Complex panel layouts should already be defined in your template.
*   **CBF Header Format:** The script is tailored to CBF headers where metadata is stored in comment lines starting with `#`. If your CBF files use a different metadata convention (e.g., only standard CIF tags without the `#` prefix), the `parse_cbf_header` function in the script will need to be modified.
*   **Beam Center:** The script prioritizes the `Beam_xy` from the CBF to define global `beam_center_x` and `beam_center_y`. It actively comments out `panel/corner_x` and `panel/corner_y` from the template to enforce this. Ensure this behavior is desired for your workflow. If your template's `panel/corner_x/y` are meticulously set and intended to be authoritative, you might need to adjust the script's logic.
*   **Error Handling:** The script includes basic error handling for file operations but relies on the CBF header and template having a generally expected structure.

## Customization

*   **Parsing More Parameters:** To extract additional parameters from the CBF header, modify the `parse_cbf_header` function by adding new `elif key_str == "YourNewKey":` blocks and the corresponding regex/parsing logic.
*   **Merging Logic:** To change how parameters are overridden or added, or how conflicts are resolved, modify the main loop and the "Adding parameters" section within the `merge_geom_with_cbf_header` function.
