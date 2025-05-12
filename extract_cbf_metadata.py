#!/usr/bin/env python3

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Attempt to import pycbf and handle potential import errors
try:
    import pycbf
except ImportError:
    print(
        "Error: pycbf module not found. Please ensure it is installed correctly."
    )
    print(
        "You might need to install it, e.g., via pip or from its source."
    )
    sys.exit(1)
except AttributeError as e:
    # This can happen if _pycbf.so is missing or not built correctly
    if "_pycbf" in str(e): # Check if the error message mentions _pycbf
        print(f"Error importing pycbf: {e}")
        print("This might indicate that the underlying C extension (_pycbf) is missing or corrupted.")
        print("Please ensure pycbf is compiled and installed correctly for your environment.")
    else:
        print(f"An unexpected AttributeError occurred during pycbf import: {e}")
    sys.exit(1)


def cbf2str_safe(value_raw):
    """
    Safely convert pycbf output (str or bytes) to str.
    """
    if value_raw is None:
        # This can happen if pycbf's get_value() itself returns None (e.g. for empty/null CIF value)
        return None # Return None to be consistent with pandas NaN handling for missing data
    if pycbf.HAS_SWIG_PYTHON_STRICT_BYTE_CHAR:
        if isinstance(value_raw, bytes):
            try:
                return value_raw.decode("utf-8", errors="replace")
            except UnicodeDecodeError:
                 try:
                     return value_raw.decode("latin-1", errors="replace") # Common fallback
                 except Exception:
                    return repr(value_raw) # Final fallback for other decoding issues
            except Exception:
                 return repr(value_raw) # Fallback for other errors
        return str(value_raw)
    # If not strict byte char, it should already be a Python str
    return str(value_raw)

def is_cbf_error(exception_obj, error_code_str=None):
    """
    Checks if a generic Exception is likely a CBFlib error by inspecting its message.
    If error_code_str is provided, it also checks for that specific code in the message.
    """
    if not isinstance(exception_obj, Exception):
        return False
    msg = str(exception_obj)
    is_cbf = "CBFlib Error(s):" in msg
    if is_cbf and error_code_str:
        return error_code_str in msg
    return is_cbf

def extract_metadata_from_cbf(cbf_file_path, fields_to_extract):
    """
    Extracts specified metadata fields from a single CBF file.

    Args:
        cbf_file_path (str): Path to the CBF file.
        fields_to_extract (list): A list of strings, where each string is
                                  in the format "category.column".

    Returns:
        dict: A dictionary with extracted metadata. Keys are the
              "category.column" strings, values are the extracted values.
              Returns None if the file cannot be read or a field is missing.
    """
    handle = None
    try:
        handle = pycbf.cbf_handle_struct()
        handle.read_file(cbf_file_path.encode(), pycbf.MSG_NODIGEST)
    except Exception as e:
        if is_cbf_error(e):
            print(f"Warning: Could not read CBF file {cbf_file_path}: {e}", file=sys.stderr)
        else:
            print(f"Warning: An unexpected error occurred reading {cbf_file_path}: {e}", file=sys.stderr)
        return None

    metadata = {"filepath": cbf_file_path}

    try:
        try:
            handle.select_datablock(0)
            # print(f"Debug ({os.path.basename(cbf_file_path)}): Selected datablock 0. Name: {cbf2str_safe(handle.datablock_name())}")
        except Exception as db_e:
            if is_cbf_error(db_e, "CBF_NOTFOUND"):
                print(f"Warning: No datablocks found in {cbf_file_path}. Cannot extract fields.", file=sys.stderr)
            else:
                print(f"Warning: Error selecting datablock in {cbf_file_path}: {db_e}", file=sys.stderr)
            for field_key in fields_to_extract: metadata[field_key] = None
            return metadata

        for field_key in fields_to_extract:
            current_value = None
            category_name, column_name = None, None # Initialize
            try:
                category_name, column_name = field_key.split('.')
            except ValueError:
                print(f"Warning: Invalid field format '{field_key}'. Skipping.", file=sys.stderr)
                metadata[field_key] = None
                continue
            
            # --- Start Enhanced Debugging for a specific field ---
            is_debug_field = (field_key == "array_data.header_convention")
            if is_debug_field: print(f"\n--- Debugging field: {field_key} in {os.path.basename(cbf_file_path)} ---")
            # --- End Enhanced Debugging ---

            try:
                # Ensure datablock context (though find_category should use current)
                # handle.select_datablock(0) # Probably not needed here if already selected

                if is_debug_field: print(f"  Attempting find_category('{category_name}')")
                handle.find_category(category_name.encode())
                if is_debug_field: print(f"  SUCCESS: find_category. Current category: {cbf2str_safe(handle.category_name())}")

                if is_debug_field: print(f"  Attempting find_column('{column_name}')")
                handle.find_column(column_name.encode())
                if is_debug_field: print(f"  SUCCESS: find_column. Current column: {cbf2str_safe(handle.column_name())}")

                num_rows = handle.count_rows()
                if is_debug_field: print(f"  Number of rows in category: {num_rows}")

                if num_rows > 0:
                    handle.select_row(0)
                    if is_debug_field: print(f"  SUCCESS: select_row(0). Current row number: {handle.row_number()}")

                    value_type_raw = handle.get_typeofvalue()
                    value_type = cbf2str_safe(value_type_raw)
                    if is_debug_field: print(f"  Type of value: '{value_type}' (raw: {repr(value_type_raw)})")

                    if value_type == "bnry":
                        current_value = "(Binary Data)"
                    elif value_type == "null":
                        current_value = None
                        if is_debug_field: print(f"  Value is CIF null, setting to Python None.")
                    else:
                        value_raw = handle.get_value()
                        current_value = cbf2str_safe(value_raw)
                        if is_debug_field: print(f"  Raw value from get_value(): {repr(value_raw)}")
                        if is_debug_field: print(f"  Processed value (cbf2str_safe): {repr(current_value)}")
                else:
                    current_value = None
                    if is_debug_field: print(f"  No rows in category, value set to None.")
                
                if is_debug_field: print(f"--- End Debugging field: {field_key} ---")

            except Exception as field_e:
                current_value = None # Ensure it's None on error
                if is_debug_field:
                    print(f"  ERROR during extraction of {field_key}: {type(field_e).__name__}: {field_e}")
                    print(f"--- End Debugging field: {field_key} ---")

                if not is_cbf_error(field_e, "CBF_NOTFOUND"):
                    if is_cbf_error(field_e):
                        print(f"Warning: CBF Error extracting field '{field_key}' from {cbf_file_path}: {field_e}", file=sys.stderr)
                    else:
                        print(f"Warning: Unexpected Python error extracting field '{field_key}' from {cbf_file_path}: {field_e}", file=sys.stderr)
            
            metadata[field_key] = current_value

    except Exception as e:
        print(f"Warning: General error during metadata extraction for {cbf_file_path}: {e}", file=sys.stderr)
        for field_key in fields_to_extract:
            if field_key not in metadata:
                metadata[field_key] = None
    
    return metadata

def main():
    parser = argparse.ArgumentParser(
        description="Inspect metadata from CBF files and optionally plot a heatmap."
    )
    parser.add_argument(
        "directory", type=str, help="Directory containing CBF files."
    )
    parser.add_argument(
        "--pattern", type=str, default="*.cbf", help="File pattern for CBF files (e.g., '*.cbf', '*.img')."
    )
    parser.add_argument(
        "--fields",
        type=str,
        required=True,
        help="Comma-separated list of metadata fields to extract, "
             "in 'category.column' format (e.g., 'axis.id,diffrn_source.beam_X').",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Optional path to save the extracted metadata as a CSV file."
    )
    parser.add_argument(
        "--heatmap-x",
        type=str,
        help="Metadata field (category.column) for the heatmap X-axis. Must be numeric or have few unique values."
    )
    parser.add_argument(
        "--heatmap-y",
        type=str,
        help="Metadata field (category.column) for the heatmap Y-axis. Must be numeric or have few unique values."
    )
    parser.add_argument(
        "--heatmap-value",
        type=str,
        help="Optional metadata field (category.column) whose values populate the heatmap. "
             "If not provided, counts of X-Y occurrences will be plotted. Must be numeric if provided."
    )
    parser.add_argument(
        "--heatmap-agg",
        type=str,
        default="mean",
        choices=['mean', 'sum', 'count', 'median', 'std', 'var'],
        help="Aggregation function for heatmap-value (default: mean). 'count' is used if heatmap-value is not specified."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of files to process (for testing)."
    )

    args = parser.parse_args()

    fields_to_extract = [f.strip() for f in args.fields.split(',')]
    if not fields_to_extract or not all(f for f in fields_to_extract if f): # Ensure no empty strings after split
        print("Error: No valid fields specified.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    cbf_files = sorted(glob.glob(os.path.join(args.directory, args.pattern)))

    if not cbf_files:
        print(f"No files found matching pattern '{args.pattern}' in directory '{args.directory}'.", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        print(f"Limiting processing to the first {args.limit} files.")
        cbf_files = cbf_files[:args.limit]

    print(f"Found {len(cbf_files)} files to process.")

    all_metadata = []
    for i, cbf_file in enumerate(cbf_files):
        print(f"Processing file {i+1}/{len(cbf_files)}: {os.path.basename(cbf_file)}", end='\r', flush=True)
        metadata = extract_metadata_from_cbf(cbf_file, fields_to_extract)
        if metadata: # Only append if metadata extraction was successful (not None)
            all_metadata.append(metadata)
    print("\nProcessing complete.                                  ") # Spaces to clear the line

    if not all_metadata:
        print("No metadata could be extracted from any files, or all extractions failed.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_metadata)

    # Identify columns that are explicitly used for numeric heatmap operations
    numeric_heatmap_cols = []
    if args.heatmap_x: numeric_heatmap_cols.append(args.heatmap_x)
    if args.heatmap_y: numeric_heatmap_cols.append(args.heatmap_y)
    if args.heatmap_value: numeric_heatmap_cols.append(args.heatmap_value)
    
    # Convert only these specified heatmap-related columns to numeric if they exist
    for col in numeric_heatmap_cols:
        if col in df.columns:
            if "(Binary Data)" not in df[col].astype(str).unique(): # Check for our placeholder
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Info: Attempted conversion to numeric for column '{col}'.")
            else:
                print(f"Info: Column '{col}' contains '(Binary Data)' and will not be converted to numeric.", file=sys.stderr)
    
    print("\nExtracted Metadata (first 5 rows):")
    print(df.head().to_string()) # .to_string() for better console output of wide dataframes

    if args.output_csv:
        try:
            df.to_csv(args.output_csv, index=False)
            print(f"\nMetadata saved to {args.output_csv}")
        except Exception as e:
            print(f"\nError saving CSV to {args.output_csv}: {e}", file=sys.stderr)

    if args.heatmap_x and args.heatmap_y:
        # Check if heatmap fields are present *after* DataFrame creation
        if args.heatmap_x not in df.columns or args.heatmap_y not in df.columns:
            missing_fields = []
            if args.heatmap_x not in df.columns: missing_fields.append(args.heatmap_x)
            if args.heatmap_y not in df.columns: missing_fields.append(args.heatmap_y)
            print(f"Error: Heatmap X/Y field(s) {', '.join(missing_fields)} not found in extracted DataFrame columns.", file=sys.stderr)
            sys.exit(1)
        if args.heatmap_value and args.heatmap_value not in df.columns:
            print(f"Error: Heatmap value field '{args.heatmap_value}' not found in extracted DataFrame columns.", file=sys.stderr)
            sys.exit(1)
        
        # Create a copy for plotting to avoid modifying the original DataFrame
        df_plot = df.copy()


        df_plot_cleaned = df_plot.dropna(subset=[args.heatmap_x, args.heatmap_y])
        if args.heatmap_value:
            df_plot_cleaned = df_plot_cleaned.dropna(subset=[args.heatmap_value])

        if df_plot_cleaned.empty:
            print("No data available for heatmap after dropping NaNs in specified columns.", file=sys.stderr)
            sys.exit(1)

        try:
            print(f"\nGenerating heatmap for X='{args.heatmap_x}', Y='{args.heatmap_y}'"
                  f"{', Value=' + args.heatmap_value if args.heatmap_value else ', Aggregation=count'}")
            
            if args.heatmap_value:
                pivot_data = pd.pivot_table(
                    df_plot_cleaned,
                    values=args.heatmap_value,
                    index=args.heatmap_y,
                    columns=args.heatmap_x,
                    aggfunc=args.heatmap_agg
                )
                agg_label = f"{args.heatmap_agg} of {args.heatmap_value}"
            else:
                pivot_data = pd.pivot_table(
                    df_plot_cleaned,
                    index=args.heatmap_y,
                    columns=args.heatmap_x,
                    aggfunc='size',
                    fill_value=0
                )
                agg_label = "Count"

            if pivot_data.empty:
                print("Pivot table for heatmap is empty. Check data and field choices.", file=sys.stderr)
                sys.exit(1)
            
            if pivot_data.shape[0] > 50 or pivot_data.shape[1] > 50:
                print(f"Warning: Heatmap dimensions are large ({pivot_data.shape[0]}x{pivot_data.shape[1]}). "
                      "Consider binning continuous data or choosing fields with fewer unique values.", file=sys.stderr)

            plt.figure(figsize=(12, 10))
            sns.heatmap(pivot_data, annot=True, fmt=".1f" if args.heatmap_value else "d", cmap="viridis", cbar_kws={'label': agg_label})
            plt.title(f"Heatmap of {args.heatmap_y} vs {args.heatmap_x}")
            plt.xlabel(args.heatmap_x)
            plt.ylabel(args.heatmap_y)
            plt.tight_layout()
            
            heatmap_filename = "metadata_heatmap.png"
            plt.savefig(heatmap_filename)
            print(f"Heatmap saved to {heatmap_filename}")
            # plt.show() # Comment out or make optional if running in non-interactive environment

        except Exception as e:
            print(f"Error generating heatmap: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            print("Make sure heatmap-x, heatmap-y, and heatmap-value (if used) are appropriate for pivoting and aggregation.")
    elif args.heatmap_x or args.heatmap_y:
        print("Warning: Both --heatmap-x and --heatmap-y must be specified to generate a heatmap.", file=sys.stderr)

if __name__ == "__main__":
    main()
