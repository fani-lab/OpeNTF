import argparse
import pickle
import torch
import numpy as np
import pandas as pd
import scipy.sparse
import os
import csv

RED = '\033[91m'
RESET = '\033[0m'
NUM_ROWS_TO_DISPLAY = 10
MAX_ELEMENTS_PER_ROW_DISPLAY = 30

def main():
    parser = argparse.ArgumentParser(description="Display shape and content of .pkl files")
    parser.add_argument("-a", required=True, help="Path to first .pkl file")
    parser.add_argument("-b", required=True, help="Path to second .pkl file")
    args = parser.parse_args()
    
    # Load files
    with open(args.a, 'rb') as f_a: a = pickle.load(f_a)
    with open(args.b, 'rb') as f_b: b = pickle.load(f_b)
    
    # Display shape and basic info for first file
    print(f"\nFile A: {args.a}")
    if isinstance(a, dict):
        for key in a:
            if a[key] is None:
                print(f"  Key '{key}': None")
            elif hasattr(a[key], 'shape'):
                print(f"  Key '{key}': Shape {a[key].shape}, Type {type(a[key]).__name__}")
                if hasattr(a[key], 'nnz'):
                    print(f"    Non-zero elements: {a[key].nnz}")
            else:
                print(f"  Key '{key}': Type {type(a[key]).__name__}")
    else:
        print(f"  Type: {type(a).__name__}")
        if hasattr(a, 'shape'):
            print(f"  Shape: {a.shape}")
    
    # Display shape and basic info for second file
    print(f"\nFile B: {args.b}")
    if isinstance(b, dict):
        for key in b:
            if b[key] is None:
                print(f"  Key '{key}': None")
            elif hasattr(b[key], 'shape'):
                print(f"  Key '{key}': Shape {b[key].shape}, Type {type(b[key]).__name__}")
                if hasattr(b[key], 'nnz'):
                    print(f"    Non-zero elements: {b[key].nnz}")
            else:
                print(f"  Key '{key}': Type {type(b[key]).__name__}")
    else:
        print(f"  Type: {type(b).__name__}")
        if hasattr(b, 'shape'):
            print(f"  Shape: {b.shape}")
    
    # Save to CSV (combined format)
    def save_to_combined_csv(data, filename):
        if not isinstance(data, dict):
            print(f"Error: Expected dictionary data for {filename}")
            return
        
        print(f"\nExporting combined CSV for {filename}...")
        
        # Get the keys and convert each to array format
        keys = []
        arrays = {}
        row_count = None
        
        for key in data:
            if data[key] is None:
                print(f"  Skipping '{key}': None")
                continue
                
            keys.append(key)
            
            # Convert to array
            if hasattr(data[key], 'toarray'):
                arrays[key] = data[key].toarray()
            elif isinstance(data[key], torch.Tensor):
                arrays[key] = data[key].cpu().numpy()
            else:
                print(f"  Skipping '{key}': Unsupported type {type(data[key])}")
                continue
                
            # Keep track of row count
            if row_count is None:
                row_count = arrays[key].shape[0]
            elif row_count != arrays[key].shape[0]:
                print(f"  Warning: Key '{key}' has different number of rows: {arrays[key].shape[0]} vs {row_count}")
        
        if not keys:
            print("  No valid keys found for export")
            return
            
        # Create combined CSV with limited rows
        max_rows = 1000
        if row_count > max_rows:
            print(f"  Warning: Data too large, exporting first {max_rows} rows only")
        
        # Write CSV directly to avoid quotes around arrays
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(keys)
            # Write data rows
            for i in range(min(row_count, max_rows)):
                row = []
                for key in keys:
                    row.append(str(arrays[key][i].tolist()))
                writer.writerow(row)
        
        print(f"  Exported {min(row_count, max_rows)} rows to {filename}")
    
    # Export to CSV
    save_to_combined_csv(a, "a_file.csv")
    save_to_combined_csv(b, "b_file.csv")
    
    # Check if shapes match
    print("\nShape comparison:")
    if isinstance(a, dict) and isinstance(b, dict):
        all_shapes_match = True
        for key in a:
            if key not in b:
                print(f"  Key '{key}' exists in file A but not in file B")
                all_shapes_match = False
                continue
                
            if hasattr(a[key], 'shape') and hasattr(b[key], 'shape'):
                if a[key].shape == b[key].shape:
                    print(f"  Key '{key}': Shapes match {a[key].shape}")
                else:
                    print(f"  Key '{key}': Shapes DON'T match: {a[key].shape} vs {b[key].shape}")
                    all_shapes_match = False
        
        for key in b:
            if key not in a:
                print(f"  Key '{key}' exists in file B but not in file A")
                all_shapes_match = False
        
        print(f"\nOverall shapes match: {'Yes' if all_shapes_match else 'No'}")
        print("\nIMPORTANT NOTE: Shape matching only confirms the dimensions are the same.")
        print("The actual values may still differ, which is what compare.py checks.")
        print("If compare.py says 'Files are not equal', then the values differ even if the shapes match.")
        
        # Add a detailed content check for array-like data
        print("\nDetailed Content Comparison (first rows, differences in red):")
        for key in sorted(list(set(a.keys()) | set(b.keys()))): # Iterate over all unique keys from both dicts
            print(f"\n  Key '{key}':")

            in_a = key in a
            in_b = key in b

            if in_a and a[key] is None and in_b and b[key] is None:
                print("    Both values are None.")
                continue
            elif in_a and a[key] is None:
                print(f"    Value in A is None, Value in B: {type(b[key]).__name__ if in_b else 'N/A'}")
                continue
            elif in_b and b[key] is None:
                print(f"    Value in B is None, Value in A: {type(a[key]).__name__ if in_a else 'N/A'}")
                continue
            
            if not in_a:
                print(f"    Only in B. Type: {type(b[key]).__name__}, Shape: {b[key].shape if hasattr(b[key], 'shape') else 'N/A'}")
                continue
            if not in_b:
                print(f"    Only in A. Type: {type(a[key]).__name__}, Shape: {a[key].shape if hasattr(a[key], 'shape') else 'N/A'}")
                continue

            # Attempt to convert to numpy arrays
            a_val = a[key]
            b_val = b[key]
            
            a_array, b_array = None, None

            try:
                if hasattr(a_val, 'toarray'): a_array = a_val.toarray()
                elif isinstance(a_val, torch.Tensor): a_array = a_val.cpu().numpy()
                elif isinstance(a_val, np.ndarray): a_array = a_val
                elif isinstance(a_val, (list, tuple)): a_array = np.array(a_val)
            except Exception as e:
                print(f"    Could not convert A's value to numpy array: {e}")
            
            try:
                if hasattr(b_val, 'toarray'): b_array = b_val.toarray()
                elif isinstance(b_val, torch.Tensor): b_array = b_val.cpu().numpy()
                elif isinstance(b_val, np.ndarray): b_array = b_val
                elif isinstance(b_val, (list, tuple)): b_array = np.array(b_val)
            except Exception as e:
                print(f"    Could not convert B's value to numpy array: {e}")

            if a_array is None or b_array is None:
                if a_array is None and b_array is None:
                    # If neither are array-like, try direct comparison if types seem comparable
                    if type(a_val) == type(b_val) and not hasattr(a_val, 'shape'): # Simple non-array types
                         if a_val == b_val:
                            print(f"    Values are equal (non-array): {str(a_val)[:100]}")
                         else:
                            print(f"    Values differ (non-array):")
                            print(f"      (A): {RED}{str(a_val)[:100]}{RESET}")
                            print(f"      (B): {RED}{str(b_val)[:100]}{RESET}")
                    else:
                        print(f"    One or both values for key '{key}' are not array-like or failed conversion.")
                        print(f"      Type A: {type(a_val).__name__}, Type B: {type(b_val).__name__}")
                elif a_array is None:
                    print(f"    A's value for key '{key}' is not array-like or failed conversion. Type: {type(a_val).__name__}")
                else: # b_array is None
                    print(f"    B's value for key '{key}' is not array-like or failed conversion. Type: {type(b_val).__name__}")
                continue

            if a_array.ndim == 0 or b_array.ndim == 0: # Handle scalar arrays
                print(f"    Scalar values: ")
                val_a_item = a_array.item()
                val_b_item = b_array.item()
                if val_a_item == val_b_item:
                    print(f"      (A): [ {val_a_item} ]")
                    print(f"      (B): [ {val_b_item} ]")
                else:
                    print(f"      (A): [ {RED}{val_a_item}{RESET} ]")
                    print(f"      (B): [ {RED}{val_b_item}{RESET} ]")
                continue
            
            # Ensure 2D for row-wise comparison if 1D
            if a_array.ndim == 1: a_array = np.expand_dims(a_array, axis=0)
            if b_array.ndim == 1: b_array = np.expand_dims(b_array, axis=0)


            if a_array.shape != b_array.shape:
                print(f"    Shapes differ: A is {a_array.shape}, B is {b_array.shape}. Printing separately.")
                print(f"    (A) First {NUM_ROWS_TO_DISPLAY} rows (up to {MAX_ELEMENTS_PER_ROW_DISPLAY} elements):")
                for i in range(min(NUM_ROWS_TO_DISPLAY, a_array.shape[0])):
                    row_str_parts = [str(a_array[i, j]) for j in range(min(MAX_ELEMENTS_PER_ROW_DISPLAY, a_array.shape[1]))]
                    suffix = " ..." if a_array.shape[1] > MAX_ELEMENTS_PER_ROW_DISPLAY else ""
                    print(f"      Row {i}: [ {' '.join(row_str_parts)}{suffix} ]")
                
                print(f"    (B) First {NUM_ROWS_TO_DISPLAY} rows (up to {MAX_ELEMENTS_PER_ROW_DISPLAY} elements):")
                for i in range(min(NUM_ROWS_TO_DISPLAY, b_array.shape[0])):
                    row_str_parts = [str(b_array[i, j]) for j in range(min(MAX_ELEMENTS_PER_ROW_DISPLAY, b_array.shape[1]))]
                    suffix = " ..." if b_array.shape[1] > MAX_ELEMENTS_PER_ROW_DISPLAY else ""
                    print(f"      Row {i}: [ {' '.join(row_str_parts)}{suffix} ]")
                continue

            num_rows_to_show = min(NUM_ROWS_TO_DISPLAY, a_array.shape[0])
            if num_rows_to_show == 0:
                print("    Arrays are empty.")
                continue

            for i in range(num_rows_to_show):
                print(f"    Row {i}:")
                row_a_str_parts = []
                row_b_str_parts = []
                
                num_cols_to_show = min(MAX_ELEMENTS_PER_ROW_DISPLAY, a_array.shape[1])

                for j in range(num_cols_to_show):
                    val_a = a_array[i, j]
                    val_b = b_array[i, j]
                    
                    if hasattr(val_a, 'item'): val_a = val_a.item() # Convert numpy types to native python types
                    if hasattr(val_b, 'item'): val_b = val_b.item()

                    str_val_a = str(val_a)
                    str_val_b = str(val_b)

                    if val_a != val_b:
                        row_a_str_parts.append(f"{RED}{str_val_a}{RESET}")
                        row_b_str_parts.append(f"{RED}{str_val_b}{RESET}")
                    else:
                        row_a_str_parts.append(str_val_a)
                        row_b_str_parts.append(str_val_b)
                
                suffix_a = " ..." if a_array.shape[1] > MAX_ELEMENTS_PER_ROW_DISPLAY else ""
                suffix_b = " ..." if b_array.shape[1] > MAX_ELEMENTS_PER_ROW_DISPLAY else ""
                
                print(f"      (A): [ {' '.join(row_a_str_parts)}{suffix_a} ]")
                print(f"      (B): [ {' '.join(row_b_str_parts)}{suffix_b} ]")
            
    
if __name__ == "__main__":
    main() 