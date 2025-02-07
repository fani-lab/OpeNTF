import pickle
import sys
from pathlib import Path
import argparse
import warnings

# Suppress the deprecation warning about sparse matrices
warnings.filterwarnings('ignore', category=DeprecationWarning, 
                       message='Please use.*sparse` namespace')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Display shapes and contents of indexes and teamsvecs data')
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., dblp)')
    parser.add_argument('input_folder', type=str, help='Input folder name (e.g., dblp.v12.json.filtered.mt75.ts3)')
    parser.add_argument('--rows', type=int, default=3, help='Number of rows to display (default: 3)')
    
    args = parser.parse_args()
    
    # Get the directory of the current script
    script_dir = Path(__file__).parent.absolute()
    preprocessed_dir = script_dir.parent / 'data' / 'preprocessed'
    
    # Load pickle files from the input folder
    data_dir = preprocessed_dir / args.dataset / args.input_folder
    
    try:
        with open(data_dir / 'indexes.pkl', 'rb') as f:
            indexes = pickle.load(f)
            
        with open(data_dir / 'teamsvecs.pkl', 'rb') as f:
            teamsvecs = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find {e.filename}")
        return

    # Print shapes and first n_rows items of indexes
    print("=== INDEXES ===")
    if isinstance(indexes, dict):
        for k, v in indexes.items():
            print(f"\n{k}: {type(v)} - {len(v) if hasattr(v, '__len__') else 'N/A'}")
            if isinstance(v, dict):
                print(f"First {args.rows} items:")
                for i, (key, value) in enumerate(v.items()):
                    if i >= args.rows: break
                    print(f"  {key}: {value}")

    # Print shapes and first n_rows items of teamsvecs
    print("\n=== TEAMSVECS ===")
    if isinstance(teamsvecs, dict):
        for k, v in teamsvecs.items():
            if hasattr(v, 'shape'):
                print(f"\n{k}: {type(v)} - shape: {v.shape}")
                print(f"First {args.rows} rows:")
                # Convert to dense array for the first n_rows only
                dense_slice = v[:args.rows].toarray()
                for i, row in enumerate(dense_slice):
                    print(f"  Row {i}: {row}")
            else:
                print(f"\n{k}: {type(v)} - {len(v) if hasattr(v, '__len__') else 'N/A'}")
                if hasattr(v, '__iter__'):
                    print(f"First {args.rows} items:")
                    for i, item in enumerate(v):
                        if i >= args.rows: break
                        print(f"  {item}")

if __name__ == '__main__':
    main()