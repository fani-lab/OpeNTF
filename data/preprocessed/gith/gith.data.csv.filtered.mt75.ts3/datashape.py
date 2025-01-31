import pickle
import sys

# Get number of rows from command line argument, default to 10 if not provided
n_rows = 3
if len(sys.argv) > 1:
    try:
        n_rows = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number. Using default of 10.")

# Load the pickle files
with open('indexes.pkl', 'rb') as f:
    indexes = pickle.load(f)

with open('teamsvecs.pkl', 'rb') as f:
    teamsvecs = pickle.load(f)

# Print shapes and first n_rows items of indexes
print("=== INDEXES ===")
if isinstance(indexes, dict):
    for k, v in indexes.items():
        print(f"\n{k}: {type(v)} - {len(v) if hasattr(v, '__len__') else 'N/A'}")
        if isinstance(v, dict):
            print(f"First {n_rows} items:")
            for i, (key, value) in enumerate(v.items()):
                if i >= n_rows: break
                print(f"  {key}: {value}")

# Print shapes and first n_rows items of teamsvecs
print("\n=== TEAMSVECS ===")
if isinstance(teamsvecs, dict):
    for k, v in teamsvecs.items():
        if hasattr(v, 'shape'):
            print(f"\n{k}: {type(v)} - shape: {v.shape}")
            print(f"First {n_rows} rows:")
            # Convert to dense array for the first n_rows only
            dense_slice = v[:n_rows].toarray()
            for i, row in enumerate(dense_slice):
                print(f"  Row {i}: {row}")
        else:
            print(f"\n{k}: {type(v)} - {len(v) if hasattr(v, '__len__') else 'N/A'}")
            if hasattr(v, '__iter__'):
                print(f"First {n_rows} items:")
                for i, item in enumerate(v):
                    if i >= n_rows: break
                    print(f"  {item}")