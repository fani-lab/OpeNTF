import argparse
import torch
import pickle
import numpy as np
import sys
import yaml
import subprocess
import os

sys.path.append('../')
import pkgmgr as opentf

with open('../__config__.yaml', 'r') as file:
        config = yaml.safe_load(file)
torch = opentf.install_import(config['data']['pytorch'], 'torch')

def main():
    parser = argparse.ArgumentParser(description="Compare two .pkl files (teamsvecs assumed)")
    parser.add_argument("-a", required=True, help="Path to first .pkl file")
    parser.add_argument("-b", required=True, help="Path to second .pkl file")
    parser.add_argument("--sort", action="store_true", help="Sort rows by 'skill' key before comparison")
    parser.add_argument("--print-shapes", action="store_true", help="Print detailed shape and content comparison using shape.py")
    args = parser.parse_args()

    def load_pkl(fp): return pickle.load(open(fp, 'rb'))

    def to_dense_tensor(m, is_sorted_np_array=False):
        if m is None: return None
        if m.shape[0] == 0: return torch.empty((0, 1, m.shape[1] if m.ndim > 1 else 0), dtype=torch.uint8)
        if is_sorted_np_array: return torch.from_numpy(m).type(torch.uint8)
        return torch.tensor(m.toarray(), dtype=torch.uint8)

    a, b = load_pkl(args.a), load_pkl(args.b)
    eq = True
    keys = ['skill', 'member', 'loc']

    # Sort data first if requested
    if args.sort:
        print("Sorting by 'skill' key...")
        # Convert skill matrices to dense for sorting
        a_skill_np = a['skill'].toarray()
        b_skill_np = b['skill'].toarray()
        # Get sort indices
        sort_idx_a = np.lexsort(np.transpose(a_skill_np))
        sort_idx_b = np.lexsort(np.transpose(b_skill_np))
        # Sort all matrices using these indices
        for k in keys:
            if a[k] is not None: a[k] = a[k].toarray()[sort_idx_a]
            if b[k] is not None: b[k] = b[k].toarray()[sort_idx_b]

    # Now compare the matrices
    for k in keys:
        va, vb = a[k], b[k]
        if va is None and vb is None: continue
        if va is None or vb is None: eq=False; break
        
        ta, tb = to_dense_tensor(va, args.sort), to_dense_tensor(vb, args.sort)
        eq = torch.equal(ta, tb) if torch.is_tensor(ta) else np.array_equal(ta, tb)
        if not eq: break
    print("\nFiles are equal" if eq else "\nFiles are not equal")

    if args.print_shapes:
        print("\n--- Running shape.py for detailed comparison ---")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        shape_py_path = os.path.join(script_dir, 'print-shapes.py')
        command = [sys.executable, shape_py_path, '-a', args.a, '-b', args.b]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running print-shapes.py: {e}")
        except FileNotFoundError:
            print(f"Error: print-shapes.py not found at {shape_py_path}. Make sure it's in the same directory as compare.py.")

if __name__ == "__main__":
    main()
