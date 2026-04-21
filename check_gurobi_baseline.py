import os
import pickle
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Check average objective of Gurobi solutions from preprocessing.")
    parser.add_argument("--sol_dir", type=str, default="./dataset/SC_test/solution",
                        help="Directory containing .sol files")
    args = parser.parse_args()

    sol_dir = args.sol_dir
    if not os.path.exists(sol_dir):
        print(f"Error: {sol_dir} does not exist. Run process_data.py first.")
        return

    sol_files = sorted([f for f in os.listdir(sol_dir) if f.endswith('.sol')])
    if not sol_files:
        print(f"No .sol files found in {sol_dir}")
        return

    best_objs = []
    for f in sol_files:
        data = pickle.load(open(os.path.join(sol_dir, f), 'rb'))
        objs = data['objs']
        best_obj = objs[0]  # best solution (sorted by Gurobi)
        best_objs.append(best_obj)
        print(f"{f}: best_obj = {best_obj:.4f}, n_sols = {len(objs)}")

    best_objs = np.array(best_objs)
    print(f"\n{'='*50}")
    print(f"Total instances: {len(best_objs)}")
    print(f"Average best obj: {best_objs.mean():.4f}")
    print(f"Std:              {best_objs.std():.4f}")
    print(f"Min:              {best_objs.min():.4f}")
    print(f"Max:              {best_objs.max():.4f}")


if __name__ == '__main__':
    main()
