import os
import pickle
import argparse
from multiprocessing import Process, Queue
from solver.solver_utils import SOLVER_CLASSES
from utils import get_a_new2


def solve_instance(mode, filepath, log_dir, solver_class, settings):
    """Solve a single instance and return results"""
    solver = solver_class()
    solver.hide_output_to_console()
    solver.load_model(filepath)
    solver.set_aggressive()
    
    # Configure log path
    log_path = os.path.join(log_dir, f'{os.path.basename(filepath)}.log')
    solver.solve(log_file=log_path, data_mode=mode, time_limit=settings['max_time'], threads=settings['threads'], max_solutions=settings['max_solutions'], search_mode=settings['search_mode'])
    
    # Collect solution data
    variables = solver.get_vars()
    var_names = [solver.varname(var) for var in variables]
    solutions, objectives = solver.get_sol_data()
    
    return {
        'var_names': var_names,
        'sols': solutions,
        'objs': objectives
    }

def process_files(mode, queue, input_dir, output_dirs, solver_class, settings):
    """Worker process handler"""
    import time as _time
    pid = os.getpid()
    while True:
        filename = queue.get()
        if filename is None:  # Termination signal
            break

        file_path = os.path.join(input_dir, filename)

        try:
            print(f"[worker {pid}] Processing {filename} (skip solving) ...", flush=True)
            # Skip solving, use empty solution data
            solution_data = {
                'var_names': [],
                'sols': [],
                'objs': []
            }

            # Generate bipartite graph data
            adjacency, var_map, var_nodes, cons_nodes, bin_vars = get_a_new2(file_path)
            bg_data = (adjacency, var_map, var_nodes, cons_nodes, bin_vars)

            # Save results
            base_name = os.path.splitext(filename)[0]
            pickle.dump(solution_data,
                      open(os.path.join(output_dirs['solution'], f'{base_name}.sol'), 'wb'))
            pickle.dump(bg_data,
                      open(os.path.join(output_dirs['BG'], f'{base_name}.bg'), 'wb'))
            print(f"[worker {pid}] Saved {base_name}.bg + {base_name}.sol", flush=True)

        except Exception as e:
            print(f"[worker {pid}] Error processing {filename}: {str(e)}", flush=True)

def prepare_directories(output_root):
    """Prepare output directory structure"""
    dirs = {
        'solution': os.path.join(output_root, 'solution'),
        'logs': os.path.join(output_root, 'logs'),
        'BG': os.path.join(output_root, 'BG'),
        'NBP': os.path.join(output_root, 'NBP'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def get_parser():
    parser = argparse.ArgumentParser(description="collect data for MILP problems")
    
    # Data path parameters
    parser.add_argument('--data_dir', type=str, default=None,
                      help='Base directory for input data. If not set, uses learn2branch-ecole paths.')
    parser.add_argument('--train_instance_dir', type=str,
                      default='/home/lmh/private/learn2branch-ecole/data/l2o_milp',
                      help='Directory containing train instances (default: %(default)s)')
    parser.add_argument('--test_instance_dir', type=str,
                      default='/home/lmh/private/learn2branch-ecole/data/l2o_milp_test',
                      help='Directory containing test instances (default: %(default)s)')
    
    # Task parameters
    parser.add_argument("--problem_type", type=str, nargs='+', default=['SC'],
                      help="Problem type(s) to process, e.g. --problem_type SC CA MVC")
    
    # Parallel processing parameters
    parser.add_argument('--workers', type=int, default=1,
                      help='Number of parallel worker processes (default: CPU count)')
    parser.add_argument('--threads', type=int, default=1,
                      help='Threads per worker process (default: %(default)s)')
    
    # Solver configuration
    parser.add_argument('--solver', choices=SOLVER_CLASSES.keys(), default='gurobi',
                      help='Optimization solver to use (default: %(default)s)')
    parser.add_argument('--max_time', type=int, default=3600,
                      help='Maximum solving time per instance (seconds) (default: %(default)s)')
    parser.add_argument('--max_solutions', type=int, default=50,
                      help='Maximum solutions to store per instance (default: %(default)s)')
   
    parser.add_argument('--mode', type=str, default='train',
                      help='generate data for train/test (default: %(default)s)')
    
    return parser

def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    data_mode = args.mode
    solver_settings = {
        'max_time': args.max_time,
        'max_solutions': args.max_solutions,
        'threads': args.threads,
        'search_mode': 2
    }

    for task in args.problem_type:
        print(f"\n{'='*60}")
        print(f"Processing problem type: {task}")
        print(f"{'='*60}", flush=True)

        # Prepare directory structure
        if args.data_dir is not None:
            input_dir = os.path.join(args.data_dir, data_mode, task)
        elif data_mode == 'test':
            input_dir = os.path.join(args.test_instance_dir, task)
        else:
            input_dir = os.path.join(args.train_instance_dir, task)

        if data_mode == 'test':
            output_dir = os.path.join('./dataset', f"{task}_test")
        else:
            output_dir = os.path.join('./dataset', f"{task}")

        output_dirs = prepare_directories(output_dir)

        # Initialize task queue
        file_queue = Queue()
        existing_files = set(os.listdir(output_dirs['BG']))

        # Add new files to process, skip already processed ones
        n_total = 0
        n_skipped = 0
        for filename in sorted(os.listdir(input_dir)):
            n_total += 1
            base_name = os.path.splitext(filename)[0]
            if f"{base_name}.bg" in existing_files:
                n_skipped += 1
                continue
            file_queue.put(filename)

        n_todo = n_total - n_skipped
        print(f"Found {n_total} instances, {n_skipped} already processed, {n_todo} to process.", flush=True)

        if n_todo == 0:
            print(f"Skipping {task}: all instances already processed.", flush=True)
            continue

        # Add termination signals
        for _ in range(args.workers):
            file_queue.put(None)

        # Start worker processes
        processes = []
        print(f"Starting {args.workers} worker processes...")
        for _ in range(args.workers):
            p = Process(
                target=process_files,
                args=(data_mode, file_queue, input_dir, output_dirs,
                    SOLVER_CLASSES[args.solver], solver_settings)
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        print(f"Problem type {task} completed.", flush=True)

    print("\nAll problem types processed.")

if __name__ == '__main__':
    main()