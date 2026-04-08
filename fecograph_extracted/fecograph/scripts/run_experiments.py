# =============================================================================
# Experiment Runner — Automates Paper Experiments
#
# Modifies config.py, runs server + clients, and saves results to separate
# directories. Covers:
#   - Table IV:  Lambda sensitivity     (λ = 0.01, 0.03, 0.05, 0.07, 0.1, 0.5)
#   - Table V:   Temperature sensitivity (τ = 0.1, 0.3, 0.5, 0.7, 0.9)
#   - Table VI:  SupCon vs SSLCon ablation
#   - Fig. 7:    Label proportion       (10%, 30%, 50%, 70%)
#   - Table II:  FedAvg vs Ditto comparison
#
# Usage:
#   python scripts/run_experiments.py --experiment lambda
#   python scripts/run_experiments.py --experiment temperature
#   python scripts/run_experiments.py --experiment sslcon
#   python scripts/run_experiments.py --experiment label_prop
#   python scripts/run_experiments.py --experiment fedavg
#   python scripts/run_experiments.py --experiment all
# =============================================================================

import os
import sys
import shutil
import subprocess
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "config", "config.py")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPERIMENTS = {
    "lambda": {
        "description": "Table IV: Lambda (λ_ce) sensitivity",
        "param": "LAMBDA_CE",
        "values": [0.01, 0.03, 0.05, 0.07, 0.1, 0.5],
    },
    "temperature": {
        "description": "Table V: Temperature (τ) sensitivity",
        "param": "TEMPERATURE",
        "values": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "sslcon": {
        "description": "Table VI: SupCon vs SSLCon ablation",
        "param": "CONTRASTIVE_MODE",
        "values": ["supcon", "sslcon"],
    },
    "label_prop": {
        "description": "Fig. 7: Label proportion sensitivity",
        "param": "LABEL_PROPORTION",
        "values": [0.1, 0.3, 0.5, 0.7],
    },
    "fedavg": {
        "description": "Table II: FedAvg vs Ditto comparison",
        "param": "FL_SCHEME",
        "values": ["fedavg", "ditto"],
    },
}


def read_config():
    with open(CONFIG_PATH, "r") as f:
        return f.read()


def write_config(content):
    with open(CONFIG_PATH, "w") as f:
        f.write(content)


def set_config_param(original_config, param_name, new_value):
    """Replace a parameter value in config.py content string."""
    import re
    if isinstance(new_value, str):
        replacement = f'{param_name} = "{new_value}"'
        pattern = rf'^{param_name}\s*=\s*["\'].*?["\']'
    elif isinstance(new_value, float):
        replacement = f"{param_name} = {new_value}"
        pattern = rf"^{param_name}\s*=\s*[\d.]+"
    elif isinstance(new_value, int):
        replacement = f"{param_name} = {new_value}"
        pattern = rf"^{param_name}\s*=\s*\d+"
    else:
        replacement = f"{param_name} = {new_value}"
        pattern = rf"^{param_name}\s*=\s*\S+"

    new_config = re.sub(pattern, replacement, original_config, count=1, flags=re.MULTILINE)
    if new_config == original_config:
        print(f"  WARNING: Could not find/replace {param_name} in config.py")
    return new_config


def clear_logs_and_checkpoints():
    for d in ["logs", "checkpoints"]:
        path = os.path.join(BASE_DIR, d)
        if os.path.exists(path):
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                if os.path.isfile(fp):
                    os.remove(fp)


def run_training(timeout_minutes=120):
    """Start server + 2 clients, wait for completion."""
    print("  Starting server...")
    server_proc = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)

    client_procs = []
    for cid in range(2):
        print(f"  Starting client {cid}...")
        p = subprocess.Popen(
            [sys.executable, "client.py", "--client_id", str(cid)],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        client_procs.append(p)
        time.sleep(2)

    # Wait for server to finish
    print(f"  Training in progress (timeout={timeout_minutes}min)...")
    try:
        server_proc.wait(timeout=timeout_minutes * 60)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT! Killing processes...")
        server_proc.kill()
        for p in client_procs:
            p.kill()
        return False

    for p in client_procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()

    return server_proc.returncode == 0


def save_results(experiment_name, value_label, results_dir):
    """Copy logs and checkpoints to results directory."""
    dest = os.path.join(results_dir, f"{experiment_name}_{value_label}")
    os.makedirs(dest, exist_ok=True)

    for src_dir in ["logs", "checkpoints"]:
        src_path = os.path.join(BASE_DIR, src_dir)
        if os.path.exists(src_path):
            dst_path = os.path.join(dest, src_dir)
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)

    print(f"  Results saved to: {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description="FeCoGraph Experiment Runner")
    parser.add_argument("--experiment", required=True,
                        choices=list(EXPERIMENTS.keys()) + ["all"],
                        help="Which experiment to run")
    parser.add_argument("--results_dir", default=os.path.join(BASE_DIR, "results"),
                        help="Base directory for results")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per run in minutes (default: 120)")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    original_config = read_config()

    if args.experiment == "all":
        exp_names = list(EXPERIMENTS.keys())
    else:
        exp_names = [args.experiment]

    total_runs = sum(len(EXPERIMENTS[e]["values"]) for e in exp_names)
    run_count = 0

    for exp_name in exp_names:
        exp = EXPERIMENTS[exp_name]
        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {exp['description']}")
        print(f"Parameter:  {exp['param']}")
        print(f"Values:     {exp['values']}")
        print("=" * 70)

        for value in exp["values"]:
            run_count += 1
            value_label = str(value).replace(".", "p")
            print(f"\n--- Run {run_count}/{total_runs}: {exp['param']} = {value} ---")

            # Modify config
            modified_config = set_config_param(original_config, exp["param"], value)
            write_config(modified_config)
            print(f"  Config updated: {exp['param']} = {value}")

            # Clear previous run data
            clear_logs_and_checkpoints()

            # Run training
            success = run_training(timeout_minutes=args.timeout)
            if success:
                print("  Training completed successfully.")
            else:
                print("  Training FAILED or timed out.")

            # Save results
            save_results(exp_name, value_label, args.results_dir)

    # Restore original config
    write_config(original_config)
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Config restored to original values.")
    print(f"Results saved to: {args.results_dir}")
    print("=" * 70)

    # Print comparison command hints
    print("\nTo compare results, run:")
    for exp_name in exp_names:
        exp = EXPERIMENTS[exp_name]
        csv_args = " ".join(
            f"--csv {os.path.join(args.results_dir, f'{exp_name}_{str(v).replace(chr(46), chr(112))}', 'logs', 'training_history.csv')}"
            for v in exp["values"]
        )
        label_args = "--labels " + " ".join(f'"{exp["param"]}={v}"' for v in exp["values"])
        print(f"\n  python scripts/plot_convergence.py {csv_args} {label_args}")


if __name__ == "__main__":
    main()
