#!/usr/bin/env python3
"""
PANDO Job Monitor
-----------------
Submits train.slurm and live-monitors it via squeue until completion.

Usage (on PANDO frontend):
    cd ~/DiffDet4SAR-project/DiffDet4SAR
    python pando/monitor_job.py

Options:
    --submit        Submit a new job (default if no --job-id given)
    --job-id <id>   Monitor an already-running job instead of submitting
    --interval <s>  Polling interval in seconds (default: 10)
"""

import subprocess
import time
import sys
import argparse
import re
from datetime import datetime


# Slurm state codes and their meanings
STATES = {
    "PD": "Pending    (waiting for resources)",
    "R":  "Running    (job is executing)",
    "CG": "Completing (finishing up)",
    "CD": "Completed  (finished successfully)",
    "F":  "Failed     (exit code != 0)",
    "TO": "Timeout    (exceeded time limit)",
    "CA": "Cancelled",
    "NF": "Node Fail",
    "OOM": "Out of Memory",
}

TERMINAL_STATES = {"CD", "F", "TO", "CA", "NF", "OOM"}


def submit_job(slurm_file="pando/train.slurm"):
    print(f"[{now()}] Submitting: sbatch {slurm_file}")
    result = subprocess.run(
        ["sbatch", slurm_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: sbatch failed:\n{result.stderr}")
        sys.exit(1)

    # Extract job ID from "Submitted batch job 12345"
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        print(f"ERROR: Could not parse job ID from: {result.stdout}")
        sys.exit(1)

    job_id = match.group(1)
    print(f"[{now()}] Job submitted: ID = {job_id}")
    return job_id


def query_job(job_id, user="a.jesus"):
    """Returns (state_code, reason, time_used) or None if job not in queue."""
    result = subprocess.run(
        ["squeue", "-u", user, "-j", job_id,
         "--format=%T|%R|%M|%N", "--noheader"],
        capture_output=True, text=True
    )
    line = result.stdout.strip()
    if not line:
        return None  # Job finished / no longer in queue

    parts = line.split("|")
    state  = parts[0] if len(parts) > 0 else "?"
    reason = parts[1] if len(parts) > 1 else ""
    time_  = parts[2] if len(parts) > 2 else ""
    node   = parts[3] if len(parts) > 3 else ""
    return state, reason, time_, node


def check_exit_code(job_id):
    """Use sacct to get the final exit code after job leaves squeue."""
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=JobID,State,ExitCode", "--noheader"],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def now():
    return datetime.now().strftime("%H:%M:%S")


def print_status(job_id, state, reason, time_used, node):
    label = STATES.get(state, state)
    reason_str = f" ({reason})" if reason and reason != "None" else ""
    node_str   = f" on {node}" if node and node != "None" and state == "R" else ""
    print(f"[{now()}] Job {job_id} | {label}{reason_str}{node_str} | elapsed: {time_used}")


def monitor(job_id, user="a.jesus", interval=10):
    print(f"\n[{now()}] Monitoring job {job_id} for user '{user}' (Ctrl+C to stop)\n")

    last_state = None
    try:
        while True:
            info = query_job(job_id, user)

            if info is None:
                # Job left the queue
                print(f"\n[{now()}] Job {job_id} is no longer in squeue.")
                sacct = check_exit_code(job_id)
                if sacct:
                    print(f"\nFinal status (sacct):\n{sacct}")
                else:
                    print("(sacct not available or job info expired)")
                break

            state, reason, time_used, node = info
            if state != last_state:
                print()  # blank line on state change
            print_status(job_id, state, reason, time_used, node)
            last_state = state

            if state in TERMINAL_STATES:
                print(f"\n[{now()}] Job reached terminal state: {state}")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n[{now()}] Monitor stopped by user.")
        print(f"  Job {job_id} may still be running.")
        print(f"  Check with:  squeue -u {user}")
        print(f"  Cancel with: scancel {job_id}")

    print(f"\n  Log file: diffdet4sar_{job_id}.out")
    print(f"  Tail log: tail -f diffdet4sar_{job_id}.out\n")


def main():
    parser = argparse.ArgumentParser(description="Submit and monitor a PANDO Slurm job")
    parser.add_argument("--job-id",   type=str, help="Monitor existing job ID instead of submitting")
    parser.add_argument("--submit",   action="store_true", help="Force submit a new job")
    parser.add_argument("--slurm",    type=str, default="pando/train.slurm", help="Path to slurm file")
    parser.add_argument("--user",     type=str, default="a.jesus", help="PANDO username")
    parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds")
    args = parser.parse_args()

    if args.job_id:
        job_id = args.job_id
        print(f"[{now()}] Attaching to existing job: {job_id}")
    else:
        job_id = submit_job(args.slurm)

    monitor(job_id, user=args.user, interval=args.interval)


if __name__ == "__main__":
    main()
