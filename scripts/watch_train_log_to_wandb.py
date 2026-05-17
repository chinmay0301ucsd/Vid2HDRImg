"""Tail a training log file and stream parsed loss/lr lines to W&B.

Useful when a training run was kicked off without `--report_to wandb` — runs
in a separate process, reads the rolling log file, parses lines of the form

    HH:MM:SS | __main__ | INFO | step N/M  loss=X.XXXX  lr=Y.YYe-ZZ

and calls wandb.log({"train/loss": ..., "train/lr": ...}, step=N) for each.

Usage:
    export WANDB_API_KEY=...
    python scripts/watch_train_log_to_wandb.py \\
        --log /tmp/wan_lora_v0.log \\
        --project wan-vace-lora \\
        --run_name wan_vace_lora_v0_attached
"""

import argparse
import os
import re
import time

import wandb


STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s+loss=([0-9.+-eE]+)\s+MAE?\s*[0-9.+-eE]*\s*st?\s*\|?\s*"
    r".*?lr=([0-9.+-eE]+)"
)
# Simpler fallback for the Wan trainer's exact format:
SIMPLE_RE = re.compile(r"step\s+(\d+)/(\d+)\s+loss=([0-9.+-eE]+)\s+lr=([0-9.+-eE]+)")


def parse_line(line: str):
    m = SIMPLE_RE.search(line)
    if m:
        step, total, loss, lr = m.groups()
        return int(step), int(total), float(loss), float(lr)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log",      required=True, help="Path to the training log file to tail.")
    p.add_argument("--project",  default="wan-vace-lora")
    p.add_argument("--run_name", default=None)
    p.add_argument("--poll_seconds", type=float, default=5.0)
    p.add_argument("--from_start", action="store_true",
                   help="Read the file from the beginning (default: only new lines).")
    args = p.parse_args()

    wandb.init(project=args.project, name=args.run_name, config={"source": args.log})

    last_step_logged = -1
    f = open(args.log, "r")
    if not args.from_start:
        f.seek(0, os.SEEK_END)
    print(f"watching {args.log}  (project={args.project}, run={wandb.run.name})")
    try:
        while True:
            line = f.readline()
            if not line:
                time.sleep(args.poll_seconds)
                continue
            parsed = parse_line(line)
            if parsed is None:
                continue
            step, total, loss, lr = parsed
            if step <= last_step_logged:
                continue
            wandb.log({"train/loss": loss, "train/lr": lr,
                       "train/total_steps": total}, step=step)
            last_step_logged = step
            print(f"  logged step={step:>6d}  loss={loss:.4f}  lr={lr:.2e}")
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        f.close()
        wandb.finish()


if __name__ == "__main__":
    main()
