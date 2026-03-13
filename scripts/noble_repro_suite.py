#!/usr/bin/env python3
"""
Run a budget-aware NOBLE reproduction suite and emit per-node artifact bundles.

The script executes a fixed set of train.py variants, retries transient failures,
parses summary metrics from logs, and writes:
  - table CSVs
  - PNG comparison plots
  - spend ledger CSVs
for each Flywheel empirical node.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_FLOAT_KEYS = {
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
}
SUMMARY_INT_KEYS = {"num_steps", "depth", "sequence_len", "eval_sequence_len", "noble_rank", "mlm_mask_token"}
SUMMARY_STR_KEYS = {"objective", "noble_activation", "noble_targets"}
SUMMARY_KEYS = SUMMARY_FLOAT_KEYS | SUMMARY_INT_KEYS | SUMMARY_STR_KEYS | {"mlm_mask_prob"}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    node_key: str
    env: Dict[str, str]
    timeout_seconds: int = 900


def parse_summary(log_text: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for line in log_text.splitlines():
        m = re.match(r"^([a-zA-Z0-9_]+):\s*(.+?)\s*$", line.strip())
        if not m:
            continue
        key, raw_value = m.group(1), m.group(2)
        if key not in SUMMARY_KEYS:
            continue
        if key in SUMMARY_FLOAT_KEYS:
            try:
                out[key] = float(raw_value)
            except ValueError:
                continue
        elif key in SUMMARY_INT_KEYS:
            try:
                out[key] = int(float(raw_value))
            except ValueError:
                continue
        elif key == "mlm_mask_prob":
            try:
                out[key] = float(raw_value)
            except ValueError:
                continue
        else:
            out[key] = raw_value
    return out


def run_once(repo_dir: Path, spec: RunSpec, logs_dir: Path) -> Dict[str, object]:
    env = os.environ.copy()
    env.update(spec.env)
    log_path = logs_dir / f"{spec.run_id}.log"
    started_at = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            ["uv", "run", "train.py"],
            cwd=repo_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=spec.timeout_seconds,
            check=False,
        )
    ended_at = time.time()
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    summary = parse_summary(log_text)
    ok = proc.returncode == 0 and "val_bpb" in summary
    return {
        "run_id": spec.run_id,
        "label": spec.label,
        "node_key": spec.node_key,
        "returncode": proc.returncode,
        "elapsed_wall_seconds": ended_at - started_at,
        "log_path": str(log_path),
        "ok": ok,
        **summary,
    }


def run_with_retries(repo_dir: Path, spec: RunSpec, logs_dir: Path, retries: int) -> Dict[str, object]:
    last_result: Dict[str, object] = {}
    for attempt in range(1, retries + 2):
        result = run_once(repo_dir, spec, logs_dir)
        result["attempt"] = attempt
        last_result = result
        if result["ok"]:
            return result
    return last_result


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_metric(path: Path, frame: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.bar(frame[x_col], frame[y_col], color="#1f77b4")
    plt.title(title)
    plt.ylabel(y_col)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def build_run_specs(train_seconds: int) -> List[RunSpec]:
    common = {
        "AR_TRAIN_SECONDS": str(train_seconds),
    }
    return [
        RunSpec(
            run_id="r001_baseline_causal",
            label="baseline_causal",
            node_key="repro001",
            env={
                **common,
                "AR_OBJECTIVE": "causal",
                "AR_NOBLE_RANK": "0",
                "AR_NOBLE_MLP_FC": "0",
                "AR_NOBLE_MLP_PROJ": "0",
            },
        ),
        RunSpec(
            run_id="r002_noble_causal",
            label="noble_cos_rank32_causal",
            node_key="repro002",
            env={
                **common,
                "AR_OBJECTIVE": "causal",
                "AR_NOBLE_RANK": "32",
                "AR_NOBLE_MLP_FC": "1",
                "AR_NOBLE_MLP_PROJ": "1",
                "AR_NOBLE_ACT": "cos_net",
            },
        ),
        RunSpec(
            run_id="r003_cos",
            label="act_cos_net",
            node_key="repro003",
            env={
                **common,
                "AR_OBJECTIVE": "causal",
                "AR_NOBLE_RANK": "32",
                "AR_NOBLE_MLP_FC": "1",
                "AR_NOBLE_MLP_PROJ": "1",
                "AR_NOBLE_ACT": "cos_net",
            },
        ),
        RunSpec(
            run_id="r003_gelu",
            label="act_gelu_net",
            node_key="repro003",
            env={
                **common,
                "AR_OBJECTIVE": "causal",
                "AR_NOBLE_RANK": "32",
                "AR_NOBLE_MLP_FC": "1",
                "AR_NOBLE_MLP_PROJ": "1",
                "AR_NOBLE_ACT": "gelu_net",
            },
        ),
        RunSpec(
            run_id="r003_silu",
            label="act_silu_net",
            node_key="repro003",
            env={
                **common,
                "AR_OBJECTIVE": "causal",
                "AR_NOBLE_RANK": "32",
                "AR_NOBLE_MLP_FC": "1",
                "AR_NOBLE_MLP_PROJ": "1",
                "AR_NOBLE_ACT": "silu_net",
            },
        ),
        RunSpec(
            run_id="r003_tanh",
            label="act_tanh_net",
            node_key="repro003",
            env={
                **common,
                "AR_OBJECTIVE": "causal",
                "AR_NOBLE_RANK": "32",
                "AR_NOBLE_MLP_FC": "1",
                "AR_NOBLE_MLP_PROJ": "1",
                "AR_NOBLE_ACT": "tanh_net",
            },
        ),
        RunSpec(
            run_id="r004_rank16",
            label="rank_16",
            node_key="repro004",
            env={
                **common,
                "AR_OBJECTIVE": "causal",
                "AR_NOBLE_RANK": "16",
                "AR_NOBLE_MLP_FC": "1",
                "AR_NOBLE_MLP_PROJ": "1",
                "AR_NOBLE_ACT": "cos_net",
            },
        ),
        RunSpec(
            run_id="r004_rank32",
            label="rank_32",
            node_key="repro004",
            env={
                **common,
                "AR_OBJECTIVE": "causal",
                "AR_NOBLE_RANK": "32",
                "AR_NOBLE_MLP_FC": "1",
                "AR_NOBLE_MLP_PROJ": "1",
                "AR_NOBLE_ACT": "cos_net",
            },
        ),
        RunSpec(
            run_id="r004_rank64",
            label="rank_64",
            node_key="repro004",
            env={
                **common,
                "AR_OBJECTIVE": "causal",
                "AR_NOBLE_RANK": "64",
                "AR_NOBLE_MLP_FC": "1",
                "AR_NOBLE_MLP_PROJ": "1",
                "AR_NOBLE_ACT": "cos_net",
            },
        ),
        RunSpec(
            run_id="r005_mlm_baseline",
            label="mlm_baseline",
            node_key="repro005",
            env={
                **common,
                "AR_OBJECTIVE": "mlm",
                "AR_NOBLE_RANK": "0",
                "AR_NOBLE_MLP_FC": "0",
                "AR_NOBLE_MLP_PROJ": "0",
            },
        ),
        RunSpec(
            run_id="r005_mlm_noble",
            label="mlm_noble_cos_rank32",
            node_key="repro005",
            env={
                **common,
                "AR_OBJECTIVE": "mlm",
                "AR_NOBLE_RANK": "32",
                "AR_NOBLE_MLP_FC": "1",
                "AR_NOBLE_MLP_PROJ": "1",
                "AR_NOBLE_ACT": "cos_net",
            },
        ),
    ]


def estimate_costs(df: pd.DataFrame, hourly_usd: float) -> pd.DataFrame:
    df = df.copy()
    df["cost_usd_estimate"] = df["elapsed_wall_seconds"].astype(float) / 3600.0 * hourly_usd
    df["cumulative_cost_usd_estimate"] = df["cost_usd_estimate"].cumsum()
    return df


def subset(df: pd.DataFrame, run_ids: List[str]) -> pd.DataFrame:
    return df[df["run_id"].isin(run_ids)].copy().reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/noble-repro-suite"))
    parser.add_argument("--train-seconds", type=int, default=300)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--hourly-usd", type=float, default=1.10)
    args = parser.parse_args()

    repo_dir = args.repo_dir.resolve()
    artifacts_dir = (repo_dir / args.artifacts_dir).resolve()
    logs_dir = artifacts_dir / "logs"
    safe_mkdir(logs_dir)

    specs = build_run_specs(args.train_seconds)
    results: List[Dict[str, object]] = []

    for spec in specs:
        print(f"[run] {spec.run_id} ({spec.label})")
        result = run_with_retries(repo_dir, spec, logs_dir, retries=args.retries)
        results.append(result)
        print(
            f"  ok={result.get('ok')} returncode={result.get('returncode')} "
            f"val_bpb={result.get('val_bpb')} log={result.get('log_path')}"
        )

    full_df = pd.DataFrame(results)
    full_df = estimate_costs(full_df, args.hourly_usd)
    full_csv = artifacts_dir / "all_runs_metrics.csv"
    safe_mkdir(artifacts_dir)
    full_df.to_csv(full_csv, index=False)

    # Node bundles: each includes metrics table + spend ledger + PNG.
    node_specs = {
        "repro001": {
            "run_ids": ["r001_baseline_causal"],
            "plot_title": "Repro 001: Baseline Causal",
            "plot_x": "label",
        },
        "repro002": {
            "run_ids": ["r001_baseline_causal", "r002_noble_causal"],
            "plot_title": "Repro 002: Baseline vs NOBLE (Causal)",
            "plot_x": "label",
        },
        "repro003": {
            "run_ids": ["r003_cos", "r003_gelu", "r003_silu", "r003_tanh"],
            "plot_title": "Repro 003: Activation Sweep",
            "plot_x": "label",
        },
        "repro004": {
            "run_ids": ["r004_rank16", "r004_rank32", "r004_rank64"],
            "plot_title": "Repro 004: Rank Sweep",
            "plot_x": "noble_rank",
        },
        "repro005": {
            "run_ids": ["r005_mlm_baseline", "r005_mlm_noble"],
            "plot_title": "Repro 005: MLM Baseline vs NOBLE",
            "plot_x": "label",
        },
    }

    for node_key, cfg in node_specs.items():
        node_dir = artifacts_dir / node_key
        safe_mkdir(node_dir)
        node_df = subset(full_df, cfg["run_ids"])
        node_df.to_csv(node_dir / "run_metrics.csv", index=False)

        spend_cols = ["run_id", "label", "elapsed_wall_seconds", "cost_usd_estimate", "cumulative_cost_usd_estimate", "attempt", "ok"]
        spend_df = node_df[spend_cols].copy()
        spend_df.to_csv(node_dir / "spend_ledger.csv", index=False)

        ok_df = node_df[node_df["ok"] == True].copy()  # noqa: E712
        if not ok_df.empty and "val_bpb" in ok_df.columns:
            plot_metric(
                node_dir / "comparison.png",
                ok_df,
                x_col=cfg["plot_x"],
                y_col="val_bpb",
                title=cfg["plot_title"],
            )
        else:
            # Always emit a PNG so preview remains available even if all runs failed.
            plt.figure(figsize=(8, 4.8))
            plt.text(0.5, 0.5, "No successful runs", ha="center", va="center")
            plt.axis("off")
            plt.title(cfg["plot_title"])
            plt.tight_layout()
            plt.savefig(node_dir / "comparison.png", dpi=160)
            plt.close()

    summary = {
        "total_runs": int(len(full_df)),
        "successful_runs": int((full_df["ok"] == True).sum()),  # noqa: E712
        "failed_runs": int((full_df["ok"] != True).sum()),  # noqa: E712
        "estimated_total_cost_usd": float(full_df["cost_usd_estimate"].sum()),
        "estimated_total_wall_seconds": float(full_df["elapsed_wall_seconds"].sum()),
        "artifacts_dir": str(artifacts_dir),
    }
    summary_rows = [{"key": k, "value": v} for k, v in summary.items()]
    write_csv(
        artifacts_dir / "suite_summary.csv",
        summary_rows,
        fieldnames=["key", "value"],
    )

    print("[done] suite_summary.csv")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
