"""Run the prespecified, paired MovieLens 1M experiment and preserve evidence."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numba
import numpy as np
import pandas as pd
import scipy

from .baselines import MatrixFactorization, MeanBaseline
from .data import load_dataset, split_audit, split_indices, write_json
from .metrics import evaluate, reliability_bins
from .model import Matchbox
from .validation import run_checks

ROOT = Path(__file__).resolve().parents[3]


def evaluate_model(model, data, idx):
    start = time.perf_counter()
    p = model.predict_proba(data["u"][idx], data["i"][idx])
    seconds = time.perf_counter() - start
    result = evaluate(data["r"][idx], p)
    result.update(predict_seconds=seconds, predict_microseconds_per_rating=1e6 * seconds / len(idx))
    return result, p


def source_manifest():
    base = ROOT / "Code"
    return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(base.rglob("*")) if p.is_file() and p.suffix in (".py", ".json", ".toml", ".txt")
            and "__pycache__" not in str(p)}


def run(config_path, data_root, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.csv").exists():
        raise FileExistsError("Choose a new output folder to avoid overwriting an earlier experiment")
    config = json.loads(config_path.read_text())
    write_json(output / "config.json", config)
    write_json(output / "source_hashes_at_run.json", source_manifest())
    env = dict(start_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
               platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
               versions={x.__name__: x.__version__ for x in [np, scipy, pd, matplotlib, numba]},
               command=sys.argv, threads={k: os.environ.get(k, "unset") for k in
               ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]},
               timing="Single process, CPU; fit time excludes compilation, data I/O and KPI evaluation. MF includes validation/checkpoint selection and scale calibration. Prediction time is measured separately.")
    write_json(output / "environment.json", env)
    if not run_checks(output / "validation.json"):
        raise RuntimeError("Mathematical validation failed")
    # Warm both numerical kernels so timing excludes first-call JIT compilation.
    tiny = MatrixFactorization(2, 2, k=config["mf"]["k"])
    tiny.fit_epoch(np.array([0, 1]), np.array([0, 1]), np.array([2, 4]), np.arange(2), 0)
    data = load_dataset(data_root)
    write_json(output / "dataset_audit.json", data["stats"])
    rows, traces, calibration, audits = [], [], [], []
    log = open(output / "run.log", "w", buffering=1)

    def note(message):
        print(message, flush=True)
        log.write(message + "\n")

    for seed in config["seeds"]:
        splits = split_indices(len(data["r"]), seed, config["train_fraction"], config["validation_fraction"])
        tr, va, te = splits
        audits.append(split_audit(data, splits, seed))
        np.savez_compressed(output / f"split_seed_{seed}.npz", train=tr, validation=va, test=te)
        u, i, r = data["u"][tr], data["i"][tr], data["r"][tr]
        nu, ni = len(data["uid"]), len(data["iid"])
        order = np.arange(len(tr))  # tr is already a seeded random permutation.
        note(f"seed={seed} train={len(tr)} validation={len(va)} test={len(te)}")

        def record(model, name, k, passes, fit_seconds, extras=None):
            values, p = evaluate_model(model, data, te)
            vr, _ = evaluate_model(model, data, va)
            row = dict(seed=seed, method=name, k=k, passes=passes, train_seconds=fit_seconds,
                       validation_rmse=vr["rmse"], validation_nll=vr["nll"], **values, **(extras or {}))
            rows.append(row)
            for b in reliability_bins(data["r"][te], p):
                calibration.append(dict(seed=seed, method=name, k=k, passes=passes, **b))
            note(f"{name} K={k} pass={passes}: RMSE={row['rmse']:.6f} NLL={row['nll']:.6f} fit={fit_seconds:.3f}s")
            pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)

        for kind in ["global", "user", "item"]:
            start = time.perf_counter()
            model = MeanBaseline(kind, config["mean_prior_strength"]).fit(u, i, r, nu, ni)
            record(model, f"{kind}_mean", 0, 1, time.perf_counter() - start)

        settings = dict(config["mf"])
        epochs = settings.pop("epochs")
        model = MatrixFactorization(nu, ni, seed=seed, **settings)
        start, best_rmse, selected_epoch = time.perf_counter(), np.inf, 0
        best_state = None
        rng = np.random.default_rng(seed + 1000)
        for epoch in range(epochs):
            model.fit_epoch(u, i, r, rng.permutation(len(r)), epoch)
            val_point = np.clip(model.point_predict(data["u"][va], data["i"][va]), 1, 5)
            rmse = float(np.sqrt(np.mean((val_point - data["r"][va]) ** 2)))
            traces.append(dict(seed=seed, method="mf", k=settings["k"], pass_index=epoch + 1,
                               validation_raw_rmse=rmse, cumulative_train_seconds=time.perf_counter() - start))
            if rmse < best_rmse:
                best_rmse, best_state, selected_epoch = rmse, copy.deepcopy(model), epoch + 1
        model = best_state
        model.calibrate(data["u"][va], data["i"][va], data["r"][va])
        elapsed = time.perf_counter() - start
        raw = np.clip(model.point_predict(data["u"][te], data["i"][te]), 1, 5)
        record(model, "mf", settings["k"], selected_epoch, elapsed,
               dict(raw_point_rmse=float(np.sqrt(np.mean((raw - data["r"][te]) ** 2))),
                    raw_point_mae=float(np.mean(np.abs(raw - data["r"][te]))),
                    observation_variance=model.noise_var, site_memory_bytes=0))

        for k in config["latent_dimensions"]:
            # Run ADF separately to measure its actual O((U+I)K) memory path.
            adf = Matchbox(nu, ni, k=k, seed=seed, **config["model"])
            start = time.perf_counter()
            adf.initialize_training(r, "adf")
            adf_diag = adf.fit_pass(u, i, r, order)
            elapsed = time.perf_counter() - start
            record(adf, "adf", k, 1, elapsed, adf_diag)
            model = Matchbox(nu, ni, k=k, seed=seed, **config["model"])
            start = time.perf_counter()
            model.initialize_training(r, "ep")
            elapsed = time.perf_counter() - start
            for pass_index in range(1, config["ep_passes"] + 1):
                # The same fixed order is used on every pass and for ADF.
                before = [model.uh / model.up, model.ih / model.ip]
                start = time.perf_counter()
                diag = model.fit_pass(u, i, r, order)
                elapsed += time.perf_counter() - start
                after = [model.uh / model.up, model.ih / model.ip]
                diag["rms_parameter_mean_change"] = float(np.sqrt(
                    sum(np.square(a - b).sum() for a, b in zip(after, before)) /
                    sum(a.size for a in after)))
                validation, _ = evaluate_model(model, data, va)
                trace = dict(seed=seed, method="ep", k=k, pass_index=pass_index,
                             cumulative_train_seconds=elapsed, **diag,
                             **{"validation_" + key: value for key, value in validation.items()})
                traces.append(trace)
                if pass_index == 1:
                    equality = max(float(np.max(np.abs(a - b))) for a, b in
                                   zip([adf.up, adf.uh, adf.ip, adf.ih], [model.up, model.uh, model.ip, model.ih]))
                    if equality != 0:
                        raise AssertionError("EP first pass differs from ADF")
                    note(f"ADF/EP1 full-data maximum natural-parameter difference={equality}")
                if pass_index in config["report_passes"]:
                    record(model, "ep", k, pass_index, elapsed, diag)
                if seed == config["seeds"][0] and k == max(config["latent_dimensions"]) and pass_index == config["ep_passes"]:
                    model.save(output / "matchbox_seed42_k8_ep5.npz")
                    np.savez_compressed(output / "id_mapping.npz", user_ids=data["uid"], item_ids=data["iid"])
            del model, adf
        pd.DataFrame(traces).to_csv(output / "learning_trace.csv", index=False)
        pd.DataFrame(calibration).to_csv(output / "calibration_bins.csv", index=False)
        write_json(output / "split_audits.json", audits)
    frame = pd.DataFrame(rows)
    keys = ["method", "k", "passes"]
    numeric = [x for x in frame.select_dtypes(include="number").columns if x not in ["seed", "k", "passes"]]
    summary = frame.groupby(keys)[numeric].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)
    env["end_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "environment.json", env)
    note("Completed. Test metrics are descriptive, prespecified checkpoints; no test-based tuning.")
    log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "Code/configs/m2.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Data/raw/ml-1m")
    parser.add_argument("--output", type=Path, default=ROOT / "Results/reproduced")
    args = parser.parse_args()
    run(args.config, args.data_root, args.output)


if __name__ == "__main__":
    main()
