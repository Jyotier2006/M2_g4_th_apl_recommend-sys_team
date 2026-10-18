"""Official MovieLens 1M download, validation and reproducible split protocol."""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
FILE_HASHES = {
    "ratings.dat": "506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f",
    "users.dat": "1dc3a95300cb19f7c10e027daf29b8cdf1d908dab65b5edfb45950aa389a10a3",
    "movies.dat": "0140fc2356357c1a851d0f52e893a1e4d3696df632c4141cea8d5bc3d621f0b9",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_dataset(root):
    root = Path(root)
    if not all((root / n).exists() for n in FILE_HASHES):
        root.mkdir(parents=True, exist_ok=True)
        archive = root.parent / "ml-1m.zip"
        if not archive.exists():
            with urllib.request.urlopen(URL, timeout=60) as response:
                archive.write_bytes(response.read())
        if sha256(archive) != ZIP_SHA256:
            raise ValueError("MovieLens archive checksum mismatch; inspect the source before proceeding")
        with zipfile.ZipFile(archive) as z:
            for name in [*FILE_HASHES, "README"]:
                (root / name).write_bytes(z.read("ml-1m/" + name))
    for name, expected in FILE_HASHES.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    return root


def load_dataset(root):
    root = ensure_dataset(root)
    rows = np.array([line.strip().split("::") for line in (root / "ratings.dat").read_text().splitlines()], dtype=np.int64)
    if rows.shape != (1000209, 4):
        raise ValueError(f"Unexpected ratings shape {rows.shape}")
    if not np.all((rows[:, 2] >= 1) & (rows[:, 2] <= 5)):
        raise ValueError("Ratings must be integers from 1 to 5")
    uid, iid = np.unique(rows[:, 0]), np.unique(rows[:, 1])
    # The map depends on ID vocabulary only, never on held-out rating values.
    u, i = np.searchsorted(uid, rows[:, 0]), np.searchsorted(iid, rows[:, 1])
    pair_keys = u * len(iid) + i
    if len(np.unique(pair_keys)) != len(rows):
        raise ValueError("Repeated user-item pairs would violate the split assumptions")
    movies = [x.split("::") for x in (root / "movies.dat").read_text(encoding="latin-1").splitlines()]
    people = [x.split("::") for x in (root / "users.dat").read_text().splitlines()]
    stats = dict(ratings=len(rows), users=len(uid), rated_items=len(iid),
                 movie_metadata_rows=len(movies), user_metadata_rows=len(people),
                 rating_counts=np.bincount(rows[:, 2], minlength=6)[1:].tolist(),
                 density=len(rows) / (len(uid) * len(iid)),
                 timestamp_min=int(rows[:, 3].min()), timestamp_max=int(rows[:, 3].max()),
                 duplicate_user_item_pairs=0, hashes=FILE_HASHES,
                 metadata_used_for_training=False)
    return dict(u=u.astype(np.int64), i=i.astype(np.int64),
                r=rows[:, 2].astype(np.int64), uid=uid, iid=iid, stats=stats)


def split_indices(n, seed, train_fraction=0.8, validation_fraction=0.1):
    order = np.random.default_rng(seed).permutation(n)
    ntr, nval = int(n * train_fraction), int(n * validation_fraction)
    return order[:ntr], order[ntr:ntr + nval], order[ntr + nval:]


def split_audit(data, splits, seed):
    train, val, test = splits
    used_u, used_i = np.unique(data["u"][train]), np.unique(data["i"][train])
    result = dict(seed=seed, protocol="random interaction 80/10/10; no chronological or user cold-start claim",
                  split_hashes={}, counts={}, heldout_cold_counts={})
    for name, idx in zip(["train", "validation", "test"], splits):
        result["counts"][name] = len(idx)
        result["split_hashes"][name] = hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest()
        if name != "train":
            result["heldout_cold_counts"][name] = dict(
                user=int((~np.isin(data["u"][idx], used_u)).sum()),
                item=int((~np.isin(data["i"][idx], used_i)).sum()))
    assert len(np.unique(np.concatenate(splits))) == len(data["r"])
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
