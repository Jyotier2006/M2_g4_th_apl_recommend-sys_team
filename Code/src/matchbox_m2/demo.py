"""Fast synthetic pipeline demo or inference from the supplied real checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np

from .model import Matchbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--item-id", type=int, default=1)
    args = parser.parse_args()
    if args.checkpoint:
        if not args.mapping:
            parser.error("--mapping is required with --checkpoint")
        model = Matchbox.load(args.checkpoint)
        with np.load(args.mapping) as maps:
            users, items = maps["user_ids"], maps["item_ids"]
        if args.user_id not in users or args.item_id not in items:
            parser.error("ID is absent from this checkpoint's vocabulary")
        u, i = int(np.searchsorted(users, args.user_id)), int(np.searchsorted(items, args.item_id))
        context = "MovieLens checkpoint query; this pair is not guaranteed to be held out"
    else:
        model = Matchbox(3, 3, k=2, seed=42)
        utrain, itrain = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 2, 0, 2])
        rtrain = np.array([5, 4, 2, 1, 4, 5])
        model.initialize_training(rtrain)
        for _ in range(3):
            model.fit_pass(utrain, itrain, rtrain, np.arange(6))
        u, i = 0, 2
        context = "Synthetic six-rating smoke demo; not a MovieLens result"
    p = model.predict_proba(np.array([u]), np.array([i]))[0]
    print(json.dumps(dict(context=context, rating_probabilities={str(j+1):float(x) for j,x in enumerate(p)},
                          expected_rating=float(p @ np.arange(1,6)), sum_probabilities=float(p.sum())), indent=2))


if __name__ == "__main__":
    main()
