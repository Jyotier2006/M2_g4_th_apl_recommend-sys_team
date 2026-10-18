# MovieLens 1M: source and preparation

Official landing page: https://grouplens.org/datasets/movielens/1m/

Official archive: https://files.grouplens.org/datasets/movielens/ml-1m.zip

The raw data are **not bundled**. The GroupLens README permits research use with acknowledgement but requires separate permission for redistribution. The experiment downloads the official archive and extracts the three data files plus its README to `Data/raw/ml-1m/`. This directory is ignored by Git.

## Download and prepare

After installing the package, from `Code/`:

```bash
python -m matchbox_m2.experiment --output ../Results/new_run
```

For manual download, extract the official archive, retain `ratings.dat`, `users.dat`, `movies.dat` and `README` together, then pass that directory using `--data-root`. A changed checksum stops execution instead of silently accepting a different dataset.

| File | SHA-256 |
|---|---|
| `ml-1m.zip` | `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20` |
| `ratings.dat` | `506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f` |
| `users.dat` | `1dc3a95300cb19f7c10e027daf29b8cdf1d908dab65b5edfb45950aa389a10a3` |
| `movies.dat` | `0140fc2356357c1a851d0f52e893a1e4d3696df632c4141cea8d5bc3d621f0b9` |

## Actual audited contents

- 1,000,209 rating rows; 6,040 users; 3,706 movies with ratings.
- 3,883 metadata movie rows; the largest movie ID is not the count of rated movies.
- Ratings are integers from 1 through 5; there are no repeated user-item pairs.
- `ratings.dat`: `UserID::MovieID::Rating::Timestamp`.
- `users.dat`: `UserID::Gender::Age::Occupation::Zip-code`.
- `movies.dat`: `MovieID::Title::Genres`; titles require Latin-1 decoding.

IDs are mapped to contiguous indices using sorted unique observed IDs. This uses only the vocabulary, not held-out rating values. Metadata is parsed and validated but **not used for M2 training**. Missing ratings are unobserved, not zeros or dislikes.

## Split protocol

For each seed in `[42, 43, 44]`, NumPy's `default_rng(seed).permutation(N)` determines row indices. The first `floor(0.8N)` rows train the model; the next `floor(0.1N)` are validation; the rest are test. This gives 800,167 / 100,020 / 100,022 rows. Every method receives the same rows for a seed. The randomized training order is fixed across EP passes. MF uses separately seeded epoch shuffles.

Split arrays are stored in `Results/reference/split_seed_*.npz`. The arrays are indices into the official file, not copies of raw rating records. Split hashes, sizes and counts of held-out entities absent from training are recorded in `split_audits.json`. Training statistics use only training labels; MF epoch/scale selection uses only validation labels. Test metrics are computed at declared checkpoints, without test-based tuning.

These are interaction splits. They are not temporal splits or a controlled cold-start benchmark. Rows from the same user can occur in multiple splits, as is standard for this initial within-domain rating-prediction protocol.

Please cite F. Maxwell Harper and Joseph A. Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), Article 19, https://doi.org/10.1145/2827872, as requested by GroupLens.
