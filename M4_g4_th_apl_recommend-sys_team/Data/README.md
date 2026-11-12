# MovieLens 1M

Source: https://grouplens.org/datasets/movielens/1m/
Official archive: https://files.grouplens.org/datasets/movielens/ml-1m.zip
Dataset documentation/license: https://files.grouplens.org/datasets/movielens/ml-1m-README.txt

Raw data are not bundled. Follow the GroupLens license in the official README.
Archive SHA-256: `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`.

The loader downloads the official archive and verifies
its SHA-256 and all three data-file hashes. If the download fails, extract the
official archive so this folder contains `ml-1m/ratings.dat`, `movies.dat`, and
`users.dat`. Pass that directory through `--data`.

The dataset has 1,000,209 ratings from 6,040 users and 3,706 rated movies.
The movie catalogue contains 3,883 entries. Missing interactions are not zeros.

| File | Fields | Actual model use |
|---|---|---|
| ratings.dat | UserID, MovieID, Rating, Timestamp | IDs map to zero-based arrays `u`, `i`; `r` is the 1–5 target. Time is audited only. |
| movies.dat | MovieID, Title, Genres | Genres become 18 multi-hot columns for graph construction. Title is a display name. |
| users.dat | UserID, Gender, Age, Occupation, Zip-code | Parsed/audited; demographic columns are not predictors. |

Fields are separated by `::`. Movie text uses Latin-1 encoding. Genre labels use
`|` separators. Age and occupation are category codes, not continuous numbers.
The graph is rebuilt from each condition's training ratings, with fixed genre
metadata allowed for held-out movies. Validation/test ratings never build edges.

Reference: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets:
History and Context*. ACM TiiS 5(4), Article 19. https://doi.org/10.1145/2827872
