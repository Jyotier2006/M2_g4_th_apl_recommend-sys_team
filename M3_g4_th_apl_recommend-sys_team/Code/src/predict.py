"""Predict one user/movie rating from a saved M3 checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import bayes
import gmrf


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--mapping',type=Path,required=True)
    ap.add_argument('--user',type=int,required=True)
    ap.add_argument('--movie',type=int,required=True)
    args=ap.parse_args()
    ids=np.load(args.mapping)
    us=np.flatnonzero(ids['user_ids']==args.user)
    it=np.flatnonzero(ids['movie_ids']==args.movie)
    if not len(us) or not len(it):
        ap.error('ID outside this fitted checkpoint vocabulary; choose an ID from id_mapping.npz.')
    raw=np.load(args.model,allow_pickle=False)
    m={k:raw[k].item() if raw[k].ndim==0 else raw[k] for k in raw.files}
    p=bayes.prob(m,us,it)[0] if 'up' in m else gmrf.prob(m,us,it)[0]
    print('Probabilities for stars 1..5:',np.round(p,6))
    print('Expected stars:',float(p@np.arange(1,6)))
    print('Most likely star:',int(np.argmax(p)+1))


if __name__=='__main__':main()
