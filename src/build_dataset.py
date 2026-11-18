"""Build candidate feature tables for the train (80) and student_test (10) partitions."""
import sys
import time

import pandas as pd
from joblib import Parallel, delayed

from data_loader import list_trips, load_trip
from features import candidate_table

OUT = __import__("pathlib").Path(__file__).resolve().parents[1] / "results"


def one(part, tid):
    df, _ = candidate_table(load_trip(part, tid))
    return df


if __name__ == "__main__":
    for part in sys.argv[1:] or ["train", "student_test"]:
        t0 = time.time()
        dfs = Parallel(n_jobs=-1)(delayed(one)(part, t) for t in list_trips(part))
        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(OUT / f"candidates_{part}.parquet")
        obs = df.groupby(["trip", "obs"]).label.max()
        print(f"{part}: trips={df.trip.nunique()} obs={len(obs)} rows={len(df)} "
              f"candidate_miss_rate={1 - obs.mean():.4f} ({time.time() - t0:.0f}s)")
