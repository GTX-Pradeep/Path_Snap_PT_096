"""Portable predictor for the final LightGBM model: plain numpy, no LightGBM needed at inference.

Why: LightGBM's text model file can break when Git converts its line endings (Windows), and model
files are tied to library versions. The trees are exported once to a binary .npz of flat arrays
(split feature, threshold, children, leaf values) and evaluated here with numpy, giving the same
probabilities as LightGBM (verified: max abs difference ~1e-15 on the student-test candidates).

Export : python tree_predictor.py ../models/lgbm_tuned.txt ../models/lgbm_tuned_trees.npz
Use    : TreeEnsemble("models/lgbm_tuned_trees.npz").predict_proba(X)  -> P(class 1)
"""
import sys

import numpy as np


def export(booster_txt, out_npz):
    import lightgbm as lgb
    text = open(booster_txt, "rb").read().decode("utf-8").replace("\r\n", "\n")
    d = lgb.Booster(model_str=text).dump_model()
    assert d["objective"].startswith("binary"), d["objective"]
    feat, thr, left, right, leafval, roots = [], [], [], [], [], []

    def add(node):
        """Append node; internal nodes get index >= 0, leaves are stored as -(leaf_id + 1)."""
        if "split_index" in node:
            assert node["decision_type"] == "<=" and node["missing_type"] == "None"
            i = len(feat)
            feat.append(node["split_feature"]); thr.append(node["threshold"]); left.append(0); right.append(0)
            left[i] = add(node["left_child"]); right[i] = add(node["right_child"])
            return i
        leafval.append(node["leaf_value"])
        return -len(leafval)

    for t in d["tree_info"]:
        roots.append(add(t["tree_structure"]))
    np.savez_compressed(out_npz, feature=np.array(feat, np.int32), threshold=np.array(thr, np.float64),
                        left=np.array(left, np.int32), right=np.array(right, np.int32),
                        leaf_value=np.array(leafval, np.float64), roots=np.array(roots, np.int32))
    print(f"exported {len(roots)} trees, {len(feat)} splits, {len(leafval)} leaves -> {out_npz}")


class TreeEnsemble:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.f, self.t, self.l, self.r = z["feature"], z["threshold"], z["left"], z["right"]
        self.v, self.roots = z["leaf_value"], z["roots"]

    def raw_score(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))   # missing_type None: NaN is treated as 0
        rows = np.arange(len(X))
        score = np.zeros(len(X))
        for root in self.roots:
            node = np.full(len(X), root, dtype=np.int64)
            active = node >= 0
            while active.any():
                n = node[active]
                go_left = X[rows[active], self.f[n]] <= self.t[n]
                node[active] = np.where(go_left, self.l[n], self.r[n])
                active = node >= 0
            score += self.v[-node - 1]
        return score

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-self.raw_score(X)))


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
