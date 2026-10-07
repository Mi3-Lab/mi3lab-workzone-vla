#!/usr/bin/env python3
"""Reviewer W4: choose the joint estimator's fitting protocol on calibration data
only.  5-fold cross-validation over the 312 calibration videos: fit on 4 folds
(all of them, or a 100-video subset), score on the held-out fold.  Validation is
not touched."""
import glob
import os
import tempfile

import numpy as np

import ablate_joint as A

H = os.path.expanduser
files = sorted(glob.glob(H("~/eval_cache/joint_dump_cal2/*.npz")))
rng = np.random.default_rng(0)
order = rng.permutation(len(files))
folds = np.array_split(order, 5)


def linkdir(idx):
    d = tempfile.mkdtemp(prefix="cv_")
    for i in idx:
        os.symlink(files[i], os.path.join(d, os.path.basename(files[i])))
    return d


res = {"all training folds (~250)": [], "100-video subset": []}
for k in range(5):
    test = folds[k]; train = np.concatenate([folds[j] for j in range(5) if j != k])
    tdir = linkdir(test)
    for name, tr in (("all training folds (~250)", train), ("100-video subset", rng.permutation(train)[:100])):
        a, f1, ap, ins, ex, p, r, fa = A.score(tdir, A.fit(linkdir(tr), {}), {})
        res[name].append((f1, fa, p))
print("5-fold CV on the 312 calibration videos (validation untouched)")
for name, v in res.items():
    v = np.array(v)
    print(f"  {name:28s} macro-F1 {v[:,0].mean():.3f} (sd {v[:,0].std():.3f})   FA/h {v[:,1].mean():5.1f} (sd {v[:,1].std():.1f})   INSIDE prec {v[:,2].mean()*100:.1f}%")
