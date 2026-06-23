"""
Run this on the HOST (where stable-baselines3 is installed) to extract the
VecNormalize statistics as a plain .npz file the container can load without SB3.

Usage:
    python3 extract_vecnorm.py
"""

import pickle, pathlib, numpy as np

PKL  = pathlib.Path(__file__).parent / "reference/trained_policy/vec_normalize_20km_ciric_version1.pkl"
OUT  = pathlib.Path(__file__).parent / "reference/trained_policy/vecnorm_arrays.npz"

with open(PKL, "rb") as f:
    vn = pickle.load(f)

np.savez(
    OUT,
    obs_mean  = vn.obs_rms.mean.astype(np.float32),
    obs_var   = vn.obs_rms.var.astype(np.float32),
    clip_obs  = np.float32(getattr(vn, "clip_obs",  10.0)),
    epsilon   = np.float32(getattr(vn, "epsilon",   1e-8)),
)

print(f"Saved  {OUT}")
print(f"  obs_mean[:4] = {vn.obs_rms.mean[:4]}")
print(f"  obs_var[:4]  = {vn.obs_rms.var[:4]}")
print(f"  clip_obs     = {getattr(vn, 'clip_obs', 10.0)}")
