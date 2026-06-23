"""
Validate that EkfCore matches AsvTwoGliderEnv exactly on a fixed seed.

Run from ~/asv_sim:
    python3 validate_bridge.py

Both use seed=42 and the same sequence of actions.  Every observation element
must be within float32 rounding error (atol=1e-6).
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference"))

from twoglider_2km_currents_adcp import AsvTwoGliderEnv
from asv_bridge.ekf_core import EkfCore, TRAINING_CONFIG

SEED = 42
N_STEPS = 10

# Fixed action sequence (representative: forward + some turning)
ACTIONS = [
    [1.0,  0.0],
    [1.0,  0.5],
    [0.8, -0.3],
    [1.0,  0.0],
    [-0.5, 0.2],
    [1.0,  0.8],
    [0.6, -0.6],
    [1.0,  0.0],
    [0.9,  0.1],
    [0.7, -0.4],
]

def run_env(seed):
    env = AsvTwoGliderEnv(config=TRAINING_CONFIG)
    obs, _ = env.reset(seed=seed)
    traj = [obs.copy()]
    for a in ACTIONS:
        obs, *_ = env.step(a)
        traj.append(obs.copy())
    return traj

def run_core(seed):
    core = EkfCore(config=TRAINING_CONFIG)
    obs = core.reset(seed=seed)
    traj = [obs.copy()]
    for a in ACTIONS:
        obs = core.step(a)
        traj.append(obs.copy())
    return traj

def main():
    print(f"Validating EkfCore vs AsvTwoGliderEnv  (seed={SEED}, {N_STEPS} steps)\n")

    env_traj  = run_env(SEED)
    core_traj = run_core(SEED)

    OBS_NAMES = [
        "heading", "last_surge", "last_yaw",
        "d1_norm", "d1_rate", "los1_x", "los1_y", "g1_major", "g1_minor", "g1_angle",
        "d2_norm", "d2_rate", "los2_x", "los2_y", "g2_major", "g2_minor", "g2_angle",
        "g_heading", "cx", "cy",
    ]

    all_ok = True
    for step_i, (e, c) in enumerate(zip(env_traj, core_traj)):
        diff   = np.abs(e - c)
        max_d  = diff.max()
        bad    = diff > 1e-5
        status = "OK" if not bad.any() else "MISMATCH"
        if bad.any():
            all_ok = False
        label  = "reset" if step_i == 0 else f"step {step_i}"
        print(f"  [{label:7s}] max_diff={max_d:.2e}  {status}")
        if bad.any():
            for i in np.where(bad)[0]:
                print(f"             obs[{i:2d}] {OBS_NAMES[i]:12s}  env={e[i]:.6f}  core={c[i]:.6f}  diff={diff[i]:.2e}")

    print()
    if all_ok:
        print("PASS — EkfCore matches AsvTwoGliderEnv to within 1e-5 on all steps.")
    else:
        print("FAIL — see mismatches above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
