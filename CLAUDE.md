# Project: ASV two-glider acoustic-localization — sim-to-sim transfer

## Goal
Transfer a trained SAC policy (an ASV that shuttles between two underwater gliders
doing range-only acoustic EKF localization) from an idealized Python Gym env into a
realistic Gazebo + ROS 2 simulation, then domain-randomize fine-tune and MEASURE transfer.
This is a thesis; several months of runway; aiming for the ambitious finish line
(robustness + transfer results, not just a demo).

## Host environment (do NOT change the host OS)
- Ubuntu 22.04, ROS 2 Humble on host (left untouched on purpose).
- The sim stack runs in Docker, NOT natively, to avoid repartitioning.
- Docker = official apt engine (the snap version was removed; it broke GPU passthrough).
- NVIDIA Container Toolkit installed; RTX 2000 Ada (driver 535) verified visible in containers.

## The container (already built + verified)
- Image: `asv_sim:jazzy` — ROS 2 Jazzy desktop + Gazebo Harmonic 8.11 + ros-gz.
- Built from `~/asv_sim/Dockerfile`.
- Launch via `~/asv_sim/run.sh <command>` — handles --gpus, X11, and NVIDIA PRIME
  render offload; mounts `~/asv_sim` -> `/work`.
- VERIFIED WORKING: `gz sim` opens with GPU rendering (OpenGL renderer = NVIDIA, not Intel).
  X11 GUI from container works. Container can git-clone the VRX repo
  (branches `main` and `jazzy` exist and are reachable).

## Plan (current step = 1)
1. Add VRX to the image; get the WAM-V boat floating with working thrusters.  <-- NEXT
2. Build a ROS 2 "bridge" node reproducing the Python env's EKF + observation vector
   + VecNormalize, validated to match the Python env on a fixed seed BEFORE adding physics.
3. Run the trained SAC policy on the boat: kinematic first, then real VRX dynamics
   with a low-level velocity controller tracking [surge_cmd, yaw_rate_cmd].
4. Add the two gliders + acoustic range measurements (slant range + noise, comm-range gated).
5. Domain-randomized fine-tune + measure transfer vs. the idealized baseline.

## CRITICAL transfer gotcha
The policy consumes VecNormalize-normalized observations (saved normalizer pkl), NOT the
raw env observations. The bridge MUST: (a) build the raw 20-D obs with the env's EXACT
normalization constants, then (b) apply the saved VecNormalize mean/std (clip 10), then
feed the actor. Action: [surge, yaw_rate] in [-1,1], rescaled by max_cmd / max_yaw_rate.
The original training env file + notebook + trained model + normalizer pkl live in
~/Desktop/Thesis (also backed up to Google Drive).

## Working style
Build incrementally, verify each step before moving on. GUI checks (does the boat float,
does Gazebo render) require the human to look — ask them to run the GUI command and report.

## Reference artifacts (in this project, copied from ~/Desktop/Thesis)
- reference/twoglider_2km_currents_adcp.py     — training environment (obs builder, EKF, dynamics)
- reference/train_sac_curriculum_local(1).ipynb — training/eval notebook (shows how model+normalizer are loaded)
- reference/trained_policy/best_model.zip       — trained SAC model (run: 20km_ciric_version1)
- reference/trained_policy/vec_normalize_20km_ciric_version1.pkl — MATCHING VecNormalize (same run)
MATCHED PAIR from run 20km_ciric_version1 — do not mix with other runs.
The 20km config is a 10x scale-up of the 2km setup; the env file is shared (despite the 2km name).
