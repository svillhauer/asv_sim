#!/usr/bin/env python3
"""
Plot ASV + glider trajectories from ROS 2 MCAP bags.
Handles truncated bags (computer-died mid-write) by reading forward-only.
"""

import math
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from mcap.stream_reader import StreamReader
from mcap.records import Schema, Channel, Message, Metadata
from mcap_ros2.decoder import DecoderFactory
from mcap.well_known import MessageEncoding


BAG_PATHS = {
    "run4": Path("bags/run_20260624_175012/run_20260624_175012_0.mcap"),
}

TOPICS_WANTED = {
    "/wamv/sensors/gps/gps/fix",
    "/asv_bridge/glider_poses",
    "/asv_bridge/obs_normalized",
    "/asv_bridge/action",
    "/wamv/thrusters/left/thrust",
    "/wamv/thrusters/right/thrust",
}


def gps_to_xy(lat, lon, lat0, lon0):
    """Approximate local East/North [m] from a reference origin."""
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111320.0
    y = (lat - lat0) * 111320.0
    return x, y


def iter_decoded(mcap_path):
    """Forward-only decoder that survives truncated MCAP files."""
    factory = DecoderFactory()
    schemas = {}
    channels = {}
    decoders = {}  # schema_id -> decode fn

    def get_decoder(schema):
        if schema.id not in decoders:
            fn = factory.decoder_for(MessageEncoding.CDR, schema)
            decoders[schema.id] = fn
        return decoders[schema.id]

    with open(mcap_path, "rb") as f:
        sr = StreamReader(f, record_size_limit=None)
        try:
            for record in sr.records:
                rtype = type(record).__name__
                if rtype == "Schema":
                    schemas[record.id] = record
                elif rtype == "Channel":
                    channels[record.id] = record
                elif rtype == "Message":
                    ch = channels.get(record.channel_id)
                    if ch is None or ch.topic not in TOPICS_WANTED:
                        continue
                    sch = schemas.get(ch.schema_id)
                    if sch is None:
                        continue
                    dec = get_decoder(sch)
                    if dec is None:
                        continue
                    try:
                        msg = dec(record.data)
                    except Exception:
                        continue
                    yield record.log_time, ch.topic, msg
        except Exception as e:
            print(f"  [truncated at {mcap_path.name}: {e}]", file=sys.stderr)


def read_bag(mcap_path):
    """Return a dict of lists, one per topic."""
    print(f"Reading {mcap_path.name} ...", flush=True)
    data = {t: [] for t in TOPICS_WANTED}
    for t_ns, topic, msg in iter_decoded(mcap_path):
        data[topic].append((t_ns, msg))
    for topic, entries in data.items():
        if entries:
            print(f"  {topic}: {len(entries)} msgs", flush=True)
    return data


def extract_gps(entries):
    """Return arrays (t_s, lat, lon) using sim-time stamp from the message header."""
    times, lats, lons = [], [], []
    for _log_ns, msg in entries:
        # Use simulation time from the message header, not wall-clock log_time.
        # log_time has jitter (bag recorder batches) and runs at wall-clock rate
        # (RTF×  faster than sim), both of which inflate the speed estimate.
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        times.append(t)
        lats.append(msg.latitude)
        lons.append(msg.longitude)
    return np.array(times), np.array(lats), np.array(lons)


def extract_glider_poses(entries):
    """Return list of (t_s, [(gx, gy), ...]) — one entry per message."""
    result = []
    for t_ns, msg in entries:
        poses = []
        for pose in msg.poses:
            poses.append((pose.position.x, pose.position.y))
        result.append((t_ns / 1e9, poses))
    return result


def extract_float_array(entries):
    """Return (t_s, array[N, D])."""
    times, arrs = [], []
    for t_ns, msg in entries:
        times.append(t_ns / 1e9)
        arrs.append(list(msg.data))
    return np.array(times), np.array(arrs) if arrs else np.empty((0, 0))


def extract_thrust(entries):
    times, vals = [], []
    for t_ns, msg in entries:
        times.append(t_ns / 1e9)
        vals.append(msg.data)
    return np.array(times), np.array(vals)


def plot_run(run_name, data, out_dir):
    gps_entries = data["/wamv/sensors/gps/gps/fix"]
    glider_entries = data["/asv_bridge/glider_poses"]
    action_entries = data["/asv_bridge/action"]
    thrust_l_entries = data["/wamv/thrusters/left/thrust"]
    thrust_r_entries = data["/wamv/thrusters/right/thrust"]
    obs_entries = data["/asv_bridge/obs_normalized"]

    if not gps_entries:
        print(f"  [{run_name}] no GPS data, skipping trajectory plot")
        return

    t_gps, lats, lons = extract_gps(gps_entries)
    lat0, lon0 = lats[0], lons[0]
    t0 = t_gps[0]

    xs = np.array([(lon - lon0) * math.cos(math.radians(lat0)) * 111320.0 for lon in lons])
    ys = np.array([(lat - lat0) * 111320.0 for lat in lats])
    t_rel = t_gps - t0
    dur = t_rel[-1]

    # ------------------------------------------------------------------ Fig 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{run_name}  —  {dur:.0f} s  ({len(gps_entries)} GPS pts)")

    ax = axes[0]
    sc = ax.scatter(xs, ys, c=t_rel, cmap="plasma", s=4, zorder=3)
    plt.colorbar(sc, ax=ax, label="time [s]")
    ax.plot(xs[0], ys[0], "g^", ms=10, label="start", zorder=5)
    ax.plot(xs[-1], ys[-1], "rs", ms=10, label="end", zorder=5)
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("ASV trajectory"); ax.set_aspect("equal"); ax.legend()
    ax.grid(True, alpha=0.3)

    # Glider trajectories (one point per EKF tick)
    if glider_entries:
        glider_data = extract_glider_poses(glider_entries)
        colors_g = ["cyan", "magenta"]
        # Transpose: list-of-steps × n_gliders  →  per-glider list of (x,y)
        n_gliders = len(glider_data[0][1]) if glider_data else 0
        for i in range(n_gliders):
            gxs = [poses[i][0] for _, poses in glider_data]
            gys = [poses[i][1] for _, poses in glider_data]
            c = colors_g[i % len(colors_g)]
            ax.plot(gxs, gys, "-o", color=c, ms=5, lw=1.2,
                    alpha=0.8, label=f"glider {i}", zorder=4)
            ax.plot(gxs[0],  gys[0],  "^", color=c, ms=9, zorder=5)
            ax.plot(gxs[-1], gys[-1], "D", color=c, ms=9, zorder=5)
        ax.legend()

    # ------------------------------------------------------------------ Fig 1b — speed over time
    ax2 = axes[1]
    if len(xs) > 1:
        dt = np.diff(t_gps)
        dx = np.diff(xs)
        dy = np.diff(ys)
        speed = np.sqrt(dx**2 + dy**2) / np.where(dt > 0, dt, 1e-9)
        t_mid = 0.5 * (t_rel[:-1] + t_rel[1:])
        ax2.plot(t_mid, speed, lw=1, label="speed [m/s]")
        ax2.set_xlabel("time [s]"); ax2.set_ylabel("speed [m/s]")
        ax2.set_title("ASV speed over time"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / f"{run_name}_trajectory.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  -> {p}")

    # ------------------------------------------------------------------ Fig 2: actions + thrust
    if action_entries or thrust_l_entries:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
        fig.suptitle(f"{run_name}  —  policy actions & thruster commands")

        if action_entries:
            t_act, act = extract_float_array(action_entries)
            t_act -= t0
            ax = axes[0]
            if act.ndim == 2 and act.shape[1] >= 2:
                ax.plot(t_act, act[:, 0], label="surge cmd")
                ax.plot(t_act, act[:, 1], label="yaw_rate cmd")
            ax.set_ylabel("action [-1,1]"); ax.set_title("Policy actions")
            ax.legend(); ax.grid(True, alpha=0.3)

        if thrust_l_entries:
            t_tl, vl = extract_thrust(thrust_l_entries)
            t_tr, vr = extract_thrust(thrust_r_entries)
            t_tl -= t0; t_tr -= t0
            ax = axes[1]
            ax.plot(t_tl, vl, label="left thrust [N]")
            ax.plot(t_tr, vr, label="right thrust [N]")
            ax.set_xlabel("time [s]"); ax.set_ylabel("thrust [N]")
            ax.set_title("Thruster commands"); ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        p = out_dir / f"{run_name}_actions.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  -> {p}")

    # ------------------------------------------------------------------ Fig 3: obs (if present)
    if obs_entries:
        t_obs, obs = extract_float_array(obs_entries)
        if obs.shape[0] > 1:
            t_obs -= t0
            fig, ax = plt.subplots(figsize=(12, 5))
            for i in range(min(obs.shape[1], 20)):
                ax.plot(t_obs, obs[:, i], lw=0.8, alpha=0.7, label=f"obs[{i}]")
            ax.set_xlabel("time [s]"); ax.set_ylabel("normalized value")
            ax.set_title(f"{run_name}  —  normalized observation vector (20-D)")
            ax.legend(fontsize=6, ncol=4); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            p = out_dir / f"{run_name}_obs.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"  -> {p}")


def combined_trajectory_plot(all_run_data, out_dir):
    """Overlay all runs on one plot."""
    colors = {"run1": "steelblue", "run2": "tomato"}
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_title("All runs — ASV trajectories")

    for run_name, data in all_run_data.items():
        gps_entries = data["/wamv/sensors/gps/gps/fix"]
        if not gps_entries:
            continue
        t_gps, lats, lons = extract_gps(gps_entries)
        lat0, lon0 = lats[0], lons[0]
        xs = np.array([(lon - lon0) * math.cos(math.radians(lat0)) * 111320.0 for lon in lons])
        ys = np.array([(lat - lat0) * 111320.0 for lat in lats])
        c = colors.get(run_name, "gray")
        ax.plot(xs, ys, lw=1.2, color=c, alpha=0.8, label=run_name)
        ax.plot(xs[0], ys[0], "^", color=c, ms=9, zorder=5)
        ax.plot(xs[-1], ys[-1], "s", color=c, ms=9, zorder=5)

    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_aspect("equal"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "combined_trajectories.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  -> {p}")


def main():
    out_dir = Path.home() / "asv_sim" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for run_name, bag_path in BAG_PATHS.items():
        if not bag_path.exists():
            print(f"[skip] {bag_path} not found")
            continue
        data = read_bag(bag_path)
        all_data[run_name] = data
        plot_run(run_name, data, out_dir)

    if len(all_data) > 1:
        combined_trajectory_plot(all_data, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
