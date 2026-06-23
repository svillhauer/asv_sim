#!/usr/bin/env bash
# Run once inside the container to clone + build VRX.
# Output lands in /work/vrx_ws (= ~/asv_sim/vrx_ws on the host) so it
# persists across container restarts.
set -e

source /opt/ros/jazzy/setup.bash

WS=/work/vrx_ws
SRC="$WS/src/vrx"

mkdir -p "$WS/src"

# Clone VRX jazzy branch if not already present
if [ ! -d "$SRC" ]; then
    echo "=== Cloning VRX (jazzy branch) ==="
    git clone --depth=1 -b jazzy https://github.com/osrf/vrx.git "$SRC"
else
    echo "=== VRX source already present at $SRC — skipping clone ==="
fi

cd "$WS"

# Refresh rosdep indexes and install remaining deps
echo "=== Running rosdep ==="
rosdep update --rosdistro jazzy
rosdep install --from-paths src --ignore-src -r -y \
    --skip-keys "ament_cmake_pycodestyle"

# Build (Release mode; symlink-install so Python scripts are editable)
echo "=== Building VRX (this takes 5-15 min the first time) ==="
colcon build \
    --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --event-handlers console_cohesion+

echo ""
echo "=== VRX build complete ==="
echo "Source with:  source /work/vrx_ws/install/setup.bash"
echo "Quick launch: ros2 launch vrx_gz competition.launch.py world:=sydney_regatta"
