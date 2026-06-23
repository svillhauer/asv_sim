FROM osrf/ros:jazzy-desktop

# Add OSRF Gazebo apt repo (needed for python3-sdformat14 and gz Python bindings)
RUN apt-get update && apt-get install -y curl gnupg lsb-release \
    && curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
         -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
         http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
         > /etc/apt/sources.list.d/gazebo-stable.list \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 <-> Gazebo integration; this pulls in Gazebo Harmonic on Jazzy
RUN apt-get update && apt-get install -y \
      ros-jazzy-ros-gz \
      mesa-utils \
      x11-apps \
      # --- VRX build tools ---
      python3-colcon-common-extensions \
      python3-colcon-ros \
      # --- VRX apt-installable ROS deps (speeds up rosdep step) ---
      ros-jazzy-xacro \
      ros-jazzy-robot-state-publisher \
      ros-jazzy-joint-state-publisher \
      ros-jazzy-geographic-msgs \
      ros-jazzy-joy \
      ros-jazzy-tf2-ros \
      ros-jazzy-tf2-geometry-msgs \
      python3-sdformat14 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages \
         "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --no-cache-dir --break-system-packages \
         "stable-baselines3==2.3.2" \
         "gymnasium==0.29.1"

# Auto-source ROS; also source the VRX workspace if it has been built.
# The extra GZ_SIM_RESOURCE_PATH entries let Gazebo resolve model://wamv_description/...
# and model://vrx_gazebo/... URIs, which need the share/ parent (not the models/ subdir).
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo 'if [ -f /work/vrx_ws/install/setup.bash ]; then source /work/vrx_ws/install/setup.bash; fi' >> /root/.bashrc && \
    echo 'export GZ_SIM_RESOURCE_PATH=/work/vrx_ws/install/wamv_description/share:/work/vrx_ws/install/wamv_gazebo/share:/work/vrx_ws/install/vrx_gazebo/share:${GZ_SIM_RESOURCE_PATH}' >> /root/.bashrc

CMD ["bash"]
