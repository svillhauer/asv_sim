#!/usr/bin/env bash
xhost +local:docker >/dev/null
docker run --rm -it \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e __NV_PRIME_RENDER_OFFLOAD=1 \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -e GZ_SIM_RESOURCE_PATH=/work/models:/work/worlds \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/asv_sim:/work \
  -w /work \
  asv_sim:jazzy "$@"
