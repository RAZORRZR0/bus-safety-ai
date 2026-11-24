#!/usr/bin/env sh
# Fetch the trained violence classifiers from the GitHub Release into models/.
set -eu
BASE="https://github.com/RAZORRZR0/TransportModels/releases/download/models-v1"
mkdir -p models
for f in temporal_pyramid_network.pth int2d-3d-cnn.pth legacy_mobilenetv2.pth; do
  [ -f "models/$f" ] || curl -fL --retry 3 -o "models/$f" "$BASE/$f"
done
echo "Models ready in models/"
