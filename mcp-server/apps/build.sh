#!/bin/bash
# Build all MCP Apps

APPS_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$APPS_DIR/dist"

# Clean dist
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# Build each app
for app in manifold-viewer cohort-heatmap trajectory-viewer force-field mode-flow gap-analysis; do
  echo "Building $app..."
  cd "$APPS_DIR/$app"
  npx vite build --config ../vite.config.ts --outDir "$DIST_DIR/$app"
  if [ $? -ne 0 ]; then
    echo "Failed to build $app"
    exit 1
  fi
done

echo "All apps built successfully!"
ls -la "$DIST_DIR"
