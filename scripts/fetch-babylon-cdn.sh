#!/usr/bin/env bash
set -euo pipefail

# Downloads Babylon.js UMD bundles for offline testing.
# Version is read from the local Babylon.js repo checkout
# (/testsuites/Babylon.js) or, failing that, from the npm registry.
# earcut (not on cdn.babylonjs.com) is copied from the repo checkout
# with an npm CDN fallback.
# Run once during Docker build or dev setup.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

BABYLON_REPO="/testsuites/Babylon.js"
BABYLON_PKG="$BABYLON_REPO/packages/public/umd/babylonjs/package.json"
if [ -f "$BABYLON_PKG" ]; then
    VERSION=$(python3 -c "import json; print(json.load(open('$BABYLON_PKG'))['version'])")
else
    echo "Local Babylon.js repo not found — reading latest version from npm registry..."
    VERSION=$(curl -sL "https://registry.npmjs.org/babylonjs/latest" | python3 -c "import json,sys; print(json.load(sys.stdin)['version'])")
fi

CDN="https://cdn.babylonjs.com/v${VERSION}"
OUT="$ROOT/vendor/babylon-cdn"

mkdir -p "$OUT"

echo "Fetching Babylon.js v${VERSION} UMD bundles..."

curl -sL "$CDN/babylon.js"                         -o "$OUT/babylon.js"
curl -sL "$CDN/gui/babylon.gui.min.js"             -o "$OUT/babylon.gui.min.js"
curl -sL "$CDN/loaders/babylonjs.loaders.min.js"   -o "$OUT/babylonjs.loaders.min.js"
curl -sL "$CDN/materialsLibrary/babylonjs.materials.min.js" -o "$OUT/babylonjs.materials.min.js"
curl -sL "$CDN/serializers/babylonjs.serializers.min.js"    -o "$OUT/babylonjs.serializers.min.js"

# earcut is not on cdn.babylonjs.com — copy it (and its ISC license) from the
# local Babylon.js repo checkout (babylonServer's public dir), falling back to
# the npm CDN (mapbox/earcut does not commit dist/ to GitHub).
EARcut_DIR="$BABYLON_REPO/packages/tools/babylonServer/public"
if [ -f "$EARcut_DIR/earcut.min.js" ]; then
    cp "$EARcut_DIR/earcut.min.js" "$OUT/earcut.min.js"
    if [ -f "$EARcut_DIR/earcut.license" ]; then
        cp "$EARcut_DIR/earcut.license" "$OUT/earcut.license"
    fi
else
    echo "earcut not found in local repo — fetching from npm CDN..."
    curl -sL "https://unpkg.com/earcut@2.2.4/dist/earcut.min.js" -o "$OUT/earcut.min.js"
    curl -sL "https://raw.githubusercontent.com/mapbox/earcut/main/LICENSE" -o "$OUT/earcut.license"
fi

echo "Done — bundles saved to vendor/babylon-cdn/"
ls -lh "$OUT"
