#!/bin/bash
# Recreate USDNet/ exactly as used for the challenge:
#   upstream insait-institute/USDNet @ 0ba303d
#   + usdnet_0ba303d.patch   (our changes to 10 upstream files)
#   + addons/                (files we added: env.sh, eval/train/infer scripts, ...)
# Creates <repo>/USDNet. Upstream code is MIT licensed, see LICENSE-USDNet.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$(dirname "$HERE")/USDNet"
if [ -e "$DEST" ]; then
    echo "$DEST already exists; remove it first" >&2
    exit 1
fi
git clone https://github.com/insait-institute/USDNet.git "$DEST"
git -C "$DEST" checkout -q 0ba303d
git -C "$DEST" apply --whitespace=nowarn "$HERE/usdnet_0ba303d.patch"
cp -Rp "$HERE/addons/." "$DEST/"
echo "USDNet ready at $DEST"
