#!/usr/bin/env bash
# Download the NREL windAI_bench 9k airfoil dataset (~52.7 GB) from the
# public AWS bucket. Requires the AWS CLI (no credentials needed:
# --no-sign-request). Run from the project root in Git Bash:
#     bash download_data.sh
#
# Install AWS CLI on Windows: https://aws.amazon.com/cli/
set -euo pipefail

DEST="data/airfoil_cfd_9k"
SRC="s3://nrel-pds-windai/aerodynamic_shapes/2D/9k_airfoils/"

mkdir -p "$DEST"

echo ">> This will download ~52.7 GB into $DEST"
echo ">> Listing top-level contents first:"
aws s3 ls --no-sign-request "$SRC"

read -r -p ">> Proceed with full download? [y/N] " ans
case "$ans" in
    [yY]*) ;;
    *) echo "Aborted."; exit 0 ;;
esac

echo ">> Downloading (this can take a while)..."
aws s3 sync --no-sign-request "$SRC" "$DEST"

echo
echo ">> Done. Verify the layout with:"
echo "     PYTHONPATH=src python -m airfoil_rbf inspect"
