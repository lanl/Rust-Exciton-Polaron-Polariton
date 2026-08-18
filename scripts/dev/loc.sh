#!/usr/bin/env bash
set -euo pipefail

if command -v tokei >/dev/null 2>&1; then
    tokei . \
      --exclude target \
      --exclude saved_data \
      --exclude saved_data_lm \
      --exclude pp_data \
      --exclude gnuplot_data \
      --exclude dielectric_scan \
      --exclude all_sources_pp_ex.md \
      --exclude all_sources_pp_ex_2.md
else
    echo "tokei is not installed."
    echo "Install with:"
    echo "  cargo install tokei"
    exit 1
fi
