#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || ! "$1" =~ ^[[:alnum:]_.-]+$ || "$1" == -* ]]; then
    echo "Usage: bash scripts/run_hopper_smoke.sh GMU_NETID" >&2
    exit 2
fi

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_hopper.sh" "$1"
