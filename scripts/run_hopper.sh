#!/usr/bin/env bash
set -euo pipefail

netid=${1:-}
device=${2:-}
episodes=${3:-256}
previous=${4:-}
chunk_size=${HOPPER_CHUNK_EPISODES:-100000}
if [[ $# -gt 4 || ! "$netid" =~ ^[[:alnum:]_][[:alnum:]_.-]*$ ||
      ! "$device" =~ ^(cpu|cuda)?$ || ! "$episodes" =~ ^[1-9][0-9]{0,7}$ ||
      ! "$chunk_size" =~ ^[1-9][0-9]{0,7}$ ||
      ( -n "$previous" && ! "$previous" =~ ^ppo-1v1\.[[:alnum:]]{8}$ ) ||
      ( $# -gt 1 && -z "$device" ) ]]; then
    echo "Usage: bash scripts/run_hopper.sh GMU_NETID [cpu|cuda [EPISODES [PREVIOUS_RUN_NAME]]]" >&2
    exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT

commit=$(git -C "$repo" rev-parse HEAD)
mkdir "$staging/source"
git -C "$repo" archive "$commit" | tar -xf - -C "$staging/source"
cp "$repo/scripts/hopper_smoke.slurm" "$staging/source/scripts/"
if [[ -n "$device" ]]; then
    cp "$repo/src/maritime_nsrl/"{cli,experiments}.py "$staging/source/src/maritime_nsrl/"
fi
printf '%s\n' "$commit" > "$staging/source/source-commit.txt"
# Hopper's GNU tar warns about macOS metadata in archives.
COPYFILE_DISABLE=1 tar -c --no-xattrs -zf "$staging/source.tar.gz" -C "$staging/source" .

remote_script='
    set -eu
    umask 077
    device=$1
    episodes=$2
    previous=$3
    chunk_size=$4
    root="${SCRATCH:-/scratch/$USER}/symbolic-ctf-agents"
    mkdir -p "$root"
    run_name=smoke
    if [[ -n "$device" ]]; then
        run_name=ppo-1v1
    fi
    if [[ -n "$previous" && ! -f "$root/$previous/runs/ppo-1v1/training/training_complete.json" ]]; then
        echo "Completed training checkpoint missing in $root/$previous" >&2
        exit 1
    fi
    run_dir=$(mktemp -d "$root/$run_name.XXXXXXXX")
    tar -xzf - -C "$run_dir"
    cd "$run_dir"
    resources=()
    if [[ "$device" == cuda ]]; then
        resources+=(--partition=gpuq --qos=gpu --gres=gpu:1g.10gb:1)
    fi
    printf "Run directory: %s\n" "$run_dir"
    if [[ -z "$device" ]]; then
        job=$(sbatch --parsable scripts/hopper_smoke.slurm)
        printf "Slurm job: %s\n" "$job"
        exit
    fi

    checkpoint=
    if [[ -n "$previous" ]]; then
        checkpoint="../$previous/runs/ppo-1v1/training/policy.zip"
    fi
    remaining=$episodes
    chunk=1
    dependency=()
    while (( remaining > 0 )); do
        count=$chunk_size
        run="runs/ppo-1v1-chunk-$chunk"
        if (( remaining <= chunk_size )); then
            count=$remaining
            run=runs/ppo-1v1
        fi
        remaining=$((remaining - count))
        time=01:00:00
        if (( count > 256 )); then
            time=2-00:00:00
        fi
        job=$(sbatch --parsable ${resources[@]+"${resources[@]}"} ${dependency[@]+"${dependency[@]}"} \
            --time="$time" --job-name=ppo-1v1 scripts/hopper_smoke.slurm \
            "$device" "$count" "$checkpoint" "$run")
        printf "Slurm job: %s (%s episodes, %s)\n" "$job" "$count" "$run"
        dependency=(--dependency="afterok:${job%%;*}" --kill-on-invalid-dep=yes)
        checkpoint="$run/training/policy.zip"
        chunk=$((chunk + 1))
    done
'
ssh "$netid@hopper.orc.gmu.edu" "bash -c '$remote_script' _ '$device' '$episodes' '$previous' '$chunk_size'" \
    < "$staging/source.tar.gz"
