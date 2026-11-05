#!/usr/bin/env bash
# Shared configuration for the standalone Slurm scripts.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/project-env.sh"

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing command: $1" >&2
        return 1
    }
}
