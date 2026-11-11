#!/usr/bin/env bash
set -euo pipefail
trap 'exit 0' TERM INT
sleep "${NATIVE_SANDBOX_TTL_SECONDS:-14400}" &
wait $!
