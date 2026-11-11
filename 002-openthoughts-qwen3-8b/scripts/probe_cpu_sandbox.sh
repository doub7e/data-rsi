#!/bin/bash
set -uo pipefail
id
uname -r
awk '/^(CapEff|NoNewPrivs|Seccomp):/ {print}' /proc/self/status
printf '\nUser namespace check\n'
unshare -Ur true
userns=$?
printf '\nPodman information\n'
podman --storage-driver=vfs --root /tmp/probe-storage --runroot /tmp/probe-run info --debug
info=$?
run=125
if [ "$info" -eq 0 ]; then
  timeout 90 podman --storage-driver=vfs --root /tmp/probe-storage --runroot /tmp/probe-run run --rm --network none docker.io/library/alpine:3.20 echo nested-container-ok
  run=$?
fi
printf 'PROBE_RESULT={"user_namespace_exit":%s,"podman_info_exit":%s,"nested_run_exit":%s}\n' "$userns" "$info" "$run"
exit 0
