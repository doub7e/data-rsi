#!/bin/sh
# Small CPU-worker-only capability probe. No host socket or added privileges.
set -u
id
uname -r
awk '/^(CapEff|NoNewPrivs|Seccomp):/ {print}' /proc/self/status
docker --version
dockerd --version
export DOCKER_HOST=unix:///tmp/dind-probe.sock
export DOCKER_TLS_CERTDIR=
unset DOCKER_TLS_VERIFY DOCKER_CERT_PATH

probe_mode() {
  mode="$1"
  shift
  log="/tmp/dind-$mode.log"
  dockerd --host="$DOCKER_HOST" --data-root="/tmp/docker-$mode" --exec-root="/tmp/exec-$mode" --pidfile="/tmp/docker-$mode.pid" "$@" >"$log" 2>&1 &
  daemon_pid=$!
  ready=false
  i=0
  while [ "$i" -lt 45 ]; do
    if timeout 3 docker info >/tmp/docker-info 2>&1; then ready=true; break; fi
    if ! kill -0 "$daemon_pid" 2>/dev/null; then break; fi
    sleep 1
    i=$((i + 1))
  done
  pull=null
  run=null
  import_rc=null
  local_run=null
  if [ "$ready" = true ]; then
    cat /tmp/docker-info
    # Local static BusyBox rootfs distinguishes runtime denial from registry errors.
    mkdir -p /tmp/probe-root/bin
    cp /bin/busybox /tmp/probe-root/bin/busybox
    tar -C /tmp/probe-root -cf /tmp/rootfs.tar .
    timeout 30 docker import /tmp/rootfs.tar dind-local-probe:check
    import_rc=$?
    if [ "$import_rc" -eq 0 ]; then
      timeout 30 docker run --rm --network none dind-local-probe:check /bin/busybox echo nested-container-ok
      local_run=$?
    fi
    timeout 60 docker pull hello-world:latest
    pull=$?
    if [ "$pull" -eq 0 ]; then
      timeout 30 docker run --rm --network none hello-world:latest
      run=$?
    fi
    docker image inspect hello-world:latest --format '{{json .RepoDigests}}' 2>/dev/null || true
  fi
  printf '\nDAEMON_LOG mode=%s\n' "$mode"
  cat "$log"
  kill "$daemon_pid" 2>/dev/null || true
  wait "$daemon_pid"
  daemon_exit=$?
  printf 'PROBE_RESULT={"mode":"%s","daemon_ready":%s,"daemon_exit":%s,"local_import_exit":%s,"local_run_exit":%s,"pull_exit":%s,"run_exit":%s}\n' "$mode" "$ready" "$daemon_exit" "$import_rc" "$local_run" "$pull" "$run"
}

probe_mode default
# This only disables optional Docker features; it does not change pod security.
probe_mode vfs-no-bridge --storage-driver=vfs --iptables=false --ip6tables=false --bridge=none --ip-forward=false --ip-masq=false
exit 0
