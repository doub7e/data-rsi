#!/bin/bash
set -eu
mkdir -p /logs/verifier
if [[ "$(cat /tmp/board-sandbox-test)" == sandbox-ok ]]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
