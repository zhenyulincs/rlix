#!/bin/bash
# Check and set PID/thread limits

echo "=== BEFORE ==="
echo "ulimit -u: $(ulimit -u)"
echo "cgroup pids.max: $(cat /sys/fs/cgroup/pids.max)"
THREADS=$(ps -eLf | wc -l)
echo "current threads: $THREADS"

echo ""
echo "=== SETTING LIMITS ==="
ulimit -u 65535
echo 65535 > /sys/fs/cgroup/pids.max

echo ""
echo "=== AFTER ==="
echo "ulimit -u: $(ulimit -u)"
echo "cgroup pids.max: $(cat /sys/fs/cgroup/pids.max)"
THREADS=$(ps -eLf | wc -l)
HEADROOM=$((65535 - THREADS))
echo "current threads: $THREADS"
echo "available threads: $HEADROOM"
