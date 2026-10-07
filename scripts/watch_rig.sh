#!/usr/bin/env bash
# Log the host hard enough to survive a freeze.
#
# journald syncs to disk every 5 minutes by default, so a hard lock-up takes the
# last minutes of kernel messages with it — which is why five freezes on this rig
# left no trace. This writes kernel messages as they arrive and fsyncs, so what
# is on disk is what the kernel actually said before it stopped talking.
#
#   scripts/watch_rig.sh <outdir> [sample_seconds]
set -u
OUT="${1:-data/_stability/watch-$(date +%Y%m%d-%H%M%S)}"
IVL="${2:-1}"
mkdir -p "$OUT"
echo "logging to $OUT"

# 1) kernel ring buffer, followed live, line-buffered
( stdbuf -oL dmesg --follow --time-format=iso 2>/dev/null >> "$OUT/kernel.log" ) &
DMESG_PID=$!

# 2) one-second host sample + fsync
(
  echo "ts uptime_s load1 mem_used_mb kinect_usb rec_procs" > "$OUT/sample.log"
  while true; do
    printf '%s %s %s %s %s %s\n' \
      "$(date '+%F_%T')" \
      "$(cut -d. -f1 /proc/uptime)" \
      "$(cut -d' ' -f1 /proc/loadavg)" \
      "$(free -m | awk '/^Mem:/{print $3}')" \
      "$(lsusb 2>/dev/null | grep -c '045e:097[cd]')" \
      "$(pgrep -c k4arecorder 2>/dev/null || echo 0)" \
      >> "$OUT/sample.log"
    sync
    sleep "$IVL"
  done
) &
SAMPLE_PID=$!

echo "$DMESG_PID $SAMPLE_PID" > "$OUT/pids"
trap 'kill $DMESG_PID $SAMPLE_PID 2>/dev/null; sync; echo stopped' INT TERM
wait
