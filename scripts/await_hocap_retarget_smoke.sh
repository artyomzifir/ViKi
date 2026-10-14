#!/usr/bin/env bash
set -euo pipefail

smoke_pid="$1"
while kill -0 "$smoke_pid" 2>/dev/null; do
    sleep 15
done

smoke_plan=$(find data/datasets/hocap_eightview -mindepth 2 -maxdepth 2 \
    -path '*/subject_1_20231025_165502_*/plan.h5' -print -quit)
if [[ -z "$smoke_plan" ]]; then
    echo "smoke ended without plan.h5; full HO-Cap retarget was not started" >&2
    exit 1
fi

batch_exit=0
docker compose run --rm --no-deps --entrypoint python3 cli \
    /app/scripts/run_hocap_retarget_batch.py \
    data/datasets/hocap_eightview --require-smoke --workers 4 || batch_exit=$?

if [[ -f data/datasets/hocap_eightview/retarget_batch_status.json ]]; then
    python3 scripts/build_hocap_retarget_report.py \
        data/datasets/hocap_eightview/retarget_batch_status.json \
        docs/2026-10-14-hocap-retarget-results.md
fi
exit "$batch_exit"
