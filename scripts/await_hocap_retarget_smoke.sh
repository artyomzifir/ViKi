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
    archive=paper/results/hocap/retarget
    mkdir -p "$archive"
    source_status=data/datasets/hocap_eightview/retarget_batch_status.json
    archived_status="$archive/retarget_batch_status.json"
    if [[ -f "$archived_status" ]] && ! cmp -s "$source_status" "$archived_status"; then
        echo "archived HO-Cap retarget status differs; refusing to overwrite" >&2
        exit 1
    fi
    if [[ ! -f "$archived_status" ]]; then
        cp "$source_status" "$archived_status"
    fi
    python3 scripts/build_hocap_retarget_report.py \
        "$archived_status" "$archive/2026-10-14-hocap-retarget-results.md"
    (cd "$archive" && sha256sum retarget_batch_status.json \
        2026-10-14-hocap-retarget-results.md > SHA256SUMS)
fi
exit "$batch_exit"
