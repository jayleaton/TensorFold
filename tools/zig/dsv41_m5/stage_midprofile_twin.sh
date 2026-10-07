#!/usr/bin/env bash
# Build midprofile.tar.gz for job.sh's M1_JOB=m5: the midprofile Python twin (split KV + sessions; TF_MIDPROFILE,
# default ../tensorfold-decode1, at REF, default the branch dsv41-midprofile) as /opt/midprofile/{src,tests,COMMIT},
# plus every CUDA extension it loads prebuilt for sm_120 into /opt/midprofile/ext, in the x86 e1e2 image (IMAGE,
# default tensorfold-dsv41:e1e2-8f43b45-sm120: the image r1c.tar.gz and e1e2x.tar.gz were built in, the pod's e1e2
# runtime's torch and nvcc). The build runs at the pod's absolute paths (the twin mounted at /opt/midprofile), so the
# ninja files, torch's JIT hashes and module names are the ones a pod computes: it loads them without compiling.
#
# The twin comes from `git archive` of REF (the checkout and its branch are not touched). A second prebuild over a
# copy of the extensions must compile nothing (the relocation check of scripts/e1e2/image.sh). Local only: no network
# volume, no pod. To ship it, either
#   - ZIG stage: R1C-style, `cp $OUT <stage dir>/midprofile.tar.gz` (runpod/stage-zigm1.sh needs a MIDPROFILE_TAR
#     line beside R1C_TAR to copy it into dsq/zigm1/, which its `sync --delete` would otherwise remove), or
#   - --upload (run by hand): the tarball to dsq/m5/midprofile.tar.gz on the network volume through the quant repo's
#     runpod/lib.sh (QUANT, default ../dsv41-exl3-quant); job.sh looks there when $Z has none.
#
#   tools/zig/dsv41_m5/stage_midprofile_twin.sh [--upload]      -> $OUT (default ../dsv41-zig-ref/midprofile.tar.gz)
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)
TF=${TF_MIDPROFILE:-$repo/../tensorfold-decode1}
REF=${REF:-dsv41-midprofile}
IMAGE=${IMAGE:-tensorfold-dsv41:e1e2-8f43b45-sm120}
ARCH=${ARCH:-12.0}
JOBS=${PREBUILD_JOBS:-4}
OUT=${OUT:-$repo/../dsv41-zig-ref/midprofile.tar.gz}
WORK=${WORK:-$HOME/.cache/dsv41-midprofile-twin}
QUANT=${QUANT:-$repo/../dsv41-exl3-quant}
upload=0
for a in "$@"; do [[ "$a" == --upload ]] && upload=1; done
log() { echo "[midprofile-twin $(date +%T)] $*" >&2; }

commit=$(git -C "$TF" rev-parse "$REF^{commit}")
docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "no image $IMAGE (tensorfold-kvq scripts/e1e2/image.sh, the sm120 variant)"; exit 1; }
rm -rf "$WORK" && mkdir -p "$WORK/opt/midprofile"
git -C "$TF" archive "$commit" src tests pyproject.toml | tar -x -C "$WORK/opt/midprofile"
echo "$commit" > "$WORK/opt/midprofile/COMMIT"
log "twin $REF = ${commit:0:12} from $TF -> $WORK/opt/midprofile"

run() { # the image with the twin at /opt/midprofile, as this user (the files stay ours)
    docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" -e HOME=/tmp -e MAX_JOBS="$JOBS" \
        -e PYTHONPATH=/opt/midprofile/src -e PYTHONDONTWRITEBYTECODE=1 -v "$WORK/opt/midprofile:/opt/midprofile" \
        --entrypoint sh "$IMAGE" -c "$1"
}
t0=$(date +%s)
run "TORCH_EXTENSIONS_DIR=/opt/midprofile/ext python /opt/e1e2/prebuild.py --arch $ARCH --jobs $JOBS 2>&1" \
    | tee "$WORK/opt/midprofile/prebuild.log" >&2 || true
grep -qE '^prebuild: ([0-9]+) / \1 built' "$WORK/opt/midprofile/prebuild.log" \
    || { log "prebuild FAILED:"; grep -E '^(FAIL|prebuild:)' "$WORK/opt/midprofile/prebuild.log" >&2; exit 1; }
log "prebuild: $(grep -c '^built' "$WORK/opt/midprofile/prebuild.log") modules in $(( $(date +%s) - t0 )) s"
# relocated (a copy at another path, the pod's cp -a of the tar): nothing compiles again
v=$(run "cp -a /opt/midprofile/ext /tmp/ext && TORCH_EXTENSIONS_DIR=/tmp/ext python /opt/e1e2/prebuild.py --arch $ARCH --jobs $JOBS 2>&1" || true)
if grep -q "building CUDA extension" <<< "$v" || ! grep -qE '^prebuild: ([0-9]+) / \1 built' <<< "$v"; then
    log "VERIFY FAILED: a relocated prebuild compiled again or failed"; grep -E "building|FAIL|prebuild:" <<< "$v" >&2; exit 1
fi
log "verify: the extensions load with no compile"
mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT" -C "$WORK" opt/midprofile
log "wrote $OUT ($(du -h "$OUT" | cut -f1); twin ${commit:0:12}, ext $(ls "$WORK/opt/midprofile/ext" | wc -l) modules)"
if (( upload )); then
    # shellcheck disable=SC2034  # lib.sh's rp_log reads it
    RP_TAG=stage-midprofile-twin
    # shellcheck source=/dev/null
    source "$QUANT/runpod/lib.sh"
    rp_load_config
    rp_need NETWORK_VOLUME_ID || exit 1
    rp_s3 cp "$OUT" "s3://$NETWORK_VOLUME_ID/dsq/m5/midprofile.tar.gz" --only-show-errors || { log "upload failed"; exit 1; }
    log "uploaded to dsq/m5/midprofile.tar.gz"
fi
rm -rf "$WORK"
