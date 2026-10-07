# The vision gate (dsq-zig-vision-1, TF_DSV41_IMAGES=native), a fragment the combined pod job sources inside
# M1_JOB=m2b on a TWO-GPU pod after the pack's local copy. Needs: Z (staged tree), LP (the shards up to layer $hi; the
# vision tower and the 43 gate.bias_vl live in native.safetensors, which the copy takes), LOCAL, RES, hi, and job.sh's
# `step` / `left`. Python twin: /dsv41-tf/src (prod 8474f31, e1e2x extensions; VISION_PYSRC overrides).
#   1. vision-ref (~8 min): tools/zig/dsv41_vision/vision_ref.py at TP=2, one load: 3 Pillow-made images through prod's
#      vision_prep, the tower's stage dumps and span rows on rank 0 ($LOCAL/vision-ref/img), then 3 image prompts
#      (text around the images' spans; image rows from vision.STORE, text bias) through dsv41_m2b_ref.py's own path
#      into p1..p3 (state dumps, VISION_STEPS greedy tokens, the AOT set with the image paths' Triton variants)
#   2. vision-vit (~1 min, GPU 0): tf-dsv41-m1 vision: the Zig tower's embed / block0 / 1 / 15 / 31 / vit / gelu stages
#      and span rows == Python's, bit for bit ("PASS vision ViT + aligner == Python: 3/3 images")
#   3. vision-prefill-p1..p3 (~2 min each): tf-dsv41-m1 m2b with M2B_PREFILL=1 on each prompt: the Zig prefill from an
#      empty slot (image rows into rank 0's embedding, Engram's keep), every state role == Python's, then the tokens
#   4. vision-generate (~3 min): the three prompts through the served engine (tf-dsv41-m1 generate, drafts off):
#      "PASS M4 CLI == HTTP: 3/3 replies equal" (here: Zig's replies == Python's)
# Each run prints "PASS vision ..." / "FAIL vision ..." into $RES/steps.txt. Knobs: VISION_STEPS (32), VISION_REF_MIN
# (12), VISION_RUNS ("vit prefill generate").
vision_gate() {
    local ref=$LOCAL/vision-ref vdir=$LOCAL/vision-ref/img pysrc=${VISION_PYSRC:-/dsv41-tf/src} d i=0 mb
    rm -rf "$ref"
    mkdir -p "$RES/vision"
    unset TF_DSV41_BIAS_VL TF_DSV41_KV_SPLIT
    if ! { PYTHONPATH=$pysrc step vision-ref "${VISION_REF_MIN:-12}" python -u -B "$Z/tools/zig/dsv41_vision/vision_ref.py" \
            --vdir "$vdir" -- --pack "$LP" --out "$ref" --layers "0-$hi" --steps "${VISION_STEPS:-32}" --prompts 1,2,3 \
            --port 29781 \
         && step vision-engram-host 3 python -B "$Z/tools/zig/dsv41_engram_host.py" --py-src "$pysrc" \
            --config "$LP/config.json" --tokenizer "$LP/tokenizer.json" --out "$ref/engram-host.bin"; }; then
        echo "FAIL vision: no reference" | tee -a "$RES/steps.txt"
        return 1
    fi
    for d in "$ref"/p*; do cp "$ref/engram-host.bin" "$d/"; done
    cp "$vdir/manifest.json" "$ref/trace.jsonl" "$RES/vision/"
    local runs=${VISION_RUNS:-vit prefill generate}
    if [[ " $runs " == *" vit "* ]]; then
        if CUDA_VISIBLE_DEVICES=0 step vision-vit 4 "$Z/bin/tf-dsv41-m1" vision "$LP" "$vdir"; then
            echo "PASS vision vit" | tee -a "$RES/steps.txt"
        else echo "FAIL vision vit (first differing stage in vision-vit.log)" | tee -a "$RES/steps.txt"; fi
        cp "$RES/vision-vit.log" "$RES/vision/" 2>/dev/null
    fi
    export TF_DSV41_IMAGES=native TF_DSV41_VISION_HOLD=$vdir
    if [[ " $runs " == *" prefill "* ]]; then
        for d in "$ref"/p1 "$ref"/p2 "$ref"/p3; do
            mb=$(left); (( mb > 4 )) && mb=4
            (( mb >= 3 )) || { echo "FAIL vision prefill $(basename "$d"): skipped, $mb min left" | tee -a "$RES/steps.txt"; continue; }
            if (export M2B_PREFILL=1 M2B_PORT=$(( 29783 + i ))
                step "vision-prefill-$(basename "$d")" "$mb" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" \
                    "$LP" "$d" "$RES/vision/prefill-$(basename "$d")"); then
                echo "PASS vision prefill $(basename "$d") (state roles, tokens)" | tee -a "$RES/steps.txt"
            else echo "FAIL vision prefill $(basename "$d")" | tee -a "$RES/steps.txt"; fi
            i=$(( i + 1 ))
        done
    fi
    if [[ " $runs " == *" generate "* ]] && (( $(left) >= 4 )); then
        if TF_DSV41_LAYERS=0-$hi step vision-generate 5 bash "$Z/tools/zig/dsv41_vision/gen_ranks.sh" "$Z/bin/tf-dsv41-m1" \
                "$LP" "$ref/p1" "$ref/trace.jsonl" "$RES/vision/generate"; then
            echo "PASS vision generate (replies == Python's)" | tee -a "$RES/steps.txt"
        else echo "FAIL vision generate" | tee -a "$RES/steps.txt"; fi
    fi
    unset TF_DSV41_IMAGES TF_DSV41_VISION_HOLD
    rm -rf "$ref"/p*/rank*/state
}
