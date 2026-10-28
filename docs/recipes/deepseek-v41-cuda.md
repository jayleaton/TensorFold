# Experimental DeepSeek-V4.1 CUDA asset preparation

The native engine remains experimental. CPU build and host-test results do not qualify a checkpoint or GPU.
Serving still applies the native checkpoint representation and architecture refusals.

Build the CPU generators before preparing the CUDA images:

```sh
zig build dsv41-assets
zig build test-dsv41-assets-host test-dsv41-token-map
zig-out/bin/tf-dsv41-token-map --tokenizer MODEL/tokenizer.json --config MODEL/config.json --output ASSETS/engram-token-map.bin
zig-out/bin/tf-dsv41-assets aot-needs ASSETS/sigs.json ASSETS/needs.json
```

The token-map generator validates release geometry and records the original tokenizer and config SHA256 values.
Its metadata reader accepts the official unquantized configuration for asset generation; this does not admit that
matrix representation to the serving loader. The file uses `DSENGR01`, including prime buckets, multipliers and
lookup ids, with a JSON provenance sidecar. The legacy `engram-host.bin` fixture format is not a serving asset.

`aot-needs` executes the family's actual symbolic planner on the CPU. Supply the intended rank, world, context,
slot, draft, sampling, prefill and image settings through the same environment used for serving. It never opens a
CUDA driver. Enumerating an image or two-rank plan does not execute a GPU or distributed model.

The external Triton reference tools produce `sigs.json`, compile every specialization in `needs.json` for the
actual target SM, and write `aot/`. The reference RoPE generator writes `rope.bin` using fixed CPU table blocks;
its rows must cover context plus 2048. Keep the reference revision, compiler versions, options, manifests and
hashes with these generated assets. Do not substitute captured images from another checkpoint or profile.
The recipe's reference tools are versioned separately from this native runtime.

Build installable ARM64 binaries on the native ARM64 host with real runtime and family CUDA images:

```sh
zig build dsv41-serve -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Ddsv41-nvcc=/usr/local/cuda/bin/nvcc -Dsm=121
```

This installs rank zero's `tensorfold-dsv41` and the follower tool `tf-dsv41`. The runtime reads
`TF_DSV41_ASSETS/{rope.bin,engram-token-map.bin,aot/}`. Compile-only targets use inert synthetic images and
cannot produce a serving qualification receipt. Keep generated images and compiler caches outside source control.


## Native image input

Rank zero's shared `--vision` option selects image loading. Its validated boot settings travel over the existing
host plan link before every rank sizes buffers; the follower continues to use `tf-dsv41 follow`. The legacy
`TF_DSV41_IMAGES=native` selection remains available on rank zero. Set `TF_DSV41_BIAS_VL` to the matching image
routing-bias pack on every rank. Missing biases, malformed shapes, unsupported dtypes or absent tower weights
refuse startup. Bias vectors may be F32 or exact BF16 values widened to F32; tower tensors remain BF16.

Use `TF_DSV41_IMAGES=native` and the bias setting when enumerating image AOT variants. Real family builds also
compile the optional vision CUDA image; a prebuilt directory must supply it for an image-enabled boot.

The shared server owns image URLs, fetch security and request limits. The family decodes encoded input, processes
patches, retains image spans through terminal completion, and substitutes rank-zero tower features before the
existing embedding exchange. Image-containing decode windows refuse explicitly; image segments use prefill.

`zig build test-dsv41-vision` checks codecs, processor geometry, shared API requests, cancellation, ownership,
collision tracking, cache churn and buffer planning on the CPU. These checks do not qualify tower arithmetic,
CUDA JIT execution, two-rank image replies or model fidelity. Those device gates remain required.
