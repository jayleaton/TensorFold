//! Flash Next forward helpers: the glue split and exchanges, matmul dispatch, positions and pinned uploads.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("cuda_triton.zig");
const tops = @import("cuda_torch_ops.zig");
const state = @import("cuda_state.zig");
const profs = @import("cuda_prof.zig");
const prompt = @import("cuda_prompt.zig");
const fp8 = @import("cuda_fp8.zig");
const int4 = @import("cuda_int4.zig");
const W = @import("cuda_weights.zig");
const fwd = @import("cuda_forward.zig");
const Forward = fwd.Forward;
const Bufs = fwd.Bufs;
const Split = fwd.Split;
const fp4SerialRows = fwd.fp4SerialRows;
const split_min_rows = fwd.split_min_rows;

// -- small helpers ----------------------------------------------------------------------------------------

/// The row split of a prompt chunk's glue at two ranks (null: unsplit).
pub fn splitOf(f: *const Forward, x: *const Bufs, R: usize) ?Split {
    const cm = f.comm orelse return null;
    if (!f.glue_split or !x.b.prefill or cm.world != 2 or R < split_min_rows) return null;
    // a multiple of 16 rows a rank: every row offset keeps the 16-byte alignment the kernels were specialized on
    const rh = std.mem.alignForward(usize, (R + 1) / 2, 16);
    if (2 * rh > x.b.rows) return null;
    const r: usize = cm.rank;
    return .{ .rank = r, .rh = rh, .o = r * rh, .po = (1 - r) * rh };
}

/// Every rank's rows [rank * rh, (rank + 1) * rh) of `buf` ([2 rh, row] elements of `dtype`) gathered in place.
pub fn gatherRows(f: *Forward, buf: u64, sp: Split, row_bytes: usize, dtype: cuda.nccl.DataType, elem: usize) !void {
    const cm = f.comm.?;
    const n = sp.rh * row_bytes;
    // in place (NCCL's form): RoCE refuses overlapping buffers, so these stay on NCCL at every size
    try cm.allGatherInPlace(buf + sp.o * row_bytes, buf, n / elem, dtype, f.s.handle);
}

pub fn nextEvent(f: *Forward) cuda.Event {
    f.cs_i = (f.cs_i + 1) % f.cs_ev.len;
    return f.cs_ev[f.cs_i];
}

/// Exchange `count` elements with the peer on the comm stream (`exchangeWait` joins it), or in line without one.
pub fn exchangeAsync(f: *Forward, send: u64, recv: u64, count: usize, dtype: cuda.nccl.DataType) !void {
    const cm = f.comm.?;
    const cs = f.cs orelse return cm.exchange(send, recv, count, dtype, f.s.handle);
    const e = f.nextEvent();
    try e.record(f.s);
    try cs.wait(e);
    try cm.exchange(send, recv, count, dtype, cs.handle);
}

pub fn exchangeWait(f: *Forward) !void {
    const cs = f.cs orelse return;
    const e = f.nextEvent();
    try e.record(cs);
    try f.s.wait(e);
}

/// The GPU time since the previous mark is `part`'s (profiling only).
pub fn mark(f: *Forward, part: profs.Part) !void {
    if (f.prof) |p| try p.mark(f.s, part);
}

pub fn mm(f: *Forward, x: u64, x_stride: usize, r: W.Rows, out: u64, fp32: bool, m: usize) !void {
    try f.mmAt(null, x, x_stride, r, out, fp32, m);
}

/// `mm`, its time marked as `part` when profiling (a split's `_reduce` apart, as `.reduce`).
pub fn mmAt(f: *Forward, part: ?profs.Part, x: u64, x_stride: usize, r: W.Rows, out: u64, fp32: bool, m: usize) !void {
    if (prompt.b16Takes(f.prompt_mm, m, r.n, r.k)) {
        try prompt.b16(f.t, x, x_stride, r.weight, out, fp32, m, r.n, r.k);
    } else if (f.prof != null and part != null and tri.b16SplitK(r.n, r.k) > 1 and f.t.rec == null) {
        try prompt.b16Slices(f.t, x, x_stride, r.weight, f.sc.part, fp32, m, r.n, r.k);
        try f.mark(part.?);
        try prompt.reduce(f.t, f.sc.part, out, fp32, m * r.n, tri.b16SplitK(r.n, r.k));
        try f.mark(.reduce);
        return;
    } else try f.t.b16mm(x, x_stride, r.weight, out, fp32, f.sc.part, m, r.n, r.k);
    if (part) |pp| try f.mark(pp);
}

/// A linear on its kernel: block FP8 (`l8`) or bf16 rows, marked as `part` when profiling.
pub fn denseAt(f: *Forward, part: profs.Part, x: u64, x_stride: usize, xs: u64, r: W.Rows, l8: ?fp8.Linear, out: u64, fp32: bool, m: usize) !void {
    if (l8) |l| {
        try fp8.matmul(f.k8 orelse return error.NoFp8Kernels, f.s, x, x_stride, l, out, fp32, m);
        return f.mark(part);
    }
    return f.mmxAt(part, x, x_stride, xs, r, out, fp32, m);
}

/// INT4-AutoRound projections: block-FP8 columns (`l8`) and bf16 rows (`rb`) into `out`'s columns (Python Concat).
pub fn projFace(f: *Forward, part: profs.Part, x: u64, x_stride: usize, l8: ?fp8.Linear, rb: W.Rows, out: u64, out_w: usize, m: usize) !void {
    const l = l8 orelse return f.mmAt(part, x, x_stride, rb, out, false, m);
    // block-FP8 columns stored straight into `out` (fn_qmmf_ld; TF_FLASHNEXT_FP8_LD=0: via scratch, same bytes)
    if (f.fp8_ld) {
        try fp8.matmulLd(f.k8 orelse return error.NoFp8Kernels, f.s, x, x_stride, l, out, out_w, m);
    } else {
        try fp8.matmul(f.k8 orelse return error.NoFp8Kernels, f.s, x, x_stride, l, f.sc.p8, false, m);
        try f.th.slotCopy(f.sc.p8, l.n * 2, out, out_w * 2, l.n * 2, m);
    }
    try f.mmAt(null, x, x_stride, rb, f.sc.pb, false, m);
    try f.th.slotCopy(f.sc.pb, rb.n * 2, out + l.n * 2, out_w * 2, rb.n * 2, m);
    if (l.n + rb.n != out_w) return error.ProjectionWidth;
    try f.mark(part);
}

/// The head's logits of `m` rows: bf16 rows, or INT4-AutoRound's GPTQ int4 columns.
pub fn headMm(f: *Forward, x: u64, x_stride: usize, out: u64, m: usize) !void {
    if (f.w.head4) |h| return int4.dense(f.k4 orelse return error.NoInt4Kernels, f.s, x, x_stride, h, out, false, m);
    return f.mm(x, x_stride, f.w.head, out, false, m);
}

/// `fp4At`'s matmul on `t`'s stream, unmarked (the side stream's).
pub fn fp4On(f: *Forward, t: tri.Tri, th: tops.Torch, x: u64, x_stride: usize, fp: tri.Fp4, out: u64, fp32: bool, m: usize, n: usize, k: usize) !void {
    if (f.fp4Serial(fp, x_stride, m, n, k)) {
        try th.fp4Serial(x, x_stride, fp.weight, fp.scale, out, fp32, m, n, k);
    } else if (prompt.fp4Takes(f.prompt_mm, m, n, k)) {
        try prompt.fp4(t, x, x_stride, fp, out, fp32, m, n, k);
    } else try t.fp4mm(x, x_stride, fp, out, fp32, f.sc.part, m, n, k);
}

pub fn fp4At(f: *Forward, part: profs.Part, x: u64, x_stride: usize, fp: tri.Fp4, out: u64, fp32: bool, m: usize, n: usize, k: usize) !void {
    try f.fp4On(f.t, f.th, x, x_stride, fp, out, fp32, m, n, k);
    try f.mark(part);
}

/// Whether `_fp4mm`'s single-slice launch runs fn_ops' K-serial kernel (TF_FLASHNEXT_FP4_SERIAL=1, same bits).
pub fn fp4Serial(f: *const Forward, fp: tri.Fp4, x_stride: usize, m: usize, n: usize, k: usize) bool {
    return f.fp4_serial and !fp.codes and m > 8 and m <= fp4SerialRows(n) and tri.fp4SplitK(n, k) == 1 and tops.fp4SerialFits(x_stride, n, k);
}

/// mm with group sums `xs` known: a matrix with a 4-bit mtp_q4 copy runs qmm, every other the bf16 `mm`.
pub fn mmx(f: *Forward, x: u64, x_stride: usize, xs: u64, r: W.Rows, out: u64, fp32: bool, m: usize) !void {
    if (xs != 0) if (f.mtp_q4) |t| if (t.find(r.weight)) |q| return f.ops.qmm32Split(x, x_stride, xs, q, out, m, fp32);
    try f.mm(x, x_stride, r, out, fp32, m);
}

/// mmAt (the profiled part) with the 4-bit MTP copy as `mmx`.
pub fn mmxAt(f: *Forward, part: ?profs.Part, x: u64, x_stride: usize, xs: u64, r: W.Rows, out: u64, fp32: bool, m: usize) !void {
    if (xs != 0) if (f.mtp_q4) |t| if (t.find(r.weight)) |q| {
        try f.ops.qmm32Split(x, x_stride, xs, q, out, m, fp32);
        if (part) |p| try f.mark(p);
        return;
    };
    try f.mmAt(part, x, x_stride, r, out, fp32, m);
}

pub fn setPos(f: *Forward, st: *state.State, pos: usize) !void {
    st.pos = pos;
    try f.th.fill32(st.pos_dev, @intCast(pos), 1);
}

pub fn setMtpLen(f: *Forward, st: *state.State, n: usize) !void {
    st.mtp_len = n;
    try f.th.fill32(st.mtp_pos, @intCast(n), 1);
}

/// The DeltaNet state of linear layer `li` (one buffer: kept rows are folded in place).
pub fn rec(f: *const Forward, st: *const state.State, li: usize) u64 {
    return st.rec + li * f.g.nv * state.gdn_dv * state.gdn_dk * 4;
}

pub fn conv(f: *const Forward, st: *const state.State, li: usize) u64 {
    return st.conv + li * (f.g.conv_kernel - 1) * f.g.gdnConv() * 2;
}

/// The upload of `bytes` pinned bytes at `at` to `dst`, on the stream.
pub fn upload(f: *Forward, dst: u64, at: usize, bytes: usize) !void {
    if (bytes == 0) return;
    try f.d.check(f.d.api.cuMemcpyHtoDAsync_v2(dst, f.pin.bytes[at..].ptr, bytes, f.s.handle), "cuMemcpyHtoDAsync");
}
