//! glue-check: `_hc_up_mix` and `_hc_wb_norm` against the launches they fuse, bytes compared and timed.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("cuda_triton.zig");
const prompt = @import("cuda_prompt.zig");
const samples = @import("cuda_prompt_check.zig");
const upMix = prompt.upMix;
const Fill = samples.Fill;
const fillBuf = samples.fillBuf;
const sample = samples.sample;

/// `glue-check`: `_hc_up_mix` against `_b16mm` + `_hc_mix` (mixed bytes) on random and edge inputs, timed.
pub fn glueCheck(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri) !bool {
    const D: usize = 2560;
    const S: usize = 4;
    const K: usize = 320;
    const max_m: usize = 4096;
    var prng = std.Random.DefaultPrng.init(0x61_7e);
    const r = prng.random();
    var act = try cuda.DeviceBuffer.alloc(d, max_m * K * 2);
    defer act.free();
    var w = try cuda.DeviceBuffer.alloc(d, S * D * K * 2);
    defer w.free();
    var normed = try cuda.DeviceBuffer.alloc(d, max_m * S * D * 2);
    defer normed.free();
    var up = try cuda.DeviceBuffer.alloc(d, max_m * S * D * 2);
    defer up.free();
    var m1 = try cuda.DeviceBuffer.alloc(d, max_m * D * 2);
    defer m1.free();
    var m2 = try cuda.DeviceBuffer.alloc(d, max_m * D * 2);
    defer m2.free();
    var xs = try cuda.DeviceBuffer.alloc(d, max_m * (D / 32) * 4);
    defer xs.free();
    const h1 = try gpa.alloc(u8, max_m * D * 2);
    defer gpa.free(h1);
    const h2 = try gpa.alloc(u8, max_m * D * 2);
    defer gpa.free(h2);
    var all = true;
    for ([_]Fill{ .normal, .wide, .edge, .special }) |fill| {
        try fillBuf(gpa, act, max_m * K, r, if (fill == .wide) .normal else fill);
        try fillBuf(gpa, w, S * D * K, r, if (fill == .special) .normal else fill);
        try fillBuf(gpa, normed, max_m * S * D, r, if (fill == .special) .normal else fill);
        for ([_]usize{ 16, 17, 129, 1024, 2048, 4096 }) |m| {
            for ([_]usize{ 64, 32 }) |bm| {
                try m1.fill8(0xA5, t.s.handle);
                try m2.fill8(0x5A, t.s.handle);
                try t.b16mm(act.ptr, K, w.ptr, up.ptr, false, 0, m, S * D, K);
                try t.hcMix(up.ptr, normed.ptr, m1.ptr, xs.ptr, m, D, S);
                try upMix(t, act.ptr, w.ptr, normed.ptr, m2.ptr, m, D, S, K, bm);
                try t.s.synchronize();
                try m1.download(0, h1[0 .. m * D * 2]);
                try m2.download(0, h2[0 .. m * D * 2]);
                const ok = std.mem.eql(u8, h1[0 .. m * D * 2], h2[0 .. m * D * 2]);
                if (!ok) all = false;
                std.debug.print("{s} _hc_up_mix BM {d} rows {d} {s}: mixed bytes\n", .{ if (ok) "EQUAL" else "DIFFER", bm, m, @tagName(fill) });
            }
        }
    }
    // speed at 2048 rows
    var e0 = try cuda.Event.init(d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(d, true);
    defer e1.deinit();
    var e2 = try cuda.Event.init(d, true);
    defer e2.deinit();
    for (0..2) |_| {
        try e0.record(t.s);
        for (0..10) |_| {
            try t.b16mm(act.ptr, K, w.ptr, up.ptr, false, 0, 2048, S * D, K);
            try t.hcMix(up.ptr, normed.ptr, m1.ptr, xs.ptr, 2048, D, S);
        }
        try e1.record(t.s);
        for (0..10) |_| try upMix(t, act.ptr, w.ptr, normed.ptr, m2.ptr, 2048, D, S, K, 64);
        try e2.record(t.s);
        try e2.synchronize();
    }
    const a = try cuda.Event.elapsedMs(e0, e1) / 10;
    const b = try cuda.Event.elapsedMs(e1, e2) / 10;
    std.debug.print("bench up+mix 2048 rows: _b16mm + _hc_mix {d:.3} ms, _hc_up_mix {d:.3} ms ({d:.2}x)\n", .{ a, b, a / b });
    return all;
}

/// bf16 samples as the fp32 values they stand for (the fp32 branch inputs).
fn fillF32(gpa: std.mem.Allocator, buf: cuda.DeviceBuffer, count: usize, r: std.Random, fill: Fill) !void {
    const h = try gpa.alloc(u32, count);
    defer gpa.free(h);
    for (h, 0..) |*v, i| v.* = @as(u32, sample(r, fill, i)) << 16;
    try buf.upload(0, std.mem.sliceAsBytes(h));
}

/// `glue-check`'s second part: `_hc_wb_norm` against `_hc_writeback` + `_hc_normed`, bytes, every branch kind.
pub fn wbNormCheck(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri) !bool {
    const D: usize = 2560;
    const S: usize = 4;
    const NC: usize = D / 256;
    const slots: usize = 11;
    const max_m: usize = 2048;
    const eps: f32 = 1e-6;
    var prng = std.Random.DefaultPrng.init(0x7b_0a);
    const r = prng.random();
    var h0 = try cuda.DeviceBuffer.alloc(d, max_m * S * D * 2);
    defer h0.free();
    var ha = try cuda.DeviceBuffer.alloc(d, max_m * S * D * 2);
    defer ha.free();
    var hb = try cuda.DeviceBuffer.alloc(d, max_m * S * D * 2);
    defer hb.free();
    var pa = try cuda.DeviceBuffer.alloc(d, max_m * NC * S * 4);
    defer pa.free();
    var pb = try cuda.DeviceBuffer.alloc(d, max_m * NC * S * 4);
    defer pb.free();
    var na = try cuda.DeviceBuffer.alloc(d, max_m * S * D * 2);
    defer na.free();
    var nb = try cuda.DeviceBuffer.alloc(d, max_m * S * D * 2);
    defer nb.free();
    var xs = try cuda.DeviceBuffer.alloc(d, max_m * S * D / 32 * 4);
    defer xs.free();
    var scale = try cuda.DeviceBuffer.alloc(d, S * D * 4);
    defer scale.free();
    var inj = try cuda.DeviceBuffer.alloc(d, max_m * S * 2);
    defer inj.free();
    var br16 = try cuda.DeviceBuffer.alloc(d, max_m * D * 2);
    defer br16.free();
    var yb = try cuda.DeviceBuffer.alloc(d, max_m * slots * D * 2);
    defer yb.free();
    var yf = try cuda.DeviceBuffer.alloc(d, max_m * slots * D * 4);
    defer yf.free();
    var wts = try cuda.DeviceBuffer.alloc(d, max_m * slots * 4);
    defer wts.free();
    var part = try cuda.DeviceBuffer.alloc(d, 2 * max_m * D * 4);
    defer part.free();
    const bytes = try gpa.alloc(u8, max_m * S * D * 2);
    defer gpa.free(bytes);
    const bytes2 = try gpa.alloc(u8, max_m * S * D * 2);
    defer gpa.free(bytes2);
    var all = true;
    for ([_]Fill{ .normal, .wide, .edge, .special }) |fill| {
        const calm: Fill = if (fill == .special) .normal else fill;
        try fillBuf(gpa, h0, max_m * S * D, r, fill);
        try fillF32(gpa, scale, S * D, r, if (fill == .wide) .normal else calm);
        try fillBuf(gpa, inj, max_m * S, r, calm);
        try fillBuf(gpa, br16, max_m * D, r, calm);
        try fillBuf(gpa, yb, max_m * slots * D, r, calm);
        try fillF32(gpa, yf, max_m * slots * D, r, calm);
        try fillF32(gpa, wts, max_m * slots, r, calm);
        try fillF32(gpa, part, 2 * max_m * D, r, calm);
        const kinds = [_]struct { name: []const u8, br: tri.Branch }{
            .{ .name = "none", .br = .none },
            .{ .name = "bf16", .br = .{ .bf16 = br16.ptr } },
            .{ .name = "moe bf16", .br = .{ .moe = .{ .y = yb.ptr, .y_f32 = false, .wts = wts.ptr, .slots = slots } } },
            .{ .name = "moe fp32", .br = .{ .moe = .{ .y = yf.ptr, .y_f32 = true, .wts = wts.ptr, .slots = slots } } },
            .{ .name = "ranks", .br = .{ .ranks = .{ .part = part.ptr, .world = 2 } } },
        };
        for ([_]usize{ 1, 16, 129, 1000, 2048 }) |m| {
            for (kinds) |kd| {
                const inject: ?u64 = if (kd.br == .none) null else inj.ptr;
                try ha.copyFrom(0, h0.ptr, m * S * D * 2, t.s.handle);
                try hb.copyFrom(0, h0.ptr, m * S * D * 2, t.s.handle);
                try pa.fill8(0xA5, t.s.handle);
                try pb.fill8(0x5A, t.s.handle);
                try na.fill8(0xA5, t.s.handle);
                try nb.fill8(0x5A, t.s.handle);
                try t.hcWriteback(ha.ptr, ha.ptr, pa.ptr, inject, kd.br, m, D, S);
                try t.hcNormed(ha.ptr, pa.ptr, scale.ptr, na.ptr, xs.ptr, m, D, S, eps);
                try t.hcWritebackNorm(hb.ptr, hb.ptr, pb.ptr, inject, kd.br, m, D, S, .{ .scale = scale.ptr, .normed = nb.ptr, .eps = eps });
                try t.s.synchronize();
                var ok = true;
                const parts = [_]struct { a: cuda.DeviceBuffer, b: cuda.DeviceBuffer, n: usize }{
                    .{ .a = ha, .b = hb, .n = m * S * D * 2 },
                    .{ .a = pa, .b = pb, .n = m * NC * S * 4 },
                    .{ .a = na, .b = nb, .n = m * S * D * 2 },
                };
                for (parts) |pt| {
                    try pt.a.download(0, bytes[0..pt.n]);
                    try pt.b.download(0, bytes2[0..pt.n]);
                    if (!std.mem.eql(u8, bytes[0..pt.n], bytes2[0..pt.n])) ok = false;
                }
                if (!ok) all = false;
                std.debug.print("{s} _hc_wb_norm {s} rows {d} {s}: streams, squared sums, normed bytes\n", .{ if (ok) "EQUAL" else "DIFFER", kd.name, m, @tagName(fill) });
            }
        }
    }
    // speed at 2048 rows (MoE bf16 branch)
    var e0 = try cuda.Event.init(d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(d, true);
    defer e1.deinit();
    var e2 = try cuda.Event.init(d, true);
    defer e2.deinit();
    const moe: tri.Branch = .{ .moe = .{ .y = yb.ptr, .y_f32 = false, .wts = wts.ptr, .slots = slots } };
    for (0..2) |_| {
        try e0.record(t.s);
        for (0..10) |_| {
            try t.hcWriteback(ha.ptr, ha.ptr, pa.ptr, inj.ptr, moe, 2048, D, S);
            try t.hcNormed(ha.ptr, pa.ptr, scale.ptr, na.ptr, xs.ptr, 2048, D, S, eps);
        }
        try e1.record(t.s);
        for (0..10) |_| try t.hcWritebackNorm(hb.ptr, hb.ptr, pb.ptr, inj.ptr, moe, 2048, D, S, .{ .scale = scale.ptr, .normed = nb.ptr, .eps = eps });
        try e2.record(t.s);
        try e2.synchronize();
    }
    const a = try cuda.Event.elapsedMs(e0, e1) / 10;
    const b = try cuda.Event.elapsedMs(e1, e2) / 10;
    std.debug.print("bench write-back + norm 2048 rows (moe bf16): _hc_writeback + _hc_normed {d:.3} ms, _hc_wb_norm {d:.3} ms ({d:.2}x)\n", .{ a, b, a / b });
    return all;
}
