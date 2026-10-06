//! DeepSeek-V4.1's other prod .cu kernels: CSA2 attention (attn_cuda) and its top-k (topk_cuda), the mHC boundary
//! (mhc_cuda, decode) and site (mhc_pf, prefill), the router GEMV (router_gemv), the fused prefill dense GEMM
//! (pfdense), the Engram gate copy and the L2 prefetchers. Args structs are the kernels' own (passed by value);
//! each launch mirrors its C++ entry's dispatch (instance, grid, block, shared memory, cluster) and checks.

const std = @import("std");
const cuda = @import("cuda");

fn dim(x: usize) u32 {
    return @intCast(x);
}

pub const Error = error{ Shape, Unsupported };

// ---------------------------------------------------------------------------------------------------------------
// attn_cuda.cu: dsv41_attn

pub const attn = struct {
    pub const NCH = 5;
    pub const BMQ = 16;
    pub const WINDOW = 128;
    pub const NT = 256;
    pub const SMEM: u32 = 98816;

    pub const Args = extern struct {
        q: u64 = 0,
        cv: u64 = 0,
        csc: u64 = 0,
        cvs: i64 = 0,
        css: i64 = 0,
        tok: u64 = 0,
        ts: i64 = 0,
        cnt: u64 = 0,
        sv: u64 = 0,
        ssc: u64 = 0,
        svs: i64 = 0,
        sss: i64 = 0,
        lo: u64 = 0,
        hi: u64 = 0,
        pos: u64 = 0,
        sl: u64 = 0,
        pt: u64 = 0,
        pts: i64 = 0,
        psh: c_int = 0,
        sink: u64 = 0,
        cs: u64 = 0,
        csst: i64 = 0,
        po: u64 = 0,
        pm: u64 = 0,
        pl: u64 = 0,
        out: u64 = 0,
        ticket: u64 = 0,
        R: c_int = 0,
        H: c_int = 0,
        ring: c_int = 0,
        has_comp: c_int = 0,
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "psh") == 144 and @offsetOf(Args, "sink") == 152 and @sizeOf(Args) == 232);
    }

    pub fn symbol(comptime kw: u32, comptime rows: bool, comptime hi: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN10dsv41_attn11attn_kernelILi{d}ELb{d}ELb{d}EEEvNS_4ArgsE", .{ kw, @intFromBool(rows), @intFromBool(hi) });
    }

    /// dispatch(): the instance index (kw 4 | rows | has_hi), after its checks.
    pub fn instance(a: Args, kw: u32, rows: bool, has_hi: bool) Error!usize {
        if (a.R < 1 or @rem(a.H, BMQ) != 0 or a.ring < WINDOW or (a.ring & (a.ring - 1)) != 0) return error.Shape;
        return @as(usize, if (kw == 4) 4 else 0) + @as(usize, if (rows) 2 else 0) + @intFromBool(has_hi);
    }
};

// topk_cuda.cu: dsv41_topk
pub const topk = struct {
    pub const NT = 512;
    pub const MAXCL = 8;
    pub const MAXEPT = 31;

    pub const Job = extern struct { s: u64 = 0, ss: i64 = 0, cand: u64 = 0, cs: i64 = 0, out: u64 = 0, os: i64 = 0, cnt: u64 = 0, nk: c_int = 0, k: c_int = 0, mode: c_int = 0, ept: c_int = 0 };
    pub const Args = extern struct { job: [2]Job, pos: u64 = 0, pos64: c_int = 0, ratio: c_int = 0, bs: c_int = 0 };

    comptime {
        std.debug.assert(@sizeOf(Job) == 72 and @offsetOf(Args, "pos") == 144 and @sizeOf(Args) == 168);
    }

    pub const symbol = "_ZN10dsv41_topk11topk_kernelENS_4ArgsE";

    pub fn smem(ept: usize) u32 {
        return @intCast(NT * ept * 4);
    }
};

// ---------------------------------------------------------------------------------------------------------------
// mhc_cuda.cu: dsv41_mhc (decode boundary, <= 16 rows)

pub const mhc = struct {
    pub const NB = 40;
    pub const NT = 128;
    pub const MAXR = 16;
    pub const ROWB = 128;

    pub const Args = extern struct {
        x: u64 = 0,
        xs: i64 = 0,
        xout: u64 = 0,
        g: u64 = 0,
        gr: i64 = 0,
        world: c_int = 1,
        gbf16: c_int = 0,
        post: u64 = 0,
        comb: u64 = 0,
        pre: u64 = 0,
        @"fn": u64 = 0,
        base: u64 = 0,
        scale: u64 = 0,
        part: u64 = 0,
        c: u64 = 0,
        tap: u64 = 0,
        ts: i64 = 0,
        nw: u64 = 0,
        nwbf16: c_int = 0,
        out: u64 = 0,
        opre: u64 = 0,
        opost: u64 = 0,
        ocomb: u64 = 0,
        cnt: u64 = 0,
        R: c_int = 0,
        eps: f32 = 0,
        hc_eps: f32 = 0,
        post_alpha: f32 = 0,
        iters: c_int = 0,
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "post") == 48 and @offsetOf(Args, "nwbf16") == 136 and @offsetOf(Args, "R") == 184 and @sizeOf(Args) == 208);
    }

    /// (POST, COLL, MIX, ALL4, FN32) of dispatch's cases, by mode * 2 + fn32 (mode 3 ignores fn32).
    pub const instances = [_]struct { post: bool, coll: u32, mix: bool, all4: bool, fn32: bool }{
        .{ .post = true, .coll = 2, .mix = true, .all4 = false, .fn32 = false },
        .{ .post = true, .coll = 2, .mix = true, .all4 = false, .fn32 = true },
        .{ .post = false, .coll = 1, .mix = true, .all4 = false, .fn32 = false },
        .{ .post = false, .coll = 1, .mix = true, .all4 = false, .fn32 = true },
        .{ .post = false, .coll = 2, .mix = true, .all4 = false, .fn32 = false },
        .{ .post = false, .coll = 2, .mix = true, .all4 = false, .fn32 = true },
        .{ .post = true, .coll = 2, .mix = false, .all4 = true, .fn32 = false },
    };

    pub fn symbol(comptime i: usize) [:0]const u8 {
        const s = instances[i];
        return std.fmt.comptimePrint("_ZN9dsv41_mhc15boundary_kernelILb{d}ELi{d}ELb{d}ELb{d}ELb{d}EEEvNS_4ArgsE", .{ @intFromBool(s.post), s.coll, @intFromBool(s.mix), @intFromBool(s.all4), @intFromBool(s.fn32) });
    }

    /// dispatch(): mode 0 boundary, 1 site from stream 0, 2 site with a carried pre-mix, 3 final.
    pub fn instance(a: Args, mode: u32, fn32: bool) Error!usize {
        if (a.R < 1 or a.R > MAXR or a.world < 1 or a.world > 8) return error.Shape;
        if (mode == 3) return 6;
        if (mode > 2) return error.Unsupported;
        return mode * 2 + @intFromBool(fn32);
    }
};

// mhc_pf.cu: dsv41_mhc_pf (prefill site, 16 rows a CTA)
pub const mhc_pf = struct {
    pub const NB = 40;
    pub const NT = 128;
    pub const BM = 16;
    pub const MAXW = 8;

    pub const Args = extern struct {
        x: u64 = 0,
        xs: i64 = 0,
        xout: u64 = 0,
        g: u64 = 0,
        gr: i64 = 0,
        world: c_int = 1,
        gbf16: c_int = 0,
        post: u64 = 0,
        comb: u64 = 0,
        pre: u64 = 0,
        @"fn": u64 = 0,
        part: u64 = 0,
        c: u64 = 0,
        tap: u64 = 0,
        ts: i64 = 0,
        R: c_int = 0,
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "R") == 112 and @sizeOf(Args) == 120);
    }

    /// Lay<FN32>::BYTES: staged streams, the collapsed rows and fn.
    pub fn smem(fn32: bool) u32 {
        const pb = 128 + 8;
        const pf = 128 + 4;
        const xs = 4 * BM * pb * 2;
        const cs = BM * pb * 2;
        const fnb: u32 = if (fn32) 4 * 24 * pf * 4 else 4 * 24 * pb * 2;
        return xs + cs + fnb;
    }

    /// (POST, COLL, FN32, TAP) by dispatch's case (mode * 4 + fn32 * 2 + tap).
    pub const instances = [_]struct { case: u32, post: bool, coll: u32, fn32: bool, tap: bool }{
        .{ .case = 0, .post = true, .coll = 2, .fn32 = false, .tap = false },
        .{ .case = 1, .post = true, .coll = 2, .fn32 = false, .tap = true },
        .{ .case = 2, .post = true, .coll = 2, .fn32 = true, .tap = false },
        .{ .case = 3, .post = true, .coll = 2, .fn32 = true, .tap = true },
        .{ .case = 4, .post = false, .coll = 1, .fn32 = false, .tap = false },
        .{ .case = 6, .post = false, .coll = 1, .fn32 = true, .tap = false },
        .{ .case = 8, .post = false, .coll = 2, .fn32 = false, .tap = false },
        .{ .case = 10, .post = false, .coll = 2, .fn32 = true, .tap = false },
    };

    pub fn symbol(comptime i: usize) [:0]const u8 {
        const s = instances[i];
        return std.fmt.comptimePrint("_ZN12dsv41_mhc_pf11site_kernelILb{d}ELi{d}ELb{d}ELb{d}EEEvNS_4ArgsE", .{ @intFromBool(s.post), s.coll, @intFromBool(s.fn32), @intFromBool(s.tap) });
    }

    pub fn instance(a: Args, mode: u32, fn32: bool, tap: bool) Error!usize {
        if (a.R < 1 or a.world < 1 or a.world > MAXW or (tap and mode != 0)) return error.Shape;
        const case = mode * 4 + @as(u32, if (fn32) 2 else 0) + @intFromBool(tap);
        for (instances, 0..) |s, i| if (s.case == case) return i;
        return error.Unsupported;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// router_gemv.cu: dsv41_rg

pub const rg = struct {
    pub const D = 5120;
    pub const EPL = 16;
    pub const KMAX = 8;
    pub const GE_MAX = 512;
    pub const rts = [_]u32{ 1, 2, 4, 8, 16 };
    pub const wide = [_][2]u32{ .{ 8, 1 }, .{ 4, 1 }, .{ 8, 2 }, .{ 4, 2 } };
    pub const narrow = [_]u32{ 1, 2, 4, 8 };

    pub const Args = extern struct {
        x: u64 = 0,
        xs: i64 = 0,
        w: u64 = 0,
        bias: u64 = 0,
        pick: u64 = 0,
        wts: u64 = 0,
        lg: u64 = 0,
        cnt: u64 = 0,
        R: c_int = 0,
        E: c_int = 0,
        K: c_int = 0,
        slots: c_int = 0,
        select: c_int = 0,
        scale: f32 = 0,
        kit: c_int = 0,
        gid: u64 = 0,
        gcnt: u64 = 0,
        gmem: u64 = 0,
        GE: c_int = 0,
        maxm: c_int = 0,
        group: c_int = 0,
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "gid") == 96 and @offsetOf(Args, "group") == 128 and @sizeOf(Args) == 136);
    }

    pub fn gemvSymbol(comptime rt: u32, comptime nw: u32, comptime ew: u32) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN8dsv41_rg11gemv_kernelILi{d}ELi{d}ELi{d}EEEvNS_4ArgsE", .{ rt, nw, ew });
    }

    pub fn narrowSymbol(comptime rt: u32, comptime nw: u32) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN8dsv41_rg13narrow_kernelILi{d}ELi{d}EEEvNS_4ArgsE", .{ rt, nw });
    }

    /// dsv41_rg_route_cuda's checks (row stride / alignment of x and the buffer sizes are the caller's).
    pub fn check(a: Args, rt: u32, nw: u32, ew: u32, lg_len: usize, cnt_len: usize) Error!void {
        const E: usize = @intCast(a.E);
        const R: usize = @intCast(a.R);
        if (E < 1 or E > 32 * EPL or a.K < 1 or a.K > KMAX or a.K > a.E or a.slots < a.K or a.slots > a.K + 1) return error.Shape;
        if (lg_len < R * E) return error.Shape;
        if (a.select != 0 and cnt_len < (R + rt - 1) / rt) return error.Shape;
        const is_wide = (nw == 4 or nw == 8) and (ew == 1 or ew == 2);
        const is_narrow = ew == 0 and (nw == 1 or nw == 2 or nw == 4 or nw == 8);
        if (std.mem.indexOfScalar(u32, &rts, rt) == null or !(is_wide or is_narrow)) return error.Unsupported;
        if (a.group != 0 and (a.select == 0 or !is_narrow or R > rt or a.GE < 1 or a.GE > GE_MAX)) return error.Shape;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// pfdense.cuh: dsv41_pfd (fused prefill dense GEMM, one fatbin a width)

pub const pfd = struct {
    pub const k2s = [_]u32{ 8, 10, 12, 16 };
    /// DSV41_PFD_CFGS: (BM, WM, WN, KS, NST) by id.
    pub const cfgs = [_][5]u32{ .{ 128, 2, 4, 2, 3 }, .{ 128, 2, 4, 2, 4 }, .{ 128, 4, 2, 2, 3 }, .{ 64, 2, 2, 2, 4 }, .{ 64, 1, 4, 2, 4 }, .{ 128, 2, 4, 2, 2 }, .{ 32, 1, 4, 2, 4 }, .{ 256, 4, 2, 2, 3 } };
    pub const OutType = enum(c_int) { f16 = 0, bf16 = 1, f32 = 2 };

    pub const Args = extern struct {
        xh: u64 = 0,
        T: u64 = 0,
        stride_k: i64 = 0,
        stride_nb: i64 = 0,
        svh: u64 = 0,
        bias: u64 = 0,
        out: u64 = 0,
        o_stride: i64 = 0,
        M: c_int = 0,
        K: c_int = 0,
        N: c_int = 0,
        group: c_int = 0,
        out_type: c_int = 0,
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "M") == 64 and @sizeOf(Args) == 88);
    }

    pub fn threads(cfg: usize) u32 {
        return cfgs[cfg][1] * cfgs[cfg][2] * 32;
    }

    /// Lay<K2, LANES, Cfg>::SMEM (the layout does not depend on LANES).
    pub fn smem(cfg: usize, k2: u32) u32 {
        const bm, const wm, _, const ks, const nst = cfgs[cfg];
        const a_bytes = bm * (16 * ks) * 2;
        const w_bytes = ks * 8 * (4 * k2) * 4;
        const ring = nst * (a_bytes + w_bytes);
        const main = ring + 2 * (ks * 8 * 32 * 16);
        const epi = (bm / wm) * 128 * 2 * 2;
        return @max(main, epi);
    }

    pub fn symbol(comptime k2: u32, comptime lanes: bool, comptime cfg: usize) [:0]const u8 {
        const c = cfgs[cfg];
        return std.fmt.comptimePrint("_ZN9dsv41_pfd10pfd_kernelILi{d}ELb{d}ENS_3CfgILi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEEEvNS_4ArgsE", .{ k2, @intFromBool(lanes), c[0], c[1], c[2], c[3], c[4] });
    }

    pub fn dequantSymbol(comptime k2: u32, comptime lanes: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN9dsv41_pfd18pfd_dequant_kernelILi{d}ELb{d}EEEvPKjxxP6__halfi", .{ k2, @intFromBool(lanes) });
    }

    /// pfdense.cpp gemm's checks on the shapes and strides (dtype / alignment are the caller's).
    pub fn check(a: Args, k2: u32, lanes: bool, cfg: usize, words: usize) Error!void {
        if (std.mem.indexOfScalar(u32, &k2s, k2) == null or (lanes and k2 == 16)) return error.Unsupported;
        if (cfg >= cfgs.len) return error.Unsupported;
        const ks: i64 = cfgs[cfg][3];
        const K: i64 = a.K;
        const N: i64 = a.N;
        if (a.M < 1 or @rem(K, 128) != 0 or K < 128 or @rem(K, 16 * ks) != 0) return error.Shape;
        if (@rem(N, 128) != 0 or N < 128 or @rem(a.o_stride, 2) != 0 or a.o_stride < N) return error.Shape;
        const tw: i64 = 4 * @as(i64, k2);
        if (@rem(a.stride_k, 4) != 0 or @rem(a.stride_nb, 4) != 0 or a.stride_k < 8 * tw or a.stride_nb < 8 * tw) return error.Shape;
        const last = (@divTrunc(K, 16) - 1) * a.stride_k + (@divTrunc(N, 128) - 1) * a.stride_nb + 8 * tw;
        if (last > @as(i64, @intCast(words))) return error.Shape;
        if (a.group < 1 or a.group > 1024) return error.Shape;
    }

    /// pfdense.heuristic (no sweep table): the largest row tile whose last wave is not mostly empty.
    pub fn heuristic(k: usize, n: usize, k2: u32, m: usize, sms: usize) [2]usize {
        const nb = n / 128;
        var best: ?usize = null;
        var best_eff: f64 = -1.0;
        for ([_]usize{ 0, 3, 6 }) |cfg| {
            if (k % (16 * cfgs[cfg][3]) != 0) continue;
            const bm = cfgs[cfg][0];
            const blocks: usize = if (2 * (smem(cfg, k2) + 1024) <= 102400) 2 else 1;
            const ctas = ((m + bm - 1) / bm) * nb;
            const slots = sms * blocks;
            const eff = @as(f64, @floatFromInt(ctas)) / @as(f64, @floatFromInt(((ctas + slots - 1) / slots) * slots));
            if (eff > best_eff + 0.10) {
                best = cfg;
                best_eff = eff;
            }
        }
        return .{ best orelse 0, 8 };
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Loaded functions and launches

pub const Functions = struct {
    attn: [8]cuda.Function,
    topk: cuda.Function,
    mhc: [mhc.instances.len]cuda.Function,
    mhc_pf: [mhc_pf.instances.len]cuda.Function,
    gemv: [rg.rts.len][rg.wide.len]cuda.Function,
    narrow: [rg.rts.len][rg.narrow.len]cuda.Function,
    pfd: [pfd.k2s.len][2][pfd.cfgs.len]?cuda.Function, // [k2][lanes][cfg]; K2 16 has no lanes kernels
    pfd_dequant: [pfd.k2s.len][2]?cuda.Function,
    gate_copy: cuda.Function,
    paced: cuda.Function,
    segments: [2]cuda.Function, // bulk false / true
    topk_keys: cuda.Function,

    /// `pf`: the four pfdense modules (K2 8, 10, 12, 16).
    pub fn resolve(m_attn: cuda.Module, m_topk: cuda.Module, m_mhc: cuda.Module, m_mhc_pf: cuda.Module, m_rg: cuda.Module, pf: [4]cuda.Module, m_gate: cuda.Module, m_pace: cuda.Module, m_l2pf: cuda.Module, m_keys: cuda.Module) !Functions {
        var f: Functions = undefined;
        inline for (0..8) |i| f.attn[i] = try m_attn.function(attn.symbol(if (i >= 4) 4 else 2, (i & 2) != 0, (i & 1) != 0));
        for (f.attn) |fun| try fun.allowDynamicShared(attn.SMEM);
        f.topk = try m_topk.function(topk.symbol);
        try f.topk.allowDynamicShared(topk.smem(topk.MAXEPT));
        inline for (0..mhc.instances.len) |i| f.mhc[i] = try m_mhc.function(mhc.symbol(i));
        inline for (0..mhc_pf.instances.len) |i| {
            f.mhc_pf[i] = try m_mhc_pf.function(mhc_pf.symbol(i));
            try f.mhc_pf[i].allowDynamicShared(mhc_pf.smem(mhc_pf.instances[i].fn32));
        }
        inline for (rg.rts, 0..) |rt, i| {
            inline for (rg.wide, 0..) |c, j| f.gemv[i][j] = try m_rg.function(rg.gemvSymbol(rt, c[0], c[1]));
            inline for (rg.narrow, 0..) |nw, j| f.narrow[i][j] = try m_rg.function(rg.narrowSymbol(rt, nw));
        }
        inline for (pfd.k2s, 0..) |k2, ki| inline for (.{ false, true }, 0..) |lanes, li| {
            inline for (0..pfd.cfgs.len) |c| {
                if (lanes and k2 == 16) {
                    f.pfd[ki][li][c] = null;
                } else {
                    const fun = try pf[ki].function(pfd.symbol(k2, lanes, c));
                    try fun.allowDynamicShared(pfd.smem(c, k2));
                    f.pfd[ki][li][c] = fun;
                }
            }
            f.pfd_dequant[ki][li] = if (lanes and k2 == 16) null else try pf[ki].function(pfd.dequantSymbol(k2, lanes));
        };
        f.gate_copy = try m_gate.function("_ZN17dsv41_engram_gate16gate_copy_kernelEPxiiPKhxPhxxi");
        f.paced = try m_pace.function("_ZN12dsv41_l2pace12paced_kernelEPKlijmm");
        f.segments = .{ try m_l2pf.function("_ZN6tfl2pf15segments_kernelILb0EEEvPKlij"), try m_l2pf.function("_ZN6tfl2pf15segments_kernelILb1EEEvPKlij") };
        f.topk_keys = try m_keys.function("_ZN15dsv41_topk_keys11topk_kernelEPKfxiiPfPx");
        return f;
    }
};

/// topk_keys.cu: pick.top's bounds (C columns a row within pick.COLS, k within the shared sort).
pub const topk_keys = struct {
    pub const max_cols = 1 << 16;
    pub const max_k = 1024;
};

pub const Ops = struct {
    f: *const Functions,
    s: cuda.Stream,

    fn go(o: Ops, f: cuda.Function, grid: [3]usize, block: usize, shared: u32, cluster: ?cuda.Dim3, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = dim(grid[0]), .y = dim(grid[1]), .z = dim(grid[2]) }, .block = .{ .x = dim(block) }, .shared = shared, .cluster = cluster }, o.s, a);
    }

    /// dsv41_attn_run_cuda: `rows` = the slots table is given (sl), `has_hi` = window ends are given; kw: the Triton
    /// layout's k width (attn_cuda.py reads it from Triton: 4 on Triton 3.8).
    pub fn attention(o: Ops, a: attn.Args, kw: u32, rows: bool, has_hi: bool) !void {
        const i = try attn.instance(a, kw, rows, has_hi);
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(o.f.attn[i], .{ attn.NCH, @intCast(@divTrunc(a.H, attn.BMQ)), @intCast(a.R) }, attn.NT, attn.SMEM, null, &args);
    }

    /// dsv41_topk_run_cuda: `jobs` 1-2 selections, R rows, CL CTAs a cluster, ept entries a thread.
    pub fn topK(o: Ops, a: topk.Args, jobs: usize, R: usize, CL: usize, ept: usize) !void {
        if (R < 1 or jobs < 1 or jobs > 2 or CL < 1 or CL > topk.MAXCL or ept < 1 or ept > topk.MAXEPT) return error.Shape;
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(o.f.topk, .{ CL, R, jobs }, topk.NT, topk.smem(ept), .{ .x = dim(CL) }, &args);
    }

    /// dsv41_mhc_run_cuda: `fn32` = fn is fp32.
    pub fn mhcBoundary(o: Ops, a: mhc.Args, mode: u32, fn32: bool) !void {
        const i = try mhc.instance(a, mode, fn32);
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(o.f.mhc[i], .{ mhc.NB, if (mhc.instances[i].all4) 1 else 4, 1 }, mhc.NT, @intCast(@as(u32, @intCast(a.R)) * mhc.ROWB), null, &args);
    }

    /// dsv41_mhc_pf_run_cuda: `tap` = a tap buffer is given.
    pub fn mhcSite(o: Ops, a: mhc_pf.Args, mode: u32, fn32: bool, tap: bool) !void {
        const i = try mhc_pf.instance(a, mode, fn32, tap);
        var args: cuda.Args = .{};
        args.add(a);
        const R: usize = @intCast(a.R);
        try o.go(o.f.mhc_pf[i], .{ mhc_pf.NB, (R + mhc_pf.BM - 1) / mhc_pf.BM, 1 }, mhc_pf.NT, mhc_pf.smem(mhc_pf.instances[i].fn32), null, &args);
    }

    /// dsv41_rg_route_cuda: (rt, nw, ew) a wide instance, or ew 0 the narrow kernel. The narrow kernel's shared
    /// memory carveout preference (speed only) is left at the driver default.
    pub fn route(o: Ops, a: rg.Args, rt: u32, nw: u32, ew: u32, lg_len: usize, cnt_len: usize) !void {
        try rg.check(a, rt, nw, ew, lg_len, cnt_len);
        if (a.R == 0) return;
        const ri = std.mem.indexOfScalar(u32, &rg.rts, rt).?;
        const E: usize = @intCast(a.E);
        const R: usize = @intCast(a.R);
        var args: cuda.Args = .{};
        args.add(a);
        if (ew == 0) {
            const ni = std.mem.indexOfScalar(u32, &rg.narrow, nw).?;
            return o.go(o.f.narrow[ri][ni], .{ (E + nw - 1) / nw, (R + rt - 1) / rt, 1 }, nw * 32, 0, null, &args);
        }
        var wi: usize = 0;
        for (rg.wide, 0..) |c, j| if (c[0] == nw and c[1] == ew) {
            wi = j;
        };
        try o.go(o.f.gemv[ri][wi], .{ (E + nw * ew - 1) / (nw * ew), (R + rt - 1) / rt, 1 }, nw * 32, 0, null, &args);
    }

    /// pfdense.cpp gemm: out [M, N] = xh [M, K] (rotated, fp16) through the trellis decoded in the kernel.
    pub fn pfdGemm(o: Ops, a: pfd.Args, k2: u32, lanes: bool, cfg: usize, words: usize) !void {
        try pfd.check(a, k2, lanes, cfg, words);
        const ki = std.mem.indexOfScalar(u32, &pfd.k2s, k2).?;
        const f = o.f.pfd[ki][@intFromBool(lanes)][cfg] orelse return error.Unsupported;
        const bm = pfd.cfgs[cfg][0];
        const M: usize = @intCast(a.M);
        const N: usize = @intCast(a.N);
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(f, .{ ((M + bm - 1) / bm) * (N >> 7), 1, 1 }, pfd.threads(cfg), pfd.smem(cfg, k2), null, &args);
    }

    /// pfdense.cpp dequant: W_q [K, N] fp16 through the kernel's decode (tests).
    pub fn pfdDequant(o: Ops, T: u64, stride_k: i64, stride_nb: i64, out: u64, K: usize, N: usize, k2: u32, lanes: bool) !void {
        const ki = std.mem.indexOfScalar(u32, &pfd.k2s, k2) orelse return error.Unsupported;
        const f = o.f.pfd_dequant[ki][@intFromBool(lanes)] orelse return error.Unsupported;
        var args: cuda.Args = .{};
        args.add(T);
        args.add(stride_k);
        args.add(stride_nb);
        args.add(out);
        args.add(@as(c_int, @intCast(N)));
        try o.go(f, .{ K / 32, N / 128, 1 }, 128, 0, null, &args);
    }

    /// dsv41_engram_gate_copy: wait (bounded) for the host's ready word, then copy the selected buffer.
    pub fn engramGate(o: Ops, ctl: u64, ready: c_int, err_slot: c_int, base: u64, stride: i64, dst: u64, nbytes: i64, timeout_ns: i64, vec: bool) !void {
        const units: i64 = if (vec) (nbytes + 15) >> 4 else nbytes;
        const blocks: usize = @intCast(std.math.clamp(@divTrunc(units + 255, 256), 1, 64));
        var args: cuda.Args = .{};
        args.add(ctl);
        args.add(ready);
        args.add(err_slot);
        args.add(base);
        args.add(stride);
        args.add(dst);
        args.add(nbytes);
        args.add(timeout_ns);
        args.add(@as(c_int, @intFromBool(vec)));
        try o.go(o.f.gate_copy, .{ blocks, 1, 1 }, 256, 0, null, &args);
    }

    /// dsv41_l2pace launch_paced: (address, bytes) table prefetched to L2 at a paced rate on `ctas` warps.
    pub fn l2Paced(o: Ops, table: u64, n: usize, ctas: usize, chunk: u32, ns_per_piece: u64, delay_ns: u64) !void {
        var args: cuda.Args = .{};
        args.add(table);
        args.add(@as(c_int, @intCast(n)));
        args.add(chunk);
        args.add(ns_per_piece);
        args.add(delay_ns);
        try o.go(o.f.paced, .{ ctas, 1, 1 }, 32, 0, null, &args);
    }

    /// pick.top (topk_keys.cu): each of `rows` rows' best k of bf16-valued fp32 logits [rows, C] (row stride `ld`),
    /// value descending then lower column; vals fp32 [rows, k], cols int64 [rows, k]. Logits that are not
    /// bf16-valued take Python's stable-sort path instead: not this kernel.
    pub fn topKeys(o: Ops, lg: u64, ld: usize, rows: usize, C: usize, k: usize, vals: u64, cols: u64) !void {
        if (rows < 1 or C < 1 or C > topk_keys.max_cols or k < 1 or k > @min(C, topk_keys.max_k)) return error.Shape;
        var args: cuda.Args = .{};
        args.add(lg);
        args.add(@as(i64, @intCast(ld)));
        args.add(@as(c_int, @intCast(C)));
        args.add(@as(c_int, @intCast(k)));
        args.add(vals);
        args.add(cols);
        try o.go(o.f.topk_keys, .{ rows, 1, 1 }, 1024, 0, null, &args);
    }

    /// tfl2pf launch_segments: (address, bytes) table prefetched to L2 (bulk: cp.async.bulk.prefetch).
    pub fn l2Segments(o: Ops, table: u64, n: usize, grid: usize, threads: usize, chunk: u32, bulk: bool) !void {
        var args: cuda.Args = .{};
        args.add(table);
        args.add(@as(c_int, @intCast(n)));
        args.add(chunk);
        try o.go(o.f.segments[@intFromBool(bulk)], .{ grid, 1, 1 }, threads, 0, null, &args);
    }
};

test "pfdense shared memory and heuristic as pfdense.py" {
    // cfg 0 (128, 2, 4, 2, 3) at K2 10: 3 x (128 x 32 x 2 + 2 x 8 x 40 x 4) + 2 x 8192 = 3 x 10752 + 16384
    try std.testing.expectEqual(@as(u32, 3 * 10752 + 16384), pfd.smem(0, 10));
    try std.testing.expectEqual(@as(u32, 256), pfd.threads(0));
    // a 2,048-row [5120 -> 1024] call on 48 SMs: 16 x 8 = 128 CTAs of BM 128
    const h = pfd.heuristic(5120, 1024, 10, 2048, 48);
    try std.testing.expectEqual(@as(usize, 8), h[1]);
}

test "mhc instances follow dispatch" {
    var a: mhc.Args = .{ .R = 3 };
    try std.testing.expectEqual(@as(usize, 6), try mhc.instance(a, 3, true));
    try std.testing.expectEqual(@as(usize, 3), try mhc.instance(a, 1, true));
    a.R = 17;
    try std.testing.expectError(error.Shape, mhc.instance(a, 0, false));
    const p: mhc_pf.Args = .{ .R = 40 };
    try std.testing.expectEqual(@as(usize, 5), try mhc_pf.instance(p, 1, true, false));
    try std.testing.expectError(error.Shape, mhc_pf.instance(p, 1, false, true));
    try std.testing.expectEqual(@as(u32, 47872), mhc_pf.smem(false));
}
