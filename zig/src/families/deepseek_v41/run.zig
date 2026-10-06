//! M2's call executor: block.zig's calls issued on our own buffers, nothing seeded. The buffer plan (buffers.zig) is
//! bound once: a persistent role ("s.") gets its own allocation, the window and layer roles one zeroed arena at
//! 256-byte offsets (counters start at zero, as Python allocates them). Extension calls go through dsv41_kernels'
//! bindings; Triton calls through the engine's AOT set by name (triton_call.zig: the variant Triton would pick).
//! State (pools, rings, carries) and the window's glue fill these buffers from outside (state.zig, forward.zig).

const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const calls = @import("calls.zig");
const buffers = @import("buffers.zig");
const load = @import("load.zig");
const tc = @import("triton_call.zig");
const rp = dk.replay;

pub const Runner = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    kernels: *const dk.Kernels,
    triton: *const cuda.aot.Set,
    weights: *const load.Weights,
    /// role -> device address
    addrs: std.StringHashMapUnmanaged(u64) = .empty,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    scratch: ?cuda.DeviceBuffer = null,
    /// the forward's glue steps (forward.zig: positions, exchanges over the Collective, Engram rows, kit roundings)
    glue: ?Glue = null,

    pub const Glue = struct {
        ctx: *anyopaque,
        run: *const fn (ctx: *anyopaque, r: *Runner, c: *const calls.Call) anyerror!void,
    };

    pub fn deinit(r: *Runner) void {
        for (r.owned.items) |*b| b.free();
        r.owned.deinit(r.gpa);
        if (r.scratch) |*s| s.free();
        var it = r.addrs.keyIterator();
        while (it.next()) |k| r.gpa.free(k.*);
        r.addrs.deinit(r.gpa);
    }

    /// Allocates every role of `plan`: persistent roles one buffer each, the rest one zeroed arena.
    pub fn bind(r: *Runner, plan: *const buffers.Plan) !void {
        var arena_bytes: u64 = 0;
        for (plan.sizes.keys(), plan.sizes.values()) |k, v| {
            if (buffers.scopeOf(k) == .persistent) {
                const b = try cuda.DeviceBuffer.alloc(r.d, v);
                try b.fill8(0, null);
                try r.owned.append(r.gpa, b);
                try r.addrs.put(r.gpa, try r.gpa.dupe(u8, k), b.ptr);
            } else arena_bytes += std.mem.alignForward(u64, v, 256);
        }
        const arena = try cuda.DeviceBuffer.alloc(r.d, arena_bytes);
        try arena.fill8(0, null);
        try r.owned.append(r.gpa, arena);
        var at: u64 = 0;
        for (plan.sizes.keys(), plan.sizes.values()) |k, v| {
            if (buffers.scopeOf(k) == .persistent) continue;
            try r.addrs.put(r.gpa, try r.gpa.dupe(u8, k), arena.ptr + at);
            at += std.mem.alignForward(u64, v, 256);
        }
        r.scratch = try cuda.DeviceBuffer.alloc(r.d, 64 << 20);
    }

    /// A role's device address (state.zig and the glue write through it).
    pub fn addressOf(r: *const Runner, role: []const u8) ?u64 {
        return r.addrs.get(role);
    }

    pub fn tensorAddr(r: *const Runner, t: calls.Tensor) !u64 {
        var numel: i64 = 1;
        for (t.shape) |n| numel *= n;
        if (numel == 0) return 0;
        const off: u64 = @intCast(t.offset);
        return switch (t.role) {
            .weight => |w| (r.weights.get(w) orelse return error.MissingWeight).ptr + off,
            .buf => |b| (r.addrs.get(b) orelse return error.Unbound) + off,
            .empty => error.EmptyRole,
        };
    }

    fn addrOf(ctx: *anyopaque, t: calls.Tensor) anyerror!u64 {
        const r: *const Runner = @ptrCast(@alignCast(ctx));
        return r.tensorAddr(t);
    }

    fn extArg(r: *const Runner, a: std.mem.Allocator, x: calls.Arg) !rp.Arg {
        return switch (x) {
            .i => |v| .{ .int = v },
            .f => |v| .{ .float = v },
            .b => |v| .{ .boolean = v },
            .none => .none,
            .list => |items| blk: {
                const out = try a.alloc(rp.Arg, items.len);
                for (items, out) |y, *z| z.* = try r.extArg(a, y);
                break :blk .{ .list = out };
            },
            .t, .opaque_table => |t| .{ .tensor = .{ .ptr = try r.tensorAddr(t), .dtype = std.meta.stringToEnum(rp.DType, @tagName(t.dt)).?, .shape = t.shape, .stride = t.stride } },
        };
    }

    /// Issues one call on the runner's stream. The L2 prefetchers are skipped until their address tables are built
    /// (they change no bits; R-perf work).
    pub fn issue(r: *Runner, c: *const calls.Call) !void {
        if (std.mem.startsWith(u8, c.name, "tf_dsv41_l2p")) return;
        if (c.glue) {
            const g = r.glue orelse return error.NoGlue;
            return g.run(g.ctx, r, c);
        }
        var arena = std.heap.ArenaAllocator.init(r.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        if (c.triton) {
            var mine: std.ArrayList(cuda.aot.Spec) = .empty;
            for (r.triton.variants) |v| if (std.mem.eql(u8, v.spec.@"fn", c.name)) try mine.append(a, v.spec);
            const sp = try tc.split(a, c, mine.items, .{ .ctx = @constCast(r), .of = addrOf });
            const g = c.grid;
            return r.triton.run(r.stream, c.name, .{ @intCast(g[0]), @intCast(g[1]), @intCast(g[2]) }, sp.args, sp.consts);
        }
        const dot = std.mem.lastIndexOfScalar(u8, c.name, '.') orelse return error.BadName;
        const args = try a.alloc(rp.Arg, c.args.len);
        for (c.args, args) |x, *y| y.* = try r.extArg(a, x.arg);
        var scratch: rp.Scratch = .{ .ptr = r.scratch.?.ptr, .len = r.scratch.?.len };
        if (!try rp.call(r.kernels, r.stream, c.name[0..dot], c.name[dot + 1 ..], args, &.{}, &scratch)) return error.NoBinding;
    }

    /// Issues a window's calls in order.
    pub fn window(r: *Runner, cs: []const calls.Call) !void {
        for (cs) |*c| try r.issue(c);
    }
};
