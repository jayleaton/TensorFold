//! Rank 0's image rows (vision.py ``Encoder`` / ``Store`` / ``fill``): every live request's prepared images by
//! virtual id, the tower's span rows cached by digest (TF_DSV41_VISION_CACHE_MB, default 64), and the embedding
//! hook: a window's image positions take their rows in rank 0's embedding partial before the all-sum, so the other
//! ranks receive them through the exchange they already do (row + 0 is the row: exact). Image positions are past
//! every rank's vocabulary range, so embed_send already wrote zeros there on every rank.
//! ``Engine`` decorates the served engine: a request's ``images`` are held from submit to its finished event.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const held = @import("dsv41_serve").vision.held;
const tower_mod = @import("vision_tower.zig");

const Allocator = std.mem.Allocator;
const log = std.log.scoped(.dsv41);

pub const VBASE: u32 = 1 << 24;

pub fn isVid(t: u32) bool {
    return t >= VBASE;
}

/// Whether a window's ids hold an image position.
pub fn hasImage(ids: []const u32) bool {
    for (ids) |t| if (isVid(t)) return true;
    return false;
}

const Loc = struct { img: *const held.HeldImage, idx: u32, refs: u32 };
const Span = struct { buf: cuda.DeviceBuffer, stamp: u64 };

pub const Rows = struct {
    gpa: Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    tower: *tower_mod.Tower,
    mutex: std.Io.Mutex = .init,
    where: std.AutoHashMapUnmanaged(u32, Loc) = .empty,
    holds: std.AutoHashMapUnmanaged(api.Id, *const held.Held) = .empty,
    spans: std.AutoHashMapUnmanaged([32]u8, Span) = .empty,
    span_bytes: u64 = 0,
    cap: u64,
    clock: u64 = 0,
    patches: ?cuda.DeviceBuffer = null,

    pub fn init(gpa: Allocator, io: std.Io, d: *const cuda.Driver, tower: *tower_mod.Tower) Rows {
        const mb: u64 = if (std.c.getenv("TF_DSV41_VISION_CACHE_MB")) |v| std.fmt.parseInt(u64, std.mem.trim(u8, std.mem.span(v), " "), 10) catch 64 else 64;
        return .{ .gpa = gpa, .io = io, .d = d, .tower = tower, .cap = mb << 20 };
    }

    pub fn deinit(r: *Rows) void {
        var it = r.spans.valueIterator();
        while (it.next()) |s| s.buf.free();
        r.spans.deinit(r.gpa);
        r.where.deinit(r.gpa);
        var hs = r.holds.valueIterator();
        while (hs.next()) |h| r.free(h.*);
        r.holds.deinit(r.gpa);
        if (r.patches) |*p| p.free();
    }

    /// ``Store.hold``: the request's images' ids, until ``release`` (the same image twice in a request: one ref). The
    /// images are copied: the server may free its own before the engine is done (a client gone mid-prefill).
    pub fn hold(r: *Rows, id: api.Id, src: *const held.Held) !void {
        const h = try r.copy(src);
        errdefer r.free(h);
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        try r.holds.put(r.gpa, id, h);
        for (h.slice(), 0..) |*img, k| {
            if (seenBefore(h, k)) continue;
            for (img.vids[0..img.n_vids], 0..) |v, i| {
                const got = try r.where.getOrPut(r.gpa, v);
                if (got.found_existing) got.value_ptr.refs += 1 else got.value_ptr.* = .{ .img = img, .idx = @intCast(i), .refs = 1 };
            }
        }
    }

    fn copy(r: *Rows, src: *const held.Held) !*const held.Held {
        const imgs = try r.gpa.alloc(held.HeldImage, src.n);
        var done: usize = 0;
        errdefer {
            for (imgs[0..done]) |*im| freeImage(r.gpa, im);
            r.gpa.free(imgs);
        }
        for (src.slice(), imgs) |*a, *b| {
            b.* = a.*;
            const np = @as(usize, a.vit_h) * a.vit_w * 3 * 14 * 14;
            b.patches = (try r.gpa.dupe(u16, a.patches[0..np])).ptr;
            b.vids = (r.gpa.dupe(u32, a.vids[0..a.n_vids]) catch |e| {
                r.gpa.free(b.patches[0..np]);
                return e;
            }).ptr;
            done += 1;
        }
        const h = try r.gpa.create(held.Held);
        h.* = .{ .images = imgs.ptr, .n = imgs.len };
        return h;
    }

    fn freeImage(gpa: Allocator, im: *const held.HeldImage) void {
        gpa.free(im.patches[0 .. @as(usize, im.vit_h) * im.vit_w * 3 * 14 * 14]);
        gpa.free(im.vids[0..im.n_vids]);
    }

    fn free(r: *Rows, h: *const held.Held) void {
        for (h.slice()) |*im| freeImage(r.gpa, im);
        r.gpa.free(h.images[0..h.n]);
        r.gpa.destroy(h);
    }

    fn seenBefore(h: *const held.Held, k: usize) bool {
        for (h.slice()[0..k]) |*o| if (std.mem.eql(u8, &o.digest, &h.slice()[k].digest)) return true;
        return false;
    }

    /// ``Store.release``: a finished request's ids; an id another live request holds points at that request's copy.
    pub fn release(r: *Rows, id: api.Id) void {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        const h = (r.holds.fetchRemove(id) orelse return).value;
        defer r.free(h);
        for (h.slice(), 0..) |*img, k| {
            if (seenBefore(h, k)) continue;
            for (img.vids[0..img.n_vids]) |v| {
                const e = r.where.getPtr(v) orelse continue;
                e.refs -= 1;
                if (e.refs == 0) {
                    _ = r.where.remove(v);
                } else if (e.img == img) e.img = r.otherHolder(img.digest) orelse e.img;
            }
        }
    }

    fn otherHolder(r: *Rows, digest: [32]u8) ?*const held.HeldImage {
        var it = r.holds.valueIterator();
        while (it.next()) |h| for (h.*.slice()) |*img| if (std.mem.eql(u8, &img.digest, &digest)) return img;
        return null;
    }

    /// The image's span rows on the device (the tower's output), encoded on `s` when not cached.
    fn spanOf(r: *Rows, s: cuda.Stream, img: *const held.HeldImage) !u64 {
        r.clock += 1;
        if (r.spans.getPtr(img.digest)) |e| {
            e.stamp = r.clock;
            return e.buf.ptr;
        }
        const D: usize = r.tower.shape.out;
        const bytes = img.n_vids * D * 2;
        // the cache's oldest spans out first (the stream drained: a queued copy may still read one)
        if (r.span_bytes + bytes > r.cap and r.spans.count() > 0) try s.synchronize();
        while (r.span_bytes + bytes > r.cap and r.spans.count() > 0) {
            var oldest: ?[32]u8 = null;
            var best: u64 = std.math.maxInt(u64);
            var it = r.spans.iterator();
            while (it.next()) |e| if (e.value_ptr.stamp < best) {
                best = e.value_ptr.stamp;
                oldest = e.key_ptr.*;
            };
            var gone = r.spans.fetchRemove(oldest.?).?.value;
            r.span_bytes -= gone.buf.len;
            gone.buf.free();
        }
        const np: usize = @as(usize, img.vit_h) * img.vit_w * 3 * 14 * 14 * 2;
        if (r.patches == null or r.patches.?.len < np) {
            if (r.patches) |*p| {
                try s.synchronize();
                p.free();
            }
            r.patches = try cuda.DeviceBuffer.alloc(r.d, np);
        }
        try r.patches.?.uploadAsync(0, std.mem.sliceAsBytes(img.patches[0 .. np / 2]), s.handle);
        var out = try cuda.DeviceBuffer.alloc(r.d, bytes);
        errdefer out.free();
        try r.tower.span(s, img, r.patches.?.ptr, out.ptr);
        try r.spans.put(r.gpa, img.digest, .{ .buf = out, .stamp = r.clock });
        r.span_bytes += bytes;
        return out.ptr;
    }

    /// ``vision.fill``: the window's image positions' rows into `rows` (bf16 [ids.len, D], rank 0's embedding
    /// partial), one copy a run of consecutive span positions.
    pub fn fill(r: *Rows, s: cuda.Stream, ids: []const u32, rows: u64, D: usize) !void {
        if (D != r.tower.shape.out) return error.VisionWidth;
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        var i: usize = 0;
        while (i < ids.len) {
            if (!isVid(ids[i])) {
                i += 1;
                continue;
            }
            const loc = r.where.get(ids[i]) orelse {
                log.err("image rows not held for virtual id {d} (position {d})", .{ ids[i], i });
                return error.ImageRowsNotHeld;
            };
            var j = i + 1;
            while (j < ids.len and loc.idx + (j - i) < loc.img.n_vids and ids[j] == loc.img.vids[loc.idx + (j - i)]) j += 1;
            const base = try r.spanOf(s, loc.img);
            const row = D * 2;
            try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(rows + i * row, base + loc.idx * row, (j - i) * row, s.handle), "cuMemcpyDtoDAsync");
            i = j;
        }
    }
};

/// The served engine with rank 0's image rows: a request's ``images`` (``vision/held.zig``) are held from submit to
/// its finished event.
pub const Engine = struct {
    inner: api.Engine,
    rows: *Rows,
    gpa: Allocator,
    mutex: std.Io.Mutex = .init,
    sinks: std.AutoHashMapUnmanaged(api.Id, api.Sink) = .empty,

    pub fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn self(ctx: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(ctx: *anyopaque) api.Info {
        return self(ctx).inner.info();
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        self(ctx).inner.cancel(id);
    }
    fn status(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
        self(ctx).inner.status(out, stream_tokens);
    }
    fn memory(ctx: *anyopaque, reset_peak: bool) ?api.Memory {
        return self(ctx).inner.memory(reset_peak);
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = self(ctx);
        const h: *const held.Held = @ptrCast(@alignCast(request.images orelse return e.inner.submit(id, request, sink)));
        e.rows.hold(id, h) catch return error.Busy;
        {
            e.mutex.lockUncancelable(e.rows.io);
            defer e.mutex.unlock(e.rows.io);
            e.sinks.put(e.gpa, id, sink) catch {
                e.rows.release(id);
                return error.Busy;
            };
        }
        e.inner.submit(id, request, .{ .ctx = e, .event = onEvent }) catch |err| {
            e.drop(id);
            return err;
        };
    }

    fn drop(e: *Engine, id: api.Id) void {
        e.rows.release(id);
        e.mutex.lockUncancelable(e.rows.io);
        defer e.mutex.unlock(e.rows.io);
        _ = e.sinks.remove(id);
    }

    fn onEvent(ctx: *anyopaque, id: api.Id, event: *const api.Event) void {
        const e = self(ctx);
        const sink = blk: {
            e.mutex.lockUncancelable(e.rows.io);
            defer e.mutex.unlock(e.rows.io);
            break :blk e.sinks.get(id);
        } orelse return;
        const done = event.* == .finished;
        if (done) e.drop(id); // the images go before the client hears the end (its server then frees them)
        sink.event(sink.ctx, id, event);
    }
};

// -- the forward's and the model's hooks ------------------------------------------------------------------------------
const fwd = @import("forward.zig");
const run = @import("run.zig");
const calls = @import("calls.zig");
const block = @import("block.zig");
const pack_mod = @import("pack.zig");
const mode_mod = @import("dsv41_serve").vision.mode;

/// TF_DSV41_IMAGES at boot, every rank: native sets `f.images` (Engram's keep at image positions, the same program on
/// every rank) and, on rank 0, loads the tower and the rows (`f.vision`). The image routing bias (TF_DSV41_BIAS_VL)
/// needs the MoE's image rows as their own call (vision.moe), which the prefill emitter does not have yet: refused.
pub fn boot(gpa: Allocator, io: std.Io, f: *fwd.Forward, pack: *const pack_mod.Pack, d: *const cuda.Driver, rank: u32) !void {
    const raw: ?[]const u8 = if (std.c.getenv("TF_DSV41_IMAGES")) |v| std.mem.span(v) else null;
    const mode = mode_mod.parse(raw) orelse return error.BadImagesMode;
    if (mode != .native) return;
    if (std.c.getenv("TF_DSV41_BIAS_VL")) |v| if (std.mem.trim(u8, std.mem.span(v), " ").len > 0) {
        log.err("TF_DSV41_IMAGES=native with TF_DSV41_BIAS_VL: the image rows' own MoE call (vision.moe) is not in the Zig prefill yet; unset TF_DSV41_BIAS_VL (image rows then route with the text bias, as Python without it)", .{});
        return error.BiasVlNotPorted;
    };
    f.images = true;
    if (rank != 0) return;
    const t = try tower_mod.Tower.load(gpa, io, d, pack, .{});
    errdefer t.deinit();
    const r = try gpa.create(Rows);
    r.* = Rows.init(gpa, io, d, t);
    f.vision = r;
    // the gates' replays (tf-dsv41-m1 generate / m2b): the reference's prepared images held for the whole run
    if (std.c.getenv("TF_DSV41_VISION_HOLD")) |dir| try holdDir(r, gpa, io, std.mem.span(dir));
    log.info("images (TF_DSV41_IMAGES=native): tower {d:.2} GB on rank 0; image routing bias not loaded (image rows route with the text bias)", .{@as(f64, @floatFromInt(t.bytes)) / 1e9});
}

pub fn close(f: *fwd.Forward) void {
    const r = f.vision orelse return;
    const t = r.tower;
    r.deinit();
    r.gpa.destroy(r);
    t.deinit();
    f.vision = null;
}

/// The emitter's options for a segment of `ids`: Engram's keep when it holds an image position.
pub fn segmentOptions(f: *const fwd.Forward, ids: []const u32) block.Options {
    var o = f.opts;
    o.image_keep = f.images and hasImage(ids);
    return o;
}

/// The options the buffer plan sizes with: the keep buffer exists when images are on.
pub fn planOptions(f: *const fwd.Forward) block.Options {
    var o = f.opts;
    o.image_keep = f.images;
    return o;
}

/// Glue "image_keep" (vision.window's ``keep``): fp32 [n], 0 at image positions, 1 elsewhere.
pub fn keep(f: *fwd.Forward, r: *run.Runner, c: *const calls.Call) !void {
    const t = c.args[0].arg.t;
    const n = f.ids.len;
    if (@as(usize, @intCast(t.shape[0])) != n) return error.BadGlue;
    const host = try f.gpa.alloc(f32, n);
    defer f.gpa.free(host);
    for (f.ids, host) |id, *k| k.* = if (isVid(id)) 0.0 else 1.0;
    try r.stream.synchronize(); // the previous segment's Engram has read the buffer
    try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = try r.tensorAddr(t), .len = 4 * n }, 0, std.mem.sliceAsBytes(host));
}

/// The served engine wrapped so a request's images are held while it runs (rank 0 with the tower), else `inner`.
pub fn wrap(gpa: Allocator, f: *const fwd.Forward, inner: api.Engine) !api.Engine {
    const rows = f.vision orelse return inner;
    const e = try gpa.create(Engine);
    e.* = .{ .inner = inner, .rows = rows, .gpa = gpa };
    return e.engine();
}

/// The gate's held images: DIR/manifest.json (tools/zig/dsv41_vision/vision_ref.py) lists each image's grids, digest,
/// virtual ids and its bf16 patches file; they are held under id maxInt for the process's life.
pub const Manifest = struct {
    images: []const struct { name: []const u8, vit_h: u32, vit_w: u32, llm_h: u32, llm_w: u32, digest: []const u8, vids: []const u32, patches: []const u8 },
};

pub fn readManifest(a: Allocator, io: std.Io, dir: []const u8) !struct { m: Manifest, held: held.Held } {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "manifest.json" }), a, .limited(1 << 26));
    const m = try std.json.parseFromSliceLeaky(Manifest, a, text, .{ .ignore_unknown_fields = true });
    const imgs = try a.alloc(held.HeldImage, m.images.len);
    for (m.images, imgs) |e, *h| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, e.patches }), a, .limited(1 << 30));
        const np = @as(usize, e.vit_h) * e.vit_w * 3 * 14 * 14;
        if (bytes.len != np * 2) return error.BadManifest;
        const p = try a.alloc(u16, np);
        @memcpy(std.mem.sliceAsBytes(p), bytes);
        var dg: [32]u8 = undefined;
        _ = try std.fmt.hexToBytes(&dg, e.digest);
        h.* = .{ .patches = p.ptr, .vit_h = e.vit_h, .vit_w = e.vit_w, .llm_h = e.llm_h, .llm_w = e.llm_w, .digest = dg, .vids = e.vids.ptr, .n_vids = e.vids.len };
    }
    return .{ .m = m, .held = .{ .images = imgs.ptr, .n = imgs.len } };
}

fn holdDir(r: *Rows, gpa: Allocator, io: std.Io, dir: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit(); // hold() copies
    const got = try readManifest(arena.allocator(), io, dir);
    try r.hold(std.math.maxInt(api.Id), &got.held);
    log.info("images: {d} held from {s} (TF_DSV41_VISION_HOLD)", .{ got.held.n, dir });
}
