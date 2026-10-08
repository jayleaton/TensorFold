//! M5: sessions on the GPU path (sessions.py / residency.py / sessdisk.py over the Zig forward). The family-neutral store
//! (zig/src/sessions: prefix entries, the RAM tier with its residency budget, the streamed NVMe tier) runs over the paged
//! pool (kv_state.zig); this file is DeepSeek's side of it:
//! - the slot's bounded state as the store snapshots it (`Snap`, store.Bounded): every backbone layer's SWA ring
//!   ("s.L<i>.swa.v" / ".s"), the ratio-2 carries ("s.L<i>.carry"), the position and the Engram tail. A RAM entry keeps
//!   a host copy; a park writes the same pieces as named blobs next to the entry's pages;
//! - `save`: an entry for the slot's committed ids (its pages shared, not copied), then the slot lets go of its pages;
//! - the served shape (own prefill, verify tail; Python prod's fast prefill tag): no turn entries; `savePrompt` keeps an
//!   entry at the prompt's replay point while the slot goes on prefilling (CED replay, ced.zig: its bounded state with
//!   the stash ring "s.ced.*" and the stash's positions and ids); entries carry the fast tag, replay's its own
//!   (ced.tag), so full and replay entries never resume each other, nor the old exact-tag turn entries;
//! - `resume`: the longest saved entry that strictly prefixes a prompt, from RAM (full pages adopted, the partial last one
//!   copied) or NVMe (streamed into fresh pages through the staging set), its bounded state uploaded: the slot continues at
//!   the entry's position, and the caller prefills the rest of the prompt;
//! - the residency budget (`Store.enforce`: idle chats / pool bytes held by entries alone) parks or drops the least
//!   recently used chat after every save.
//! Both ranks make the same calls in the same order (the leader's operations: forward.zig `op_sess_*`), so entries,
//! pages and evictions agree; a split rank parks and restores only the comp rows it owns.
//! Knobs as Python's: TF_DSV41_SESSIONS (0: no store), TF_DSV41_SESSION_DISK (the NVMe tier's directory; unset: RAM
//! only), TF_DSV41_SESSION_DISK_GIB (64: the tier's files, least recently used deleted past it),
//! TF_DSV41_SESSION_DISK_MIN (1024: shorter entries are dropped, not parked), TF_DSV41_SESSION_RESIDENT (idle chats
//! kept), TF_DSV41_SESSION_RESIDENT_GIB, TF_DSV41_SESSION_RAM_MIB, TF_DSV41_SESSION_STAGING_MIB.

const std = @import("std");
const cuda = @import("cuda");
const kv = @import("kv");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const kvs = @import("kv_state.zig");
const buffers = @import("buffers.zig");
const ced = @import("ced.zig");

const sessions = kv.sessions;
const Store = sessions.Store;
const Blob = sessions.disk.Blob;

pub const Error = error{ NoPool, BadBlob };

pub const Options = struct {
    /// TF_DSV41_SESSIONS (default 1, as stack.py)
    enabled: bool = true,
    disk: ?[]const u8 = null,
    disk_budget: u64 = 64 << 30,
    disk_min_tokens: u64 = 1024,
    resident: u32 = 0,
    resident_bytes: u64 = 0,
    ram_bytes: u64 = 512 << 20,
    staging_mib: u32 = 128,
};

pub fn optionsFromEnv() !Options {
    var o: Options = .{};
    if (std.c.getenv("TF_DSV41_SESSIONS")) |v| o.enabled = !std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "0");
    if (std.c.getenv("TF_DSV41_SESSION_DISK")) |v| if (std.mem.trim(u8, std.mem.span(v), " ").len > 0) {
        o.disk = std.mem.trim(u8, std.mem.span(v), " ");
    };
    if (std.c.getenv("TF_DSV41_SESSION_DISK_GIB")) |v| if (std.mem.span(v).len > 0) {
        o.disk_budget = @intFromFloat(try std.fmt.parseFloat(f64, std.mem.span(v)) * (1 << 30));
    };
    if (std.c.getenv("TF_DSV41_SESSION_DISK_MIN")) |v| if (std.mem.span(v).len > 0) {
        o.disk_min_tokens = try std.fmt.parseInt(u64, std.mem.span(v), 10);
    };
    if (std.c.getenv("TF_DSV41_SESSION_RESIDENT")) |v| o.resident = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (std.c.getenv("TF_DSV41_SESSION_RESIDENT_GIB")) |v| o.resident_bytes = @intFromFloat(try std.fmt.parseFloat(f64, std.mem.span(v)) * (1 << 30));
    if (std.c.getenv("TF_DSV41_SESSION_RAM_MIB")) |v| o.ram_bytes = (try std.fmt.parseInt(u64, std.mem.span(v), 10)) << 20;
    if (std.c.getenv("TF_DSV41_SESSION_STAGING_MIB")) |v| o.staging_mib = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    return o;
}

/// One device role of the bounded state.
const Piece = struct { role: []const u8, addr: u64, len: usize };

pub const Sessions = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    f: *fwd.Forward,
    kx: *kvs.Kv,
    store: Store,
    disk: ?*sessions.Disk = null,
    pieces: []Piece,
    bounded_bytes: u64,
    /// the slot's committed ids (the target appends what it keeps): a saved entry's ids
    history: std.ArrayList(i32) = .empty,
    names: std.heap.ArenaAllocator,

    /// The store over the forward's pool (after Runner.bind: the bounded roles have addresses).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, f: *fwd.Forward, plan: *const buffers.Plan, o: Options) !*Sessions {
        const kx = f.kv orelse return error.NoPool;
        const s = try gpa.create(Sessions);
        errdefer gpa.destroy(s);
        s.* = .{ .gpa = gpa, .io = io, .f = f, .kx = kx, .store = undefined, .pieces = &.{}, .bounded_bytes = 0, .names = std.heap.ArenaAllocator.init(gpa) };
        errdefer s.names.deinit();
        // the bounded roles in a fixed order (both ranks; a park's blobs carry their names)
        var list: std.ArrayList(Piece) = .empty;
        var la = std.heap.ArenaAllocator.init(gpa);
        defer la.deinit();
        for (try f.backbone(la.allocator())) |L| {
            for ([_][]const u8{ "swa.v", "swa.s", "carry" }) |part| {
                const role = try std.fmt.allocPrint(s.names.allocator(), "s.L{d}.{s}", .{ L, part });
                const len = plan.sizes.get(role) orelse continue;
                try list.append(s.names.allocator(), .{ .role = role, .addr = f.runner.addressOf(role) orelse return error.Unbound, .len = len });
                s.bounded_bytes += len;
            }
        }
        // CED replay: the stash ring (planned in replay mode only)
        for (ced.roles) |role| if (plan.sizes.get(role)) |len| {
            try list.append(s.names.allocator(), .{ .role = role, .addr = f.runner.addressOf(role) orelse return error.Unbound, .len = len });
            s.bounded_bytes += len;
        };
        s.pieces = list.items;
        if (o.disk) |root| {
            // the tier's compat ident: the families' layout, the split and this rank (a file is one rank's rows)
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            for (kx.layout.families()) |fam| h.update(std.mem.asBytes(&[_]u32{ fam.ratio, fam.row_bytes, @intFromBool(fam.split) }));
            h.update(std.mem.asBytes(&[_]u32{ kx.world, kx.rank, kx.pool.page }));
            var full: [32]u8 = undefined;
            h.final(&full);
            // the staging set (TF_DSV41_SESSION_STAGING_MIB) in 8 MiB chunks: a park's or restore's whole host transient
            s.disk = try sessions.Disk.open(gpa, io, .{ .root = root, .compat = full[0..16].*, .rank = kx.rank, .world = kx.world, .budget = o.disk_budget, .min_tokens = o.disk_min_tokens, .staging = @max(2, o.staging_mib / 8), .chunk = 8 << 20 });
        }
        errdefer if (s.disk) |d| d.close();
        s.store = try Store.init(gpa, io, &kx.pool, kx.dev.store(), s.disk, .{ .ram_bytes = o.ram_bytes, .chats = o.resident, .resident_bytes = o.resident_bytes });
        f.sess = .{ .ptr = s, .commit = commitFn, .save = saveFn, .restore = restoreFn, .reset = resetFn, .prompt = promptFn };
        return s;
    }

    fn commitFn(ptr: *anyopaque, ids: []const u32) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        for (ids) |t| try s.history.append(s.gpa, @intCast(t));
    }

    fn saveFn(ptr: *anyopaque) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        _ = try s.save();
    }

    fn promptFn(ptr: *anyopaque) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        _ = try s.savePrompt();
    }

    fn resetFn(ptr: *anyopaque) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        try s.reset();
    }

    fn restoreFn(ptr: *anyopaque, id: u32) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        try s.restore(id);
    }

    pub fn deinit(s: *Sessions) void {
        s.f.sess = null;
        s.store.deinit();
        if (s.disk) |d| d.close();
        s.history.deinit(s.gpa);
        s.names.deinit();
        s.gpa.destroy(s);
    }

    // -------------------------------------------------------------------------------------------------------------
    // the bounded state (store.Bounded)

    const Snap = struct {
        owner: *Sessions,
        bytes: []u8,
        pos: u64,
        tail: [eh.max_ngram]u32,
        tail_len: usize,
        stash: ced.Stash,

        fn bounded(p: *Snap) sessions.store.Bounded {
            return .{ .ptr = p, .vtable = &.{ .nbytes = nbytes, .blobs = blobs, .release = release } };
        }

        fn nbytes(ptr: *anyopaque) u64 {
            const p: *Snap = @ptrCast(@alignCast(ptr));
            return p.bytes.len;
        }

        /// The park's blobs: each role's bytes under its name, then "slot" (position, tail).
        fn blobs(ptr: *anyopaque, gpa: std.mem.Allocator) anyerror![]Blob {
            const p: *Snap = @ptrCast(@alignCast(ptr));
            const ps = p.owner.pieces;
            const out = try gpa.alloc(Blob, ps.len + 1);
            var at: usize = 0;
            for (ps, out[0..ps.len]) |pc, *b| {
                b.* = .{ .name = try gpa.dupe(u8, pc.role), .bytes = try gpa.dupe(u8, p.bytes[at..][0..pc.len]) };
                at += pc.len;
            }
            var slot: [slot_bytes]u8 = undefined;
            std.mem.writeInt(u64, slot[0..8], p.pos, .little);
            std.mem.writeInt(u64, slot[8..16], p.tail_len, .little);
            for (0..eh.max_ngram) |i| std.mem.writeInt(u32, slot[16 + 4 * i ..][0..4], if (i < p.tail_len) p.tail[i] else 0, .little);
            // CED: the stash's positions and ids (a full-mode entry's are zeros)
            const c = slot[ced_at..];
            std.mem.writeInt(u64, c[0..8], p.stash.from, .little);
            std.mem.writeInt(u64, c[8..16], p.stash.hi, .little);
            for (p.stash.ids, 0..) |t, i| std.mem.writeInt(u32, c[16 + 4 * i ..][0..4], t, .little);
            out[ps.len] = .{ .name = try gpa.dupe(u8, "slot"), .bytes = try gpa.dupe(u8, &slot) };
            return out;
        }

        fn release(ptr: *anyopaque) void {
            const p: *Snap = @ptrCast(@alignCast(ptr));
            const gpa = p.owner.gpa;
            gpa.free(p.bytes);
            gpa.destroy(p);
        }
    };

    /// The "slot" blob: position, tail length, the tail; then (CED) the stash's from / hi and its ids. Entries parked
    /// before the stash fields have the short form (their stash is empty).
    const ced_at: usize = 16 + 4 * eh.max_ngram;
    const slot_bytes: usize = ced_at + 16 + 4 * ced.ring;

    /// The slot's bounded state now (device to host, after the stream's work).
    fn snapshot(s: *Sessions) !*Snap {
        const r = s.f.runner;
        try r.stream.synchronize();
        const p = try s.gpa.create(Snap);
        errdefer s.gpa.destroy(p);
        p.* = .{ .owner = s, .bytes = try s.gpa.alloc(u8, s.bounded_bytes), .pos = s.f.slot.pos, .tail = s.f.slot.tail, .tail_len = s.f.slot.tail_len, .stash = s.f.slot.ced };
        var at: usize = 0;
        for (s.pieces) |pc| {
            try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = pc.addr, .len = pc.len }, 0, p.bytes[at..][0..pc.len]);
            at += pc.len;
        }
        return p;
    }

    fn uploadSnap(s: *Sessions, p: *const Snap) !void {
        const r = s.f.runner;
        var at: usize = 0;
        for (s.pieces) |pc| {
            try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = pc.addr, .len = pc.len }, 0, p.bytes[at..][0..pc.len]);
            at += pc.len;
        }
        s.f.slot.pos = p.pos;
        s.f.slot.tail = p.tail;
        s.f.slot.tail_len = p.tail_len;
        s.f.slot.ced = p.stash;
    }

    fn uploadBlobs(s: *Sessions, bs: []const Blob) !void {
        const r = s.f.runner;
        for (bs) |b| {
            if (std.mem.eql(u8, b.name, "slot")) {
                if (b.bytes.len < 16) return error.BadBlob;
                s.f.slot.pos = std.mem.readInt(u64, b.bytes[0..8], .little);
                s.f.slot.tail_len = @intCast(std.mem.readInt(u64, b.bytes[8..16], .little));
                for (0..eh.max_ngram) |i| s.f.slot.tail[i] = std.mem.readInt(u32, b.bytes[16 + 4 * i ..][0..4], .little);
                s.f.slot.ced = .{};
                if (b.bytes.len >= slot_bytes) {
                    const c = b.bytes[ced_at..];
                    s.f.slot.ced.from = std.mem.readInt(u64, c[0..8], .little);
                    s.f.slot.ced.hi = std.mem.readInt(u64, c[8..16], .little);
                    for (&s.f.slot.ced.ids, 0..) |*t, i| t.* = std.mem.readInt(u32, c[16 + 4 * i ..][0..4], .little);
                }
                continue;
            }
            const pc = for (s.pieces) |pc| {
                if (std.mem.eql(u8, pc.role, b.name)) break pc;
            } else return error.BadBlob;
            if (pc.len != b.bytes.len) return error.BadBlob;
            try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = pc.addr, .len = pc.len }, 0, b.bytes);
        }
    }

    // -------------------------------------------------------------------------------------------------------------
    // the operations (every rank, the leader's order)

    /// An entry for the slot's committed ids (null: a duplicate), then the slot's pages go back (the entry holds them).
    pub fn save(s: *Sessions) !?u32 {
        if (s.history.items.len != s.f.slot.pos or s.f.slot.pending != null) return error.BadWindow;
        try s.f.sendSess(.save, 0);
        if (s.history.items.len == 0) return null;
        const snap = try s.snapshot();
        const id = s.store.save(s.kx.slot, snap.bounded(), s.history.items, s.tag(), 0) catch |e| {
            Snap.release(snap);
            return e;
        };
        try s.reset();
        // the residency budget past the save: the least recently used chats parked (streamed) or dropped
        try s.store.settleAll();
        return id;
    }

    /// The served shape's prompt snapshot (batch.py `save_at`, rounds.py `_save(slot, "prompt")`): an entry for the committed
    /// ids at the slot's position (the prompt's replay point), its bounded state with the stash; the slot keeps its
    /// pages and goes on prefilling (it only writes positions past the entry's, which a resume rewrites before reading).
    pub fn savePrompt(s: *Sessions) !?u32 {
        if (s.history.items.len != s.f.slot.pos or s.f.slot.pending != null) return error.BadWindow;
        if (s.f.prefill_state.mode == .replay and s.f.slot.ced.hi != s.f.slot.pos) return error.BadWindow;
        try s.f.sendSess(.prompt, 0);
        if (s.history.items.len == 0) return null;
        const snap = try s.snapshot();
        const id = s.store.save(s.kx.slot, snap.bounded(), s.history.items, s.tag(), prompt_kind) catch |e| {
            Snap.release(snap);
            return e;
        };
        try s.store.settleAll();
        return id;
    }

    /// sessions.Entry.kind: a prompt's replay point (a turn's end is 0)
    const prompt_kind: u32 = 1;

    /// The entries' tag (sessions.Tag's CED field): replay entries never resume full ones, nor full ones replay's.
    fn tag(s: *const Sessions) u32 {
        return if (s.f.prefill_state.mode == .replay) ced.tag else ced.tag_fast;
    }

    /// The slot empty, its pages back (a session entry may hold them).
    pub fn reset(s: *Sessions) !void {
        try s.kx.release();
        s.f.slot = .{};
        s.history.clearRetainingCapacity();
    }

    /// The longest saved entry that strictly prefixes `prompt` into the (empty) slot: its pages and bounded state; the
    /// slot's position is the entry's (the caller prefills prompt[pos..]). 0: nothing found.
    pub fn resume_(s: *Sessions, prompt: []const u32) !u64 {
        if (s.f.slot.pos != 0 or s.kx.slot.len != 0) return error.SlotNotEmpty;
        const ids = try s.gpa.alloc(i32, prompt.len);
        defer s.gpa.free(ids);
        for (ids, prompt) |*d, t| d.* = @intCast(t);
        const hit = (try s.store.find(s.tag(), ids)) orelse return 0;
        if (hit.pos >= prompt.len) return 0; // a row must remain to prefill (strict prefixes only)
        try s.f.sendSess(.restore, hit.id);
        try s.restore(hit.id);
        return hit.pos;
    }

    /// Entry `id` into the empty slot (every rank: the leader found it): from RAM (pages shared, the partial last page
    /// copied) or NVMe (streamed into fresh pages), its bounded state and committed ids.
    pub fn restore(s: *Sessions, id: u32) !void {
        if (s.f.slot.pos != 0 or s.kx.slot.len != 0) return error.SlotNotEmpty;
        const e = s.store.entry(id);
        const pos = e.pos;
        if (e.ram) {
            const b = try s.store.restoreRam(id, s.kx.slot);
            try s.uploadSnap(@ptrCast(@alignCast(b.ptr)));
        } else {
            const j = try s.store.beginRestore(id, s.kx.slot);
            var restored = (try s.store.settle(j)) orelse return error.BadBlob;
            defer restored.deinit(s.gpa);
            try s.uploadBlobs(restored.blobs);
        }
        try s.kx.dev.syncTables();
        if (s.f.slot.pos != pos) return error.BadBlob;
        try s.history.resize(s.gpa, pos);
        try s.store.index.idsOf(id, s.history.items);
    }

    /// Every RAM entry parked to the NVMe tier now (the gate's interruption; a server parks through the budget).
    pub fn parkAll(s: *Sessions) !u32 {
        var n: u32 = 0;
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(s.gpa);
        var id = s.store.oldest;
        while (id != sessions.store.none) : (id = s.store.entry(id).extra.newer) try ids.append(s.gpa, id);
        for (ids.items) |e| if (try s.store.beginPark(e)) |j| {
            _ = try s.store.settle(j);
            n += 1;
        };
        return n;
    }
};
