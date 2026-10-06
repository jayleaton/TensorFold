//! M3's first integration: the lanes engine (core/lanes + draft/lanes.zig) over the GPU target (target.zig) on rank 0,
//! the other ranks following its forward operations (forward.follow). Until the DSpark GPU pass lands, streams run
//! with drafts off (`NoDraft` is the pass lanes needs and never calls), so a generation is serial decoding through
//! the whole lanes stack: the gate is its tokens == the reference's greedy tokens (M2b's), at TP=2.

const std = @import("std");
const lanes = @import("lanes");
const iface = @import("draft/iface.zig");
const ln = @import("draft/lanes.zig");
const dspark = @import("draft/dspark.zig");
const costs_mod = @import("draft/costs.zig");
const fwd = @import("forward.zig");
const target_mod = @import("target.zig");

/// The pass lanes holds while streams run with drafts off: it is never asked.
pub const NoDraft = struct {
    pub fn pass(x: *NoDraft) iface.Pass {
        return .{ .ptr = x, .vtable = &vtable };
    }
    const vtable: iface.Pass.VTable = .{ .ingest = ingest, .propose = propose, .reset = reset, .markov = markov };
    fn ingest(_: *anyopaque, _: u32, _: u64, _: iface.Taps, _: []const u32) anyerror!void {}
    fn propose(_: *anyopaque, _: []const iface.Ask, _: []iface.Proposal) anyerror!void {
        return error.NoDrafter;
    }
    fn reset(_: *anyopaque, _: u32) void {}
    fn markov(_: *anyopaque) ?dspark.Markov {
        return null;
    }
};

/// Rank 0: `max_new` tokens after `prompt` through lanes (drafts off). The target resumes the slot as it is when it
/// already holds the prompt (M2b: the reference's state), its first choice `first`.
pub fn generate(gpa: std.mem.Allocator, f: *fwd.Forward, prompt: []const u32, first: u32, max_new: u32) ![]u32 {
    var gt: target_mod.GpuTarget = .{ .f = f, .resume_first = first };
    var nd: NoDraft = .{};
    var c = try costs_mod.defaults(gpa, 64);
    defer c.deinit(gpa);
    const shape: dspark.Shape = .{ .block = f.cfg.dspark_block, .noise = f.cfg.dspark_noise_token, .window = f.cfg.window, .rank = f.cfg.dspark_markov_rank, .hidden = f.cfg.hidden, .candidates = 128 / f.comm.world() };
    const x = try ln.Lanes.init(gpa, gt.target(), nd.pass(), c, .{ .shape = shape, .slots = 1 });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    var s = try lanes.Stream.init(gpa, .{ .id = "m3", .prompt = prompt, .max_new = max_new, .drafts = false });
    defer s.deinit(gpa);
    try e.addStream(&s);
    while (e.activeCount() > 0) try e.step();
    return gpa.dupe(u32, s.emitted());
}
