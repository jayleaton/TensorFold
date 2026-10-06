//! M3's integration on the GPU: the lanes engine (core/lanes + draft/lanes.zig, driven by draft/drive.zig) over the
//! GPU target (target.zig) on rank 0, the other ranks following its forward operations (forward.follow). The gate
//! passes are drive.zig's: `NoDraft` (drafts off: serial decoding through the lanes stack) and `Oracle` (the reference's
//! own tokens as drafts, corrupted on a schedule: multi-row verify windows and keeps of every kind), and the DSpark
//! GPU pass once it lands. Each gate's tokens must equal the reference's serial greedy tokens.

const std = @import("std");
const iface = @import("draft/iface.zig");
const dspark = @import("draft/dspark.zig");
const drive = @import("draft/drive.zig");
const fwd = @import("forward.zig");
const target_mod = @import("target.zig");

pub const Result = drive.Result;

/// DSpark's shape from the model config at this world (candidates: a rank's share of 128).
pub fn shapeOf(f: *const fwd.Forward) dspark.Shape {
    return .{ .block = f.cfg.dspark_block, .noise = f.cfg.dspark_noise_token, .window = f.cfg.window, .rank = f.cfg.dspark_markov_rank, .hidden = f.cfg.hidden, .candidates = 128 / f.comm.world() };
}

/// Rank 0: `max_new` tokens after `prompt` through lanes with `pass` (drafts on when `drafts`). The slot already holds
/// the prompt (the reference's state); `first` is the reference's choice after it.
/// `max_rows`: the largest bucket the forward's buffers were planned for.
pub fn generate(gpa: std.mem.Allocator, f: *fwd.Forward, pass: iface.Pass, drafts: bool, prompt: []const u32, first: u32, max_new: u32, max_rows: u32) !Result {
    var gt: target_mod.GpuTarget = .{ .f = f, .resume_first = first, .max_rows = max_rows };
    return drive.generate(gpa, gt.target(), pass, shapeOf(f), prompt, max_new, drafts);
}
