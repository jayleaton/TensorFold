//! aot-needs' prompt passes: the served rows, TF_DSV41_PF_4K's 2,048-row fallback, TF_DSV41_IDX_SPLIT's other passes.
const std = @import("std");
const block = @import("block.zig");
const Served = @import("aot_needs.zig").Served;

comptime {
    // CED's decoder pass (at most `keep` + 1 rows) never splits, so the split's extra passes leave it out
    std.debug.assert(@import("idx_split.zig").min_rows > @import("ced.zig").keep + 1);
}

/// The served prefill rows; under TF_DSV41_PF_4K also the forward's 2,048-row segments (piece fallback, first start).
pub fn promptPasses(sv: Served) [6]?Served {
    var two = sv;
    two.o.pf4k = false;
    two.o.prefill_rows = (block.Options{}).prefill_rows;
    var out: [6]?Served = .{ sv, if (sv.o.pf4k) two else null, null, null, null, null };
    if (sv.o.idx_split == .off or sv.o.world != 2) return out;
    // TF_DSV41_IDX_SPLIT: the other rank's own row blocks, and the replicated launches the same assets serve when off
    for (0..2) |i| if (out[i]) |x| {
        var y = x;
        y.replay = false;
        y.replay_runs = false;
        var other = y;
        other.o.rank = 1 - x.o.rank;
        out[2 + i] = other;
        y.o.idx_split = .off;
        out[4 + i] = y;
    };
    return out;
}
