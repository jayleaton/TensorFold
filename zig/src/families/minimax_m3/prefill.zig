//! MiniMax-M3.1-Flash's prompt pass: page chunks, marks, a resume and the first token, over any engine `host`.
const std = @import("std");
const iface = @import("iface.zig");
const pool = @import("pool.zig");
const plan = @import("forward_plan.zig");

pub const Error = error{ PromptTooLong, BadResume, BadChunk, Cancelled };

pub fn cancelled(c: ?iface.Cancel) bool {
    const k = c orelse return false;
    return k.check(k.ptr);
}

/// Where a prompt pass of `len` ids starts: 0, or the page-multiple the adopted pages already hold.
pub fn begin(len: usize, capacity: usize, from: ?iface.Resume) Error!usize {
    if (len == 0 or len + iface.window_rows >= capacity) return error.PromptTooLong;
    const r = from orelse return 0;
    if (r.at == 0 or r.at >= len or r.at % pool.page_tokens != 0) return error.BadResume;
    return r.at;
}

/// The end of the chunk that starts at `start`: a full chunk, the prompt's end, or the next mark inside it.
pub fn chunkEnd(start: usize, len: usize, marks: ?iface.Marks) usize {
    var end = @min(len, start + iface.prefill_rows);
    if (marks) |m| for (m.at) |at| {
        if (at > start and at < end) end = at;
    };
    return end;
}

/// The chunk boundaries of a pass, for tests and for a follower that wants them ahead.
pub const Chunks = struct {
    at: usize,
    len: usize,
    marks: ?iface.Marks = null,

    pub fn next(c: *Chunks) ?struct { start: usize, end: usize } {
        if (c.at >= c.len) return null;
        const end = chunkEnd(c.at, c.len, c.marks);
        defer c.at = end;
        return .{ .start = c.at, .end = end };
    }
};

/// One chunk of a `len`-id pass: `part` is its rows, plus the next id after a full chunk (GLM's follower sends it).
pub fn prefillChunk(host: anytype, part: []const u32, start: usize, len: usize, from: ?iface.Resume, cancel: ?iface.Cancel, marks: ?iface.Marks) !?u32 {
    if ((from != null or start == 0) and try begin(len, host.capacity(), from) != start) return error.BadChunk;
    if (start >= len or start % pool.page_tokens != 0) return error.BadChunk;
    const rows = if (start + part.len == len or part.len != iface.prefill_rows + 1) part.len else iface.prefill_rows;
    if (rows == 0 or rows > iface.prefill_rows or start + rows > len) return error.BadChunk;
    if (start + rows < len and rows % pool.page_tokens != 0) return error.BadChunk;
    if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
    try host.gate(cancel);
    const last = start + rows == len;
    const ids: []const i32 = @ptrCast(part[0..rows]);
    try host.chunk(ids, start, if (last) plan.Head.last else plan.Head.none);
    const first: ?u32 = if (last) try host.first(len) else null;
    try host.advance(rows);
    return first;
}

/// The whole prompt: from the resume point (or 0) to the end; returns the first token.
pub fn prefill(host: anytype, ids: []const u32, from: ?iface.Resume, cancel: ?iface.Cancel, marks: ?iface.Marks) !u32 {
    var start = try begin(ids.len, host.capacity(), from);
    while (true) {
        const end = chunkEnd(start, ids.len, marks);
        if (try prefillChunk(host, ids[start..end], start, ids.len, null, cancel, marks)) |first| return first;
        start = end;
    }
}
