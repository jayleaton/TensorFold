//! Where `--drafter` points: the checkpoint's own `dspark/` folder by default, else a folder holding the draft's files.
const std = @import("std");

/// The draft's folder inside the checkpoint (revision 0fbb6795 of the pack ships it there).
pub const default_name = "dspark";

/// Values that name the default.
pub fn isDefault(value: []const u8) bool {
    return value.len == 0 or std.mem.eql(u8, value, "auto") or std.mem.eql(u8, value, "dspark") or std.mem.eql(u8, value, "checkpoint");
}

pub const Error = error{NoDraft} || std.mem.Allocator.Error;

fn hasConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(gpa, &.{ dir, "config.json" }) catch return false;
    defer gpa.free(path);
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

/// The draft's folder (gpa-owned) for --drafter's value: the default name, a folder, or one inside the checkpoint.
pub fn resolve(gpa: std.mem.Allocator, io: std.Io, checkpoint: []const u8, value: []const u8) Error![]u8 {
    if (isDefault(value)) {
        const path = try std.fs.path.join(gpa, &.{ checkpoint, default_name });
        if (hasConfig(gpa, io, path)) return path;
        gpa.free(path);
        return error.NoDraft;
    }
    if (hasConfig(gpa, io, value)) return gpa.dupe(u8, value);
    const inside = try std.fs.path.join(gpa, &.{ checkpoint, value });
    if (hasConfig(gpa, io, inside)) return inside;
    gpa.free(inside);
    return error.NoDraft;
}

/// The words of a failed resolve.
pub fn words(buf: []u8, checkpoint: []const u8, value: []const u8) []const u8 {
    if (isDefault(value)) return std.fmt.bufPrint(buf, "--drafter: {s}/{s} holds no draft (config.json missing): this checkpoint revision ships none, or give --drafter the folder of a dspark draft", .{ checkpoint, default_name }) catch buf[0..0];
    return std.fmt.bufPrint(buf, "--drafter: {s} holds no draft (no config.json there or in {s}/{s}); use --drafter dspark for the checkpoint's own", .{ value, checkpoint, value }) catch buf[0..0];
}

test "the default value names the checkpoint's dspark folder; a path or a name inside the checkpoint works too; nothing else does" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);
    try tmp.dir.createDirPath(io, "ck/dspark");
    try tmp.dir.createDirPath(io, "ck/other");
    try tmp.dir.createDirPath(io, "elsewhere");
    try tmp.dir.writeFile(io, .{ .sub_path = "ck/dspark/config.json", .data = "{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "ck/other/config.json", .data = "{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "elsewhere/config.json", .data = "{}" });
    const ck = try std.fs.path.join(gpa, &.{ root, "ck" });
    defer gpa.free(ck);
    const elsewhere = try std.fs.path.join(gpa, &.{ root, "elsewhere" });
    defer gpa.free(elsewhere);
    for ([_][]const u8{ "", "auto", "dspark", "checkpoint" }) |v| {
        const got = try resolve(gpa, io, ck, v);
        defer gpa.free(got);
        try std.testing.expect(std.mem.endsWith(u8, got, "ck/dspark"));
    }
    const own = try resolve(gpa, io, ck, elsewhere);
    defer gpa.free(own);
    try std.testing.expectEqualStrings(elsewhere, own);
    const inside = try resolve(gpa, io, ck, "other");
    defer gpa.free(inside);
    try std.testing.expect(std.mem.endsWith(u8, inside, "ck/other"));
    try std.testing.expectError(error.NoDraft, resolve(gpa, io, ck, "missing"));
    try std.testing.expectError(error.NoDraft, resolve(gpa, io, elsewhere, "dspark")); // no dspark/ in that folder
    var w: [400]u8 = undefined;
    try std.testing.expect(std.mem.indexOf(u8, words(&w, "/m", "dspark"), "/m/dspark holds no draft") != null);
    try std.testing.expect(std.mem.indexOf(u8, words(&w, "/m", "x/y"), "x/y holds no draft") != null);
}
