//! DeepSeek contract images use the same built artifacts as their named installs.
const std = @import("std");
const list = @import("dsv41_units.zig");
const grammar = @import("dsv41_grammar.zig");
pub fn images(b: *std.Build, runner: *std.Build.Step.Compile, install: *std.Build.Step, kernels: anytype, files: []const ?std.Build.LazyPath, prebuilt: ?[]const u8, nvcc: ?[]const u8, version: ?std.Build.LazyPath, sms: []const u8, cuobjdump: ?[]const u8) void {
    addNames(b, runner.root_module);
    runner.root_module.addAnonymousImport("contract_fatbin_" ++ grammar.image_name, .{ .root_source_file = grammar.image(b) });
    const empty = b.addWriteFiles().add("empty-contract-deepseek.fatbin", "");
    for (list.units ++ list.optional) |k| {
        const common = for (kernels, 0..) |known, i| {
            if (std.mem.eql(u8, known.name, k.name)) break i;
        } else null;
        const file = if (common) |i| files[i] else if (prebuilt) |dir|
            b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }))
        else if (nvcc) |tool|
            @import("cuda.zig").fatbin(b, tool, version.?, k, sms)
        else
            null;
        runner.root_module.addAnonymousImport(b.fmt("contract_fatbin_{s}", .{k.name}), .{ .root_source_file = file orelse empty });
        if (common != null) continue;
        if (file) |path| {
            install.dependOn(&b.addInstallFile(path, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
            if (std.mem.eql(u8, k.name, "dsv41_x3ld")) @import("contract_symbols.zig").check(b, path, sms, install, cuobjdump);
        }
    }
}
pub fn inert(b: *std.Build, m: *std.Build.Module) void {
    addNames(b, m);
    const files = b.addWriteFiles();
    m.addAnonymousImport("contract_fatbin_" ++ grammar.image_name, .{ .root_source_file = files.add(grammar.image_name, grammar.image_name) });
    for (list.units ++ list.optional) |k| m.addAnonymousImport(b.fmt("contract_fatbin_{s}", .{k.name}), .{
        .root_source_file = files.add(b.fmt("{s}.fatbin", .{k.name}), k.name),
    });
}

fn addNames(b: *std.Build, m: *std.Build.Module) void {
    var names: [list.units.len + list.optional.len + 1][]const u8 = undefined;
    for (list.units ++ list.optional, 0..) |unit, i| names[i] = unit.name;
    names[names.len - 1] = grammar.image_name;
    const options = b.addOptions();
    options.addOption([]const []const u8, "names", &names);
    m.addOptions("contract_deepseek_names", options);
}
