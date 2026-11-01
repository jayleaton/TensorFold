const std = @import("std");
const f = @import("runner_frontend.zig");
const t = std.testing;

test "frontend accepts model multiply and REPL power without computing the answer" {
    const front = f.frontend();
    const cases = [_][3][]const u8{
        .{ "python", "print(123456789012345678901234567890 * 987654321098765432109876543210)", "123456789012345678901234567890*987654321098765432109876543210" },
        .{ "py", "2**100", "2**100" },
        .{ "python3", "7 // -3", "7//-3" },
        .{ "exact", "1/3 + 2/3", "1/3+2/3" },
        .{ "calc", "1.25*2.4", "1.25*2.4" },
    };
    for (cases) |case| {
        const result = try front.translate(front.ctx, t.allocator, .{ .form = .repl, .lang = case[0], .code = case[1] });
        try t.expect(result == .program);
        defer t.allocator.free(result.program.bytes);
        try t.expectEqual(.calc, result.program.kind);
        try t.expectEqualStrings(case[2], result.program.bytes);
    }
}

test "frontend declines unsupported code and Python output mismatches" {
    const front = f.frontend();
    const cases = [_][2][]const u8{
        .{ "javascript", "2*3" },            .{ "python", "print(missing)" },  .{ "python", "import os" },
        .{ "python", "print(1/3)" },         .{ "python", "print(1.25*2.4)" }, .{ "python", "gcd(2,4)" },
        .{ "python", "print(2); print(3)" }, .{ "python", "10**100" },         .{ "python", "1//0" },
    };
    for (cases) |case| {
        const result = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = case[0], .code = case[1] });
        try t.expect(result == .abstain);
    }
}

test "frontend allocator failure is visible to its caller" {
    const front = f.frontend();
    try t.expectError(error.OutOfMemory, front.translate(null, std.testing.failing_allocator, .{
        .form = .repl,
        .lang = "python",
        .code = "2**100",
    }));
}

test {
    _ = @import("runner_frontend_calc.zig");
    _ = @import("runner_frontend_program.zig");
    _ = @import("runner_frontend_reference_test.zig");
}

test "real compiler emits circuit ELF for loops and C with negative controls" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse {
        std.debug.print("SKIP RV32 toolchain: set TF_RUNNER_CLANG and optionally TF_RUNNER_LLD\n", .{});
        return error.SkipZigTest;
    };
    var context: f.Context = .{
        .io = t.io,
        .compiler = .{ .clang = clang, .linker = env.get("TF_RUNNER_LLD") },
    };
    const front = context.frontend();
    const cases = [_][4][]const u8{
        .{ "python", "print(sum(i*i for i in range(10)))", "squares.elf", "285\n" },
        .{ "python", "print(sum(i+i for i in range(10)))", "addition.elf", "90\n" },
        .{ "python", "s = 0\nfor i in range(10):\n    s += i*i\nprint(s)", "loop.elf", "285\n" },
        .{ "c", "#include <stdio.h>\nint main(void) { int s=0; for(int i=0;i<10;i++) s+=i*i; printf(\"%d\\n\",s); return 0; }", "c.elf", "285\n" },
        .{ "c", "int main(void) { puts(\"circuit\"); return 0; }", "puts.elf", "circuit\n" },
        .{ "c", "int s=0; for(int i=0;i<10;i++) s+=i*i; printf(\"%d\\n\",s);", "snippet.elf", "285\n" },
    };
    for (cases) |case| {
        const result = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = case[0], .code = case[1] });
        try t.expect(result == .program);
        defer t.allocator.free(result.program.bytes);
        try t.expectEqual(.elf, result.program.kind);
        const repeated = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = case[0], .code = case[1] });
        try t.expect(repeated == .program);
        defer t.allocator.free(repeated.program.bytes);
        try t.expectEqualSlices(u8, result.program.bytes, repeated.program.bytes);
        try t.expect(@import("runner_frontend_program.zig").compiler.validElf(result.program.bytes));
        if (env.get("TF_RUNNER_QWENPUTER")) |root|
            try @import("runner_frontend_reference_test.zig").check(root, env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", result.program, case[3]);
        if (env.get("TF_RUNNER_FRONTEND_ARTIFACTS")) |path| {
            const dir = try std.Io.Dir.cwd().openDir(t.io, path, .{});
            defer dir.close(t.io);
            const file = try dir.createFile(t.io, case[2], .{});
            defer file.close(t.io);
            try file.writeStreamingAll(t.io, result.program.bytes);
        }
    }
    const missing = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = "c", .code = "int main(void) { unknown(); }" });
    try t.expect(missing == .abstain);
    context.compiler.clang = "/nonexistent-tensorfold-clang";
    const unavailable = try front.translate(front.ctx, t.allocator, .{
        .form = .repl,
        .lang = "python",
        .code = "print(sum(i*i for i in range(10)))",
    });
    try t.expect(unavailable == .abstain);
}

test "prior cells cannot silently shadow Python builtins" {
    const front = f.frontend();
    const result = try front.translate(null, t.allocator, .{
        .form = .repl,
        .lang = "python",
        .code = "print(2*3)",
        .prior = &.{"print = lambda x: None"},
    });
    try t.expect(result == .abstain);
}
