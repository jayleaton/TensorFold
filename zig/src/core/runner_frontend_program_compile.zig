//! Freestanding RV32 compilation; the host compiles source but never executes the resulting program.
const std = @import("std");
const Allocator = std.mem.Allocator;
const inputs = @import("runner_frontend_c_inputs.zig");
const cost = @import("runner_frontend_c_cost.zig");

pub const Config = struct {
    clang: []const u8 = "/usr/bin/clang",
    linker: ?[]const u8 = null,
    scratch_root: []const u8 = "/tmp",
    optimize: enum { O0, O1, O2 } = .O1,
    stack_limit: usize = 0,
    opaque_inputs: bool = false,
    estimated_steps: ?usize = null,
    step_budget: usize = 10522,
    deadline: ?std.Io.Timestamp = null,
};

pub const linker_script =
    \\ENTRY(_start)
    \\MEMORY {
    \\ RAM (rw) : ORIGIN = 0x100, LENGTH = 0x7f00
    \\ DROM (r) : ORIGIN = 0x20000, LENGTH = 128K
    \\ ROM (rx) : ORIGIN = 0x40000, LENGTH = 256K
    \\}
    \\PHDRS { text PT_LOAD FLAGS(5); rodata PT_LOAD FLAGS(4); data PT_LOAD FLAGS(6); }
    \\SECTIONS {
    \\ .text : { *(.text.init) *(.text .text.*) } > ROM :text
    \\ .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > DROM :rodata
    \\ .data : { *(.data .data.* .sdata .sdata.*) } > RAM :data
    \\ .bss : { *(.bss .bss.* .sbss .sbss.*) *(COMMON) } > RAM :data
    \\ ASSERT(ADDR(.bss) + SIZEOF(.bss) <= 0x7000, "RAM leaves 4K stack")
    \\ /DISCARD/ : { *(.comment) *(.eh_frame*) *(.riscv.attributes) }
    \\}
;

const runtime =
    \\typedef unsigned int size_t;
    \\typedef signed char int8_t;
    \\typedef unsigned char uint8_t;
    \\typedef short int16_t;
    \\typedef unsigned short uint16_t;
    \\typedef int int32_t;
    \\typedef unsigned int uint32_t;
    \\typedef long long int64_t;
    \\typedef unsigned long long uint64_t;
    \\typedef _Bool bool;
    \\#define true 1
    \\#define false 0
    \\#define NULL ((void*)0)
    \\void *memset(void *destination, int value, size_t count) {
    \\ unsigned char *d = destination;
    \\ for(size_t i=0;i<count;++i) d[i]=(unsigned char)value;
    \\ return destination;
    \\}
    \\void *memcpy(void *destination, const void *source, size_t count) {
    \\ unsigned char *d = destination; const unsigned char *s = source;
    \\ for(size_t i=0;i<count;++i) d[i]=s[i];
    \\ return destination;
    \\}
    \\static unsigned int tf_output_count;
    \\static int putchar(int c) {
    \\ ++tf_output_count;
    \\ register int a0 __asm__("a0") = (unsigned char)c;
    \\ register int a7 __asm__("a7") = 1;
    \\ __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
    \\ return (unsigned char)c;
    \\}
    \\static void tf_unsigned(unsigned int n) {
    \\ char digits[10]; int len = 0;
    \\ do { digits[len++] = (char)('0' + n % 10); n /= 10; } while(n);
    \\ while(len) putchar(digits[--len]);
    \\}
    \\static void tf_signed(int n) {
    \\ if(n < 0) { putchar('-'); tf_unsigned(0u - (unsigned int)n); }
    \\ else tf_unsigned((unsigned int)n);
    \\}
    \\static unsigned long long tf_div10(unsigned long long n, unsigned int *remainder) {
    \\ unsigned long long q = 0; unsigned int r = 0;
    \\ for(int bit = 0; bit < 64; ++bit) {
    \\  r = (r << 1) | (unsigned int)(n >> 63); n <<= 1; q <<= 1;
    \\  if(r >= 10) { r -= 10; q |= 1; }
    \\ }
    \\ *remainder = r; return q;
    \\}
    \\static void tf_unsigned_wide(unsigned long long n) {
    \\ if(n <= 0xffffffffULL) { tf_unsigned((unsigned int)n); return; }
    \\ char digits[20]; int len = 0; unsigned int remainder;
    \\ do { n = tf_div10(n, &remainder); digits[len++] = (char)('0' + remainder); } while(n);
    \\ while(len) putchar(digits[--len]);
    \\}
    \\static void tf_signed_wide(long long n) {
    \\ unsigned long long magnitude = (unsigned long long)n;
    \\ if(n < 0) { putchar('-'); magnitude = 0ULL - magnitude; }
    \\ tf_unsigned_wide(magnitude);
    \\}
    \\static void tf_print_int(int n) { tf_signed(n); putchar('\n'); }
    \\static int puts(const char *s) { while(*s) putchar(*s++); putchar('\n'); return 0; }
    \\static int printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
    \\static int printf(const char *fmt, ...) {
    \\ unsigned int before = tf_output_count;
    \\ __builtin_va_list args; __builtin_va_start(args, fmt);
    \\ while(*fmt) {
    \\  if(*fmt != '%') { putchar(*fmt++); continue; }
    \\  ++fmt;
    \\  if(fmt[0] == 'l' && fmt[1] == 'l') {
    \\   fmt += 2;
    \\   if(*fmt == 'd' || *fmt == 'i') tf_signed_wide(__builtin_va_arg(args, long long));
    \\   else if(*fmt == 'u') tf_unsigned_wide(__builtin_va_arg(args, unsigned long long));
    \\   else { __builtin_va_end(args); return -1; }
    \\  }
    \\  else if(*fmt == 'd' || *fmt == 'i') tf_signed(__builtin_va_arg(args, int));
    \\  else if(*fmt == 'u') tf_unsigned(__builtin_va_arg(args, unsigned int));
    \\  else if(*fmt == 'c') putchar(__builtin_va_arg(args, int));
    \\  else if(*fmt == 's') { const char *s = __builtin_va_arg(args, const char*); while(*s) putchar(*s++); }
    \\  else if(*fmt == '%') putchar('%');
    \\  else { __builtin_va_end(args); return -1; }
    \\  ++fmt;
    \\ }
    \\ __builtin_va_end(args); return (int)(tf_output_count - before);
    \\}
    \\int main(void);
    \\__asm__(".section .text.init,\"ax\"\n.globl _start\n_start:\nli sp, 32768\ncall main\nli a7, 93\necall\n1: j 1b\n");
;

/// Wrap the checked Python body for compile(); callers own the returned source.
pub fn wrapBody(a: Allocator, body: []const u8) Allocator.Error![]u8 {
    return std.fmt.allocPrint(a, "int main(void) {{\n{s}}}\n", .{body});
}

/// Returns owned ELF32 bytes, or abstains on invalid source, unavailable toolchain or compilation failure.
pub fn compile(a: Allocator, io: std.Io, source: []const u8, config: Config) Allocator.Error!?[]u8 {
    if (config.deadline) |end| if (std.Io.Clock.awake.now(io).toNanoseconds() >= end.toNanoseconds()) return null;
    const normalized = try normalizeWithTrust(a, source, config.opaque_inputs) orelse return null;
    defer a.free(normalized);
    if (!std.fs.path.isAbsolute(config.scratch_root)) return null;
    const parent = std.Io.Dir.cwd().openDir(io, config.scratch_root, .{}) catch return null;
    defer parent.close(io);
    var nonce: [16]u8 = undefined;
    var name_buf: [64]u8 = undefined;
    var name: []const u8 = undefined;
    var created = false;
    for (0..8) |_| {
        io.random(&nonce);
        name = std.fmt.bufPrint(&name_buf, "tf-rv32-{x}", .{std.mem.readInt(u128, &nonce, .little)}) catch unreachable;
        parent.createDir(io, name, .fromMode(0o700)) catch |err| switch (err) {
            error.PathAlreadyExists => continue,
            else => return null,
        };
        created = true;
        break;
    }
    if (!created) return null;
    defer parent.deleteTree(io, name) catch {};
    const dir = parent.openDir(io, name, .{ .follow_symlinks = false }) catch return null;
    defer dir.close(io);
    const complete = try std.fmt.allocPrint(a, "{s}\n{s}\n", .{ runtime, normalized });
    defer a.free(complete);
    if (!write(io, dir, "program.c", complete) or !write(io, dir, "rom.ld", linker_script)) return null;
    const path = try std.fs.path.join(a, &.{ config.scratch_root, name });
    defer a.free(path);
    const src = try std.fs.path.join(a, &.{ path, "program.c" });
    defer a.free(src);
    const elf = try std.fs.path.join(a, &.{ path, "program.elf" });
    defer a.free(elf);
    const link_path = try std.fs.path.join(a, &.{ path, "rom.ld" });
    defer a.free(link_path);
    const linker_arg = if (config.linker) |value|
        try std.fmt.allocPrint(a, "--ld-path={s}", .{value})
    else
        try a.dupe(u8, "-fuse-ld=lld");
    defer a.free(linker_arg);
    if (config.step_budget != 0) {
        if (config.estimated_steps) |steps| if (steps > config.step_budget) return null;
    }
    if (!config.opaque_inputs or (config.step_budget != 0 and config.estimated_steps == null)) {
        const ast = std.process.run(a, io, .{
            .argv = &.{ config.clang, "--target=riscv32-unknown-elf", "-march=rv32im_zba", "-mabi=ilp32", "-ffreestanding", "-fno-builtin", "-nostdinc", "-Werror=format", "-fsyntax-only", "-Xclang", "-ast-dump=json", "-x", "c", src },
            .stdout_limit = .limited(4 * 1024 * 1024),
            .stderr_limit = .limited(16384),
            .timeout = .{ .duration = .{ .raw = remaining(io, config, 500) orelse return null, .clock = .awake } },
        }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return null,
        };
        defer a.free(ast.stdout);
        defer a.free(ast.stderr);
        if (!ast.term.success()) return null;
        var instrumented: ?[]u8 = null;
        var restoration_steps: usize = 0;
        defer if (instrumented) |text| a.free(text);
        if (!config.opaque_inputs) {
            const lowered = try inputs.lowerWithCost(a, normalized, ast.stdout, runtime.len + 1) orelse return null;
            instrumented = lowered.source;
            restoration_steps = lowered.restoration_steps;
        }
        if (config.step_budget != 0 and config.estimated_steps == null) {
            const startup = restoration_steps;
            if (startup >= config.step_budget) return null;
            _ = try cost.estimate(a, ast.stdout, runtime.len + 1, config.step_budget - startup) orelse return null;
        }
        if (instrumented) |text| {
            const rewritten = try std.fmt.allocPrint(a, "{s}\n{s}\n", .{ runtime, text });
            defer a.free(rewritten);
            const file = dir.createFile(io, "program.c", .{ .truncate = true, .permissions = .fromMode(0o600) }) catch return null;
            defer file.close(io);
            file.writeStreamingAll(io, rewritten) catch return null;
        }
    }
    const result = std.process.run(a, io, .{
        .argv = &.{
            config.clang,           "--target=riscv32-unknown-elf", "-march=rv32im_zba",   "-mabi=ilp32",
            switch (config.optimize) {
                .O0 => "-O0",
                .O1 => "-O1",
                .O2 => "-O2",
            },
            "-ffreestanding",       "-fno-builtin",                 "-nostdinc",           "-nostdlib",
            "-fno-stack-protector", "-fno-pic",                     "-mno-relax",          "-Werror=format",
            "-Wno-format-security", "-fstack-usage",                "-fno-unwind-tables",  "-fno-asynchronous-unwind-tables",
            linker_arg,             "-Wl,--no-relax",               "-Wl,--build-id=none", "-Wl,-T",
            link_path,              "-x",                           "c",                   src,
            "-o",                   elf,
        },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(16384),
        .timeout = .{ .duration = .{ .raw = remaining(io, config, 1000) orelse return null, .clock = .awake } },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    defer a.free(result.stdout);
    defer a.free(result.stderr);
    if (!result.term.success()) return null;
    if (config.stack_limit != 0) {
        const usage = dir.readFileAlloc(io, "program.su", a, .limited(65536)) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return null,
        };
        defer a.free(usage);
        if (!safeStackUsage(usage, config.stack_limit)) return null;
    }
    const bytes = dir.readFileAlloc(io, "program.elf", a, .limited(1 << 20)) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    if (!validElf(bytes)) {
        a.free(bytes);
        return null;
    }
    return bytes;
}

fn remaining(io: std.Io, config: Config, milliseconds: i64) ?std.Io.Duration {
    const cap = std.Io.Duration.fromMilliseconds(milliseconds);
    const end = config.deadline orelse return cap;
    const left = std.Io.Clock.awake.now(io).durationTo(end);
    if (left.toNanoseconds() <= 0) return null;
    return .fromNanoseconds(@min(left.toNanoseconds(), cap.toNanoseconds()));
}

// Summed static frames bound every call chain in the trusted, recursion-free Python lowering.
pub fn safeStackUsage(report: []const u8, limit: usize) bool {
    var total: usize = 0;
    var functions: usize = 0;
    var lines = std.mem.splitScalar(u8, report, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, "\r");
        if (line.len == 0) continue;
        var fields = std.mem.splitScalar(u8, line, '\t');
        _ = fields.next() orelse return false;
        const size = std.fmt.parseInt(usize, fields.next() orelse return false, 10) catch return false;
        const kind = fields.next() orelse return false;
        if (fields.next() != null or !std.mem.eql(u8, kind, "static") or size > limit - total) return false;
        total += size;
        functions += 1;
    }
    return functions != 0;
}

test "trusted Python stack admission sums static frames and rejects unbounded reports" {
    try std.testing.expect(safeStackUsage("p.c:1:main\t128\tstatic\np.c:2:f\t256\tstatic\n", 384));
    try std.testing.expect(!safeStackUsage("p.c:1:main\t128\tstatic\np.c:2:f\t256\tstatic\n", 383));
    try std.testing.expect(!safeStackUsage("p.c:1:main\t32\tdynamic\n", 4096));
    try std.testing.expect(!safeStackUsage("p.c:1:main\tbad\tstatic\n", 4096));
    try std.testing.expect(!safeStackUsage("", 4096));
}

fn write(io: std.Io, dir: std.Io.Dir, name: []const u8, bytes: []const u8) bool {
    const f = dir.createFile(io, name, .{ .exclusive = true, .permissions = .fromMode(0o600) }) catch return false;
    defer f.close(io);
    f.writeStreamingAll(io, bytes) catch return false;
    return true;
}

fn normalize(a: Allocator, source: []const u8) Allocator.Error!?[]u8 {
    return normalizeWithTrust(a, source, false);
}

fn normalizeWithTrust(a: Allocator, source: []const u8, trusted: bool) Allocator.Error!?[]u8 {
    if (source.len == 0 or source.len > 16384 or std.mem.indexOfScalar(u8, source, 0) != null) return null;
    if (std.mem.indexOf(u8, source, "\\\n") != null or std.mem.indexOf(u8, source, "\\\r\n") != null) return null;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, trimmed, "#include")) {
            const header = std.mem.trim(u8, trimmed[8..], " \t");
            const allowed = [_][]const u8{ "<stdio.h>", "<stdint.h>", "<stddef.h>", "<stdbool.h>" };
            var ok = false;
            for (allowed) |include| {
                if (std.mem.eql(u8, header, include)) ok = true;
            }
            if (!ok) return null;
            continue;
        }
        if (std.mem.startsWith(u8, trimmed, "#define ")) {
            var fields = std.mem.tokenizeAny(u8, trimmed[8..], " \t");
            const name = fields.next() orelse return null;
            const value = fields.next() orelse return null;
            if (fields.next() != null or name.len == 0 or !(std.ascii.isAlphabetic(name[0]) or name[0] == '_')) return null;
            for (name) |c| if (!std.ascii.isAlphanumeric(c) and c != '_') return null;
            _ = std.fmt.parseInt(u32, value, 0) catch return null;
        } else if (std.mem.indexOfScalar(u8, line, '#') != null) return null;
        try out.appendSlice(a, line);
        try out.append(a, '\n');
    }
    const forbidden = [_][]const u8{ "__", "asm", "_Pragma", "??", "%:", "\\\n", "\\\r" };
    for (forbidden) |token| {
        if (trusted and (std.mem.eql(u8, token, "__") or std.mem.eql(u8, token, "asm"))) continue;
        if (std.mem.indexOf(u8, out.items, token) != null) return null;
    }
    if (!safeFormats(out.items)) return null;
    return try out.toOwnedSlice(a);
}

pub fn validElf(b: []const u8) bool {
    if (b.len < 52 or !std.mem.eql(u8, b[0..4], "\x7fELF") or b[4] != 1 or b[5] != 1 or b[6] != 1)
        return false;
    if (read(u16, b, 16) != 2 or read(u16, b, 18) != 243 or read(u32, b, 20) != 1 or
        read(u16, b, 40) != 52 or read(u16, b, 42) != 32 or read(u32, b, 36) != 0) return false;
    const entry = read(u32, b, 24);
    if (entry < 0x40000 or entry >= 0x80000 or entry % 4 != 0) return false;
    const off: usize = read(u32, b, 28);
    const count: usize = read(u16, b, 44);
    if (count == 0 or count > 16 or off > b.len or count * 32 > b.len - off) return false;
    var entry_loaded = false;
    for (0..count) |i| {
        const ph = off + 32 * i;
        const kind = read(u32, b, ph);
        if (kind == 2 or kind == 3 or kind == 7) return false;
        if (kind != 1) continue;
        const start: u64 = read(u32, b, ph + 8);
        const file_off: usize = read(u32, b, ph + 4);
        const file_size: usize = read(u32, b, ph + 16);
        const size: u64 = read(u32, b, ph + 20);
        const flags = read(u32, b, ph + 24);
        if (size == 0) continue;
        if (file_size > size or file_off > b.len or file_size > b.len - file_off) return false;
        const end = start + size;
        const ram = start >= 0x100 and end <= 0x7000 and flags & 1 == 0;
        const drom = start >= 0x20000 and end <= 0x40000 and flags & 3 == 0;
        const rom = start >= 0x40000 and end <= 0x80000 and flags & 2 == 0;
        if (!ram and !drom and !rom) return false;
        if (rom and flags & 1 != 0 and entry >= start and entry < start + file_size) entry_loaded = true;
    }
    return entry_loaded;
}

fn read(comptime T: type, b: []const u8, off: usize) T {
    return std.mem.readInt(T, b[off..][0..@sizeOf(T)], .little);
}

test "missing toolchain abstains" {
    try std.testing.expectEqual(@as(?[]u8, null), try compile(std.testing.allocator, std.testing.io, "int main(void) { return 0; }", .{ .clang = "/nonexistent-tensorfold-clang" }));
}

test "source cannot access host includes or inject assembly" {
    const bad = [_][]const u8{ "#include </etc/passwd>", "#define X(x) x", "asm(\".incbin x\");", "_Pragma(\"x\")", "int x = __FILE__;", "%:include </etc/passwd>", "pri\\\nntf(\"%f\", 1.0);" };
    for (bad) |s| try std.testing.expectEqual(@as(?[]u8, null), try normalize(std.testing.allocator, s));
    try std.testing.expect(!validElf("not an ELF"));
}

/// Checks literal printf conversion syntax; Clang checks argument types during compilation.
pub fn safeFormats(s: []const u8) bool {
    var i: usize = 0;
    while (i < s.len) {
        if (std.mem.startsWith(u8, s[i..], "//")) {
            while (i < s.len and s[i] != '\n') i += 1;
            continue;
        }
        if (s[i] == '/' and i + 1 < s.len and s[i + 1] == '*') {
            const end = std.mem.indexOf(u8, s[i + 2 ..], "*/") orelse return false;
            i += end + 4;
            continue;
        }
        if (s[i] == '"' or s[i] == '\'') {
            const quote = s[i];
            i += 1;
            while (i < s.len and s[i] != quote) : (i += 1) {
                if (s[i] == '\\') i += 1;
            }
            if (i >= s.len) return false;
            i += 1;
            continue;
        }
        if (std.ascii.isAlphabetic(s[i]) or s[i] == '_') {
            const start = i;
            while (i < s.len and (std.ascii.isAlphanumeric(s[i]) or s[i] == '_')) i += 1;
            if (!std.mem.eql(u8, s[start..i], "printf")) continue;
            skipSpace(s, &i);
            if (i == s.len or s[i] != '(') return false;
            i += 1;
            skipSpace(s, &i);
            if (i == s.len or s[i] != '"') return false;
            i += 1;
            while (i < s.len and s[i] != '"') : (i += 1) {
                if (s[i] == '\\') {
                    i += 1;
                    if (i == s.len or std.mem.indexOfScalar(u8, "nrtvabf\\\"?", s[i]) == null) return false;
                } else if (s[i] == '%') {
                    i += 1;
                    if (std.mem.startsWith(u8, s[i..], "ll")) {
                        i += 2;
                        if (i == s.len or std.mem.indexOfScalar(u8, "diu", s[i]) == null) return false;
                    } else if (i == s.len or std.mem.indexOfScalar(u8, "diucs%", s[i]) == null) return false;
                }
            }
            if (i == s.len) return false;
            i += 1;
            skipSpace(s, &i);
            if (i == s.len or (s[i] != ',' and s[i] != ')')) return false;
            continue;
        }
        i += 1;
    }
    return true;
}

fn skipSpace(s: []const u8, i: *usize) void {
    while (i.* < s.len and std.ascii.isWhitespace(s[i.*])) i.* += 1;
}

test "printf admits literal integer formats and refuses formats the runtime cannot reproduce" {
    try std.testing.expect(safeFormats("printf(\"%d %u %c %s %%\\n\", 1, 2u, 'c', \"text\");"));
    const bad = [_][]const u8{
        "printf(\"%f\", 1.0);", "printf(\"%ld\", 1L);",     "printf(\"%04d\", 1);",
        "printf(fmt, 1);",      "printf(\"\\x25f\", 1.0);", "printf(\"%\" \"f\", 1.0);",
        "void *p = printf;",
    };
    for (bad) |s| try std.testing.expect(!safeFormats(s));
}

pub fn hasMain(s: []const u8) bool {
    var i: usize = 0;
    while (i < s.len) {
        if (std.mem.startsWith(u8, s[i..], "//")) {
            while (i < s.len and s[i] != '\n') i += 1;
        } else if (s[i] == '/' and i + 1 < s.len and s[i + 1] == '*') {
            const end = std.mem.indexOf(u8, s[i + 2 ..], "*/") orelse return false;
            i += end + 4;
        } else if (s[i] == '"' or s[i] == '\'') {
            const quote = s[i];
            i += 1;
            while (i < s.len and s[i] != quote) : (i += 1) {
                if (s[i] == '\\') i += 1;
            }
            if (i < s.len) i += 1;
        } else if (std.ascii.isAlphabetic(s[i]) or s[i] == '_') {
            const start = i;
            while (i < s.len and (std.ascii.isAlphanumeric(s[i]) or s[i] == '_')) i += 1;
            if (std.mem.eql(u8, s[start..i], "main")) {
                skipSpace(s, &i);
                if (i < s.len and s[i] == '(') return true;
            }
        } else i += 1;
    }
    return false;
}
