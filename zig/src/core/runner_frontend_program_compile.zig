//! Freestanding RV32 compilation; the host compiles source but never executes the resulting program.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Config = struct {
    clang: []const u8 = "/usr/bin/clang",
    linker: ?[]const u8 = null,
    scratch_root: []const u8 = "/tmp",
    optimize: bool = false,
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
    \\static void tf_print_int(int n) { tf_signed(n); putchar('\n'); }
    \\static int puts(const char *s) { while(*s) putchar(*s++); putchar('\n'); return 0; }
    \\static int printf(const char *fmt, ...) {
    \\ unsigned int before = tf_output_count;
    \\ __builtin_va_list args; __builtin_va_start(args, fmt);
    \\ while(*fmt) {
    \\  if(*fmt != '%') { putchar(*fmt++); continue; }
    \\  ++fmt;
    \\  if(*fmt == 'd' || *fmt == 'i') tf_signed(__builtin_va_arg(args, int));
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
    const normalized = try normalize(a, source) orelse return null;
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
    const result = std.process.run(a, io, .{
        .argv = &.{
            config.clang,                          "--target=riscv32-unknown-elf",    "-march=rv32im_zba", "-mabi=ilp32",
            if (config.optimize) "-O2" else "-O0", "-ffreestanding",                  "-fno-builtin",      "-nostdinc",
            "-nostdlib",                           "-fno-stack-protector",            "-fno-pic",          "-mno-relax",
            "-fno-unwind-tables",                  "-fno-asynchronous-unwind-tables", linker_arg,          "-Wl,--no-relax",
            "-Wl,--build-id=none",                 "-Wl,-T",                          link_path,           "-x",
            "c",                                   src,                               "-o",                elf,
        },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(16384),
        .timeout = .{ .duration = .{ .raw = .fromSeconds(1), .clock = .awake } },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    defer a.free(result.stdout);
    defer a.free(result.stderr);
    if (!result.term.success()) return null;
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

fn write(io: std.Io, dir: std.Io.Dir, name: []const u8, bytes: []const u8) bool {
    const f = dir.createFile(io, name, .{ .exclusive = true, .permissions = .fromMode(0o600) }) catch return false;
    defer f.close(io);
    f.writeStreamingAll(io, bytes) catch return false;
    return true;
}

fn normalize(a: Allocator, source: []const u8) Allocator.Error!?[]u8 {
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
        if (std.mem.indexOfScalar(u8, line, '#') != null) return null;
        try out.appendSlice(a, line);
        try out.append(a, '\n');
    }
    const forbidden = [_][]const u8{ "__", "asm", "_Pragma", "??", "%:", "\\\n", "\\\r" };
    for (forbidden) |token| if (std.mem.indexOf(u8, out.items, token) != null) return null;
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
    const bad = [_][]const u8{ "#include </etc/passwd>", "#define X 1", "asm(\".incbin x\");", "_Pragma(\"x\")", "int x = __FILE__;", "%:include </etc/passwd>", "pri\\\nntf(\"%f\", 1.0);" };
    for (bad) |s| try std.testing.expectEqual(@as(?[]u8, null), try normalize(std.testing.allocator, s));
    try std.testing.expect(!validElf("not an ELF"));
}

fn safeFormats(s: []const u8) bool {
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
                    if (i == s.len or std.mem.indexOfScalar(u8, "diucs%", s[i]) == null) return false;
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
