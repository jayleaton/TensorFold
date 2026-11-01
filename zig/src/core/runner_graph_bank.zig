//! Boot-loaded immutable prepared weights, logical table slots and validated named binding plans.
const std = @import("std");
const model = @import("runner_circuit_model.zig");
const binary = @import("runner_graph_binary.zig");
const packet = @import("runner_graph_packet.zig");
const Allocator = std.mem.Allocator;
const missing = std.math.maxInt(u32);
pub const Slots = [packet.max_arity][2]u32;

pub const Template = struct {
    name: []const u8,
    width: u8,
    circuit: model.Circuit,
    inputs: u8,
    outputs: u8,
    in_slots: Slots,
    out_slots: Slots,
};

const Descriptor = struct {
    name: []const u8,
    width: u8,
    path: []const u8,
    inputs: ?u8 = null,
    outputs: ?u8 = null,
};
pub const PreparedPlan = struct {
    name: []const u8,
    inputs: u32,
    ops: []const packet.Op,
    result: u32,
    constants: []const packet.Value,
    total_values: usize,
};

const Manifest = struct {
    version: u32,
    templates: []const Descriptor,
    plans: []const packet.Plan = &.{},
};

pub const Bank = struct {
    gpa: Allocator,
    arena: std.heap.ArenaAllocator,
    templates: []Template,
    plans: []const PreparedPlan,

    pub fn load(a: Allocator, path: []const u8) !Bank {
        var threaded: std.Io.Threaded = .init_single_threaded;
        const bytes = try std.Io.Dir.cwd().readFileAlloc(threaded.io(), path, a, .limited(32 << 20));
        defer a.free(bytes);
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const doc = try std.json.parseFromSliceLeaky(Manifest, arena.allocator(), bytes, .{ .allocate = .alloc_always });
        if (doc.version != 1 or doc.templates.len == 0 or doc.templates.len > 256) return error.GraphManifest;
        const templates = try arena.allocator().alloc(Template, doc.templates.len);
        var loaded: usize = 0;
        errdefer for (templates[0..loaded]) |*t| t.circuit.deinit(a);
        for (doc.templates, templates) |d, *t| {
            if (!packet.widthValid(d.width) or d.name.len == 0) return error.GraphTemplate;
            for (templates[0..loaded]) |prior| if (prior.width == d.width and std.mem.eql(u8, prior.name, d.name))
                return error.GraphDuplicateTemplate;
            const resolved = if (std.fs.path.isAbsolute(d.path)) try a.dupe(u8, d.path) else try std.fs.path.join(a, &.{ std.fs.path.dirname(path) orelse ".", d.path });
            defer a.free(resolved);
            t.* = .{ .name = d.name, .width = d.width, .circuit = try binary.load(a, resolved, d.name, d.width), .inputs = 0, .outputs = 0, .in_slots = @splat(@splat(missing)), .out_slots = @splat(@splat(missing)) };
            loaded += 1;
            try slots(t);
            if (d.inputs) |n| if (n != t.inputs) return error.GraphSlotArity;
            if (d.outputs) |n| if (n != t.outputs) return error.GraphSlotArity;
        }
        const plans = try arena.allocator().alloc(PreparedPlan, doc.plans.len);
        var bank: Bank = .{ .gpa = a, .arena = arena, .templates = templates, .plans = plans };
        for (doc.plans, 0..) |p, i| {
            if (p.name.len == 0) return error.GraphPlan;
            for (doc.plans[0..i]) |prior| if (std.mem.eql(u8, prior.name, p.name)) return error.GraphDuplicatePlan;
            if (p.inputs > packet.max_values or p.literals.len > packet.max_values - p.inputs)
                return error.GraphPlanSize;
            const constants = try arena.allocator().alloc(packet.Value, p.literals.len);
            for (p.literals, constants) |text, *v| v.* = try packet.literal(text);
            plans[i] = .{ .name = p.name, .inputs = p.inputs, .ops = p.ops, .result = p.result, .constants = constants, .total_values = try bank.validate(p.inputs + p.literals.len, p.ops, p.result) };
        }
        const emitter = bank.find("emit", 62) orelse return error.GraphEmitMissing;
        if (emitter.inputs != 1 or emitter.outputs != 1) return error.GraphSlotArity;
        bank.arena = arena;
        return bank;
    }

    pub fn deinit(b: *Bank) void {
        for (b.templates) |*t| t.circuit.deinit(b.gpa);
        b.arena.deinit();
        b.* = undefined;
    }

    pub fn find(b: *const Bank, name: []const u8, width: u8) ?*const Template {
        for (b.templates) |*t| if (t.width == width and std.mem.eql(u8, t.name, name)) return t;
        return null;
    }

    pub fn plan(b: *const Bank, name: []const u8) ?*const PreparedPlan {
        for (b.plans) |*p| if (std.mem.eql(u8, p.name, name)) return p;
        return null;
    }

    pub fn validate(b: *const Bank, inputs: usize, ops: []const packet.Op, result: u32) !usize {
        if (inputs > packet.max_values or ops.len > packet.max_ops) return error.GraphPlanSize;
        var count = inputs;
        for (ops) |op| {
            if (op.outputs == 0 or op.outputs > packet.max_arity or count + op.outputs > packet.max_values)
                return error.GraphPlanSize;
            if (std.mem.eql(u8, op.op, "bind")) {
                if (op.outputs != 1 or op.copies.len > 128 or op.args != null or op.b != null)
                    return error.GraphBinding;
                for (op.copies) |c| if (c.ref >= count or c.from > 62 or c.to > 62 or
                    c.digits > 62 - c.from or c.digits > 62 - c.to) return error.GraphBinding;
                if (op.flags) |r| if (r >= count) return error.GraphReference;
            } else {
                if (std.mem.eql(u8, op.op, "emit") or op.copies.len != 0 or op.flags != null)
                    return error.GraphTemplate;
                const t = b.find(op.op, op.width) orelse return error.GraphTemplate;
                if (t.outputs != op.outputs) return error.GraphSlotArity;
                if (op.args) |args| {
                    if (args.len != t.inputs) return error.GraphSlotArity;
                    for (args) |r| if (r >= count) return error.GraphReference;
                } else {
                    if (t.inputs != 1 + @as(usize, @intFromBool(op.b != null)) or op.a >= count)
                        return error.GraphSlotArity;
                    if (op.b) |r| if (r >= count) return error.GraphReference;
                }
            }
            count += op.outputs;
        }
        if (result >= count) return error.GraphReference;
        return count;
    }
};

fn slots(t: *Template) !void {
    if (t.circuit.program.a_bit != packet.bit_scale or t.circuit.program.a_ref != packet.reference_scale)
        return error.GraphValueScale;
    if (t.circuit.tables.len != t.circuit.table_names.len) return error.GraphSlots;
    for (t.circuit.table_names, 0..) |name, table| {
        if (std.mem.eql(u8, name, "output.cursor")) continue;
        const input = std.mem.startsWith(u8, name, "in");
        const output = std.mem.startsWith(u8, name, "out");
        if (!input and !output) continue;
        const prefix: usize = if (input) 2 else 3;
        const dot = std.mem.indexOfScalarPos(u8, name, prefix, '.') orelse return error.GraphSlots;
        const logical = std.fmt.parseInt(u8, name[prefix..dot], 10) catch return error.GraphSlots;
        const offset = std.fmt.parseInt(u8, name[dot + 1 ..], 10) catch return error.GraphSlots;
        if (logical >= packet.max_arity or (offset != 0 and offset != 31)) return error.GraphSlots;
        if (offset >= t.width) return error.GraphSlots;
        const target = if (input) &t.in_slots else &t.out_slots;
        const group = offset / 31;
        if (target[logical][group] != missing) return error.GraphSlots;
        if (t.circuit.tables[table].rows.len != 128 * 128) return error.GraphSlots;
        target[logical][group] = @intCast(table);
        const arity = if (input) &t.inputs else &t.outputs;
        arity.* = @max(arity.*, logical + 1);
    }
    if (t.inputs == 0 or t.outputs == 0) return error.GraphSlots;
    const groups: usize = if (t.width > 31) 2 else 1;
    for (t.in_slots[0..t.inputs]) |s| for (s[0..groups]) |idx| if (idx == missing) return error.GraphSlots;
    for (t.out_slots[0..t.outputs]) |s| for (s[0..groups]) |idx| if (idx == missing) return error.GraphSlots;
}
