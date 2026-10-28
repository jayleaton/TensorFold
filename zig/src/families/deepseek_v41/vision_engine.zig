//! Translate shared image requests and retain inputs through terminal callbacks; cancellation alone never frees them.
const std = @import("std");
const api = @import("engine_api");
const Native = @import("vision_native.zig").Native;
pub const Engine = Wrapper(@import("vision_rows.zig").Rows);

/// The ownership seam also runs against the actual host Store, without a CUDA driver.
pub fn Wrapper(comptime Rows: type) type {
    return struct {
        const Self = @This();
        const Owner = struct {
            arena: std.heap.ArenaAllocator,
            request: api.Request,
            sink: api.Sink,
        };
        gpa: std.mem.Allocator,
        io: std.Io,
        inner: api.Engine,
        native: *Native,
        rows: *Rows,
        mutex: std.Io.Mutex = .init,
        active: std.AutoHashMapUnmanaged(api.Id, *Owner) = .empty,
        active_churn: @import("map_churn.zig").Churn = .{},

        /// Call only after the inner engine and GPU have drained successfully.
        pub fn deinit(e: *Self) void {
            var it = e.active.iterator();
            while (it.next()) |entry| {
                e.rows.release(entry.key_ptr.*);
                e.free(entry.value_ptr.*);
            }
            e.active.deinit(e.gpa);
        }

        pub fn engine(e: *Self) api.Engine {
            return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
        }

        fn self(ctx: *anyopaque) *Self {
            return @ptrCast(@alignCast(ctx));
        }
        fn info(ctx: *anyopaque) api.Info {
            const e = self(ctx);
            var i = e.inner.info();
            i.vision = e.native.offer();
            return i;
        }
        fn cancel(ctx: *anyopaque, id: api.Id) void {
            self(ctx).inner.cancel(id);
        }
        fn status(ctx: *anyopaque, out: *api.Status, tokens: []u32) void {
            self(ctx).inner.status(out, tokens);
        }
        fn memory(ctx: *anyopaque, reset: bool) ?api.Memory {
            return self(ctx).inner.memory(reset);
        }

        fn submit(ctx: *anyopaque, id: api.Id, src: *const api.Request, sink: api.Sink) api.SubmitError!void {
            const e = self(ctx);
            if (src.images.len == 0) {
                for (src.prompt) |t| if (t >= (1 << 24)) return error.InvalidRequest;
                return e.inner.submit(id, src, sink);
            }
            const owner = e.gpa.create(Owner) catch return error.Busy;
            owner.* = .{ .arena = .init(e.gpa), .request = undefined, .sink = sink };
            var admitted = false;
            errdefer if (!admitted) e.free(owner);
            const translated = e.native.request(owner.arena.allocator(), src) catch |err| return switch (err) {
                error.OutOfMemory => error.Busy,
                else => error.InvalidRequest,
            };
            owner.request = translated.request;
            {
                e.mutex.lockUncancelable(e.io);
                defer e.mutex.unlock(e.io);
                if (e.active.contains(id)) return error.InvalidRequest;
                e.active.ensureUnusedCapacity(e.gpa, 1) catch return error.Busy;
                e.rows.hold(id, &translated.images) catch |err| return switch (err) {
                    error.OutOfMemory => error.Busy,
                    else => error.InvalidRequest,
                };
                e.active.putAssumeCapacity(id, owner);
                admitted = true;
            }
            e.inner.submit(id, &owner.request, .{ .ctx = e, .event = onEvent }) catch |err| {
                e.drop(id);
                return err;
            };
        }

        fn free(e: *Self, owner: *Owner) void {
            owner.arena.deinit();
            e.gpa.destroy(owner);
        }
        fn drop(e: *Self, id: api.Id) void {
            e.mutex.lockUncancelable(e.io);
            defer e.mutex.unlock(e.io);
            const owner = (e.active.fetchRemove(id) orelse return).value;
            e.active_churn.dropped(api.Id, &e.active);
            e.rows.release(id);
            e.free(owner);
        }
        fn onEvent(ctx: *anyopaque, id: api.Id, ev: *const api.Event) void {
            const e = self(ctx);
            e.mutex.lockUncancelable(e.io);
            const owner = e.active.get(id) orelse {
                e.mutex.unlock(e.io);
                return;
            };
            const sink = owner.sink;
            e.mutex.unlock(e.io);
            if (ev.* == .finished) e.drop(id);
            sink.event(sink.ctx, id, ev);
        }
    };
}
