//! Loads an rv32im_zba ELF into a booted circuit state as data (Qwenputer core.snapshot), never executing it.
const State = @import("runner_circuit_state.zig").State;

pub fn load(s: *State, image: []const u8) !void {
    _ = s;
    _ = image;
    return error.CircuitElfUnsupported;
}
