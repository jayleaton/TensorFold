//! A request's prepared images as the server hands them to the engine (``api.Request.images``, rank 0 only): plain
pub const HeldImage = extern struct {
    /// bf16 patches [vit_h vit_w, 3, 14, 14]
    patches: [*]const u16,
    vit_h: u32,
    vit_w: u32,
    llm_h: u32,
    llm_w: u32,
    digest: [32]u8,
    /// the span's virtual ids, llm_h (llm_w + 1) + 2 of them
    vids: [*]const u32,
    n_vids: u64,
};

pub const Held = extern struct {
    images: [*]const HeldImage,
    n: u64,

    pub fn slice(h: *const Held) []const HeldImage {
        return h.images[0..h.n];
    }
};
