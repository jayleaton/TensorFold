// router_gemv v2 (TF_DSV41_RG_V2) against the kernel prod picks at each row count: bytes, a CPU fmaf oracle, times.
#include "../router_gemv2.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); exit(1); } } while (0)

using namespace dsv41_rg;

namespace {

// one rank's router (flash17): 384 routed experts, top-6 plus the shared slot, D 5,120
constexpr int E = 384, K = 6, SLOTS = 7, GE = E + 1;
constexpr float SCALE = 1.5f;
constexpr size_t FLUSH = 256u << 20;

uint16_t bf16_of(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
float f32_of(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
uint32_t bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

// the router_gemv.cu contract on the host: lane l's fmaf chain over k = 8 (l + 32 j) + i, then the xor butterfly
float oracle(const uint16_t* x, const uint16_t* w) {
    float v[32];
    for (int l = 0; l < 32; ++l) {
        float acc = 0.0f;
        for (int j = 0; j < NV; ++j)
            for (int i = 0; i < 8; ++i) {
                const int k = 8 * (l + 32 * j) + i;
                acc = fmaf(f32_of(x[k]), f32_of(w[k]), acc);
            }
        v[l] = acc;
    }
    for (int o = 16; o > 0; o >>= 1) {
        float n[32];
        for (int l = 0; l < 32; ++l) n[l] = v[l] + v[l ^ o];
        memcpy(v, n, sizeof v);
    }
    return v[0];
}

struct Bufs {
    int *pick, *cnt, *gid, *gcnt, *gmem;
    float *wts, *lg;
};

Bufs make_bufs(int R) {
    Bufs b{};
    CK(cudaMalloc(&b.pick, (size_t)R * SLOTS * 4));
    CK(cudaMalloc(&b.wts, (size_t)R * SLOTS * 4));
    CK(cudaMalloc(&b.lg, (size_t)R * E * 4));
    CK(cudaMalloc(&b.cnt, (size_t)(R + 15) * 4));
    CK(cudaMemset(b.cnt, 0, (size_t)(R + 15) * 4));
    CK(cudaMalloc(&b.gid, (size_t)GE * 4));
    CK(cudaMalloc(&b.gcnt, 4));
    CK(cudaMalloc(&b.gmem, (size_t)GE * std::max(R, 1) * 4));
    return b;
}
void free_bufs(Bufs& b) {
    for (void* p : {(void*)b.pick, (void*)b.wts, (void*)b.lg, (void*)b.cnt, (void*)b.gid, (void*)b.gcnt, (void*)b.gmem})
        CK(cudaFree(p));
}
void clear(const Bufs& b, int R) {
    CK(cudaMemset(b.pick, 0xff, (size_t)R * SLOTS * 4));
    CK(cudaMemset(b.wts, 0xff, (size_t)R * SLOTS * 4));
    CK(cudaMemset(b.lg, 0xff, (size_t)R * E * 4));
    CK(cudaMemset(b.gid, 0xff, (size_t)GE * 4));
    CK(cudaMemset(b.gcnt, 0xff, 4));
    CK(cudaMemset(b.gmem, 0xff, (size_t)GE * std::max(R, 1) * 4));
}

// prod's arguments: prefill past 16 rows (block_prefill: no kit, prune, grouping; --fold 1: kit and prune), else decode
Args args_for(int R, const uint16_t* x, const uint16_t* w, const float* bias, const Bufs& b, bool decode, bool fold) {
    Args a{};
    a.x = x; a.xs = D; a.w = w; a.bias = bias;
    a.pick = b.pick; a.wts = b.wts; a.lg = b.lg; a.cnt = b.cnt;
    a.R = R; a.E = E; a.K = K; a.slots = SLOTS; a.select = 1; a.scale = SCALE;
    a.kit = decode || fold ? 1 : 0;
    a.group = decode ? 1 : 0;
    a.gid = b.gid; a.gcnt = b.gcnt; a.gmem = b.gmem; a.GE = GE; a.maxm = std::max(R, 1);
    a.prune = Prune{};
    if (decode || fold) a.prune = Prune{0.0f, 0.85f, SCALE, 3, 0, GE, 0, 1};    // TF_DSV41_EXPERT_TOPP=0.85: prune.fold_args
    return a;
}

int tile_of(int R) {
    int t = 1;
    while (t < std::min(R, 16)) t *= 2;
    return t;
}

// the launch prod makes (router_gemv.config / row_tile): narrow (rt, 2) to 16 rows, else gemv_kernel<16, 8, 2>
cudaError_t old_launch(const Args& a) {
    if (a.R <= 16) return by_rt_narrow<2>(a, tile_of(a.R), 0);
    return launch<16, 8, 2>(a, 0);
}
cudaError_t new_launch(const Args& a) { return launch2<16, 8, 8>(a, 0); }

struct Out {
    std::vector<int> pick, gid, gmem, cnt;
    std::vector<float> wts, lg;
    int gcnt = 0;
};

Out fetch(const Bufs& b, int R) {
    Out o;
    o.pick.resize((size_t)R * SLOTS); o.wts.resize((size_t)R * SLOTS); o.lg.resize((size_t)R * E);
    o.gid.resize(GE); o.gmem.resize((size_t)GE * std::max(R, 1)); o.cnt.resize((size_t)(R + 15) / 16 * 16);
    CK(cudaMemcpy(o.pick.data(), b.pick, o.pick.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o.wts.data(), b.wts, o.wts.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o.lg.data(), b.lg, o.lg.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o.gid.data(), b.gid, o.gid.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o.gmem.data(), b.gmem, o.gmem.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&o.gcnt, b.gcnt, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o.cnt.data(), b.cnt, o.cnt.size() * 4, cudaMemcpyDeviceToHost));
    return o;
}

template <class T>
long long first_diff(const std::vector<T>& a, const std::vector<T>& b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (memcmp(&a[i], &b[i], sizeof(T)) != 0) return (long long)i;
    return -1;
}

// median ms of `reps` calls, each after a 256 MB write (L2 flushed: x and the gate start in DRAM)
template <class F>
double time_ms(F&& go, int reps, void* flush) {
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    std::vector<float> ms;
    for (int i = 0; i < reps + 1; ++i) {
        CK(cudaMemsetAsync(flush, i & 0xff, FLUSH, 0));
        CK(cudaEventRecord(t0, 0));
        CK(go());
        CK(cudaEventRecord(t1, 0));
        CK(cudaEventSynchronize(t1));
        float m = 0.0f;
        CK(cudaEventElapsedTime(&m, t0, t1));
        if (i > 0) ms.push_back(m);                  // the first call warms the module up
    }
    CK(cudaEventDestroy(t0));
    CK(cudaEventDestroy(t1));
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> rows = parse_list("4096,2048,4100,300,37,1,2,4,6,16");
    int reps = 21, samples = 4096, fold = 0;
    unsigned seed = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--rows") rows = parse_list(argv[i + 1]);
        else if (k == "--reps") reps = atoi(argv[i + 1]);
        else if (k == "--samples") samples = atoi(argv[i + 1]);
        else if (k == "--seed") seed = (unsigned)atoi(argv[i + 1]);
        else if (k == "--fold") fold = atoi(argv[i + 1]);
        else {
            fprintf(stderr, "router_bench [--rows 4096,2048,4100,300,37,1,2,4,6,16] [--reps 21] [--samples 4096] [--seed 1] [--fold 0]\n");
            return 2;
        }
    }
    const int rmax = *std::max_element(rows.begin(), rows.end());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ux(-3.0f, 3.0f), uw(-0.05f, 0.05f), ub(-0.5f, 0.5f);
    std::vector<uint16_t> hx((size_t)rmax * D), hw((size_t)E * D);
    std::vector<float> hb(E);
    for (auto& v : hx) v = bf16_of(ux(rng));
    for (auto& v : hw) v = bf16_of(uw(rng));
    for (auto& v : hb) v = ub(rng);
    uint16_t *x, *w;
    float* bias;
    void* flush;
    CK(cudaMalloc(&x, hx.size() * 2));
    CK(cudaMalloc(&w, hw.size() * 2));
    CK(cudaMalloc(&bias, E * 4));
    CK(cudaMalloc(&flush, FLUSH));
    CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bias, hb.data(), E * 4, cudaMemcpyHostToDevice));
    cudaFuncAttributes fa{};
    CK(cudaFuncGetAttributes(&fa, gemv2_kernel<16, 8, 8>));
    printf("gemv2_kernel<16,8,8>: %d registers, %zu B local, %zu B static shared\n", fa.numRegs, fa.localSizeBytes,
           fa.sharedSizeBytes);

    int bad = 0, checked = 0, oracle_bad = 0;
    double pf_old = 0.0, pf_new = 0.0;
    for (const int R : rows) {
        const bool decode = R <= 16;
        Bufs bo = make_bufs(R), bn = make_bufs(R);
        const Args ao = args_for(R, x, w, bias, bo, decode, fold != 0);
        Args an = args_for(R, x, w, bias, bn, decode, fold != 0);
        an.group = 0;                                // v2 folds no grouping (Zig selects it past 16 rows only)
        clear(bo, R);
        clear(bn, R);
        CK(old_launch(ao));
        CK(new_launch(an));
        CK(cudaDeviceSynchronize());
        const Out o = fetch(bo, R), n = fetch(bn, R);
        long long d[4] = {first_diff(o.lg, n.lg), first_diff(o.pick, n.pick), first_diff(o.wts, n.wts), -1};
        for (int c : n.cnt) if (c != 0) d[3] = 0;   // the counters are back at 0 (a graph replay starts clean)
        for (int c : o.cnt) if (c != 0) d[3] = 0;
        const bool same = d[0] < 0 && d[1] < 0 && d[2] < 0 && d[3] < 0;
        bad += !same;
        std::mt19937 pick_rng(seed * 7919u + (unsigned)R);
        int ob = 0, ns = std::min(samples, R * E);
        for (int s = 0; s < ns; ++s) {
            const int idx = R * E <= samples ? s : (int)(pick_rng() % (unsigned)(R * E));
            const int r = idx / E, e = idx % E;
            const uint32_t want = bits(oracle(hx.data() + (size_t)r * D, hw.data() + (size_t)e * D));
            ob += bits(n.lg[idx]) != want || bits(o.lg[idx]) != want;
        }
        checked += ns;
        oracle_bad += ob;
        const double t_old = time_ms([&] { return old_launch(ao); }, reps, flush);
        const double t_new = time_ms([&] { return new_launch(an); }, reps, flush);
        if (R == 2048) { pf_old = t_old; pf_new = t_new; }
        printf("R=%-5d %s old=%s %.4f ms  v2=%.4f ms  x%.2f  lg/pick/wts/cnt %s (first diffs %lld %lld %lld %lld)  "
               "oracle %d/%d%s\n", R, decode ? "decode " : "prefill", decode ? "narrow<rt,2>" : "gemv<16,8,2>", t_old,
               t_new, t_old / t_new, same ? "identical" : "DIFFER", d[0], d[1], d[2], d[3], ns - ob, ns,
               decode ? "  (group: old only; v2 is not selected at <= 16 rows)" : "");
        free_bufs(bo);
        free_bufs(bn);
    }
    printf("SUMMARY router_bench v2=gemv2<16,8,8> fold=%d shapes=%zu bytes=%s oracle=%d/%d prefill2048_old_ms=%.4f "
           "prefill2048_v2_ms=%.4f speedup=%.2f\n", fold, rows.size(), bad ? "DIFFER" : "identical", checked - oracle_bad,
           checked, pf_old, pf_new, pf_new > 0 ? pf_old / pf_new : 0.0);
    CK(cudaFree(x));
    CK(cudaFree(w));
    CK(cudaFree(bias));
    CK(cudaFree(flush));
    return bad || oracle_bad ? 1 : 0;
}
