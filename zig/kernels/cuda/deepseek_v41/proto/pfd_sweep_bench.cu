// pfdense (pfd_kernel, lanes) GB10 re-tune: every config x group per shape, bytes vs cfg 5 group 8, times, a table.
#include "../pfdense.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); exit(1); } } while (0)

namespace {

constexpr int REF_CFG = 5, REF_GROUP = 8, SMALL_ROWS = 512, TUNE_ROWS = 4096;
constexpr int GROUPS[] = {1, 4, 8, 16};
constexpr size_t FLUSH = 256u << 20;
enum Fill : int { F_F16 = 0, F_WORDS = 2 };

// a projection [M, K] @ [K, N] and the prod table's pick (spark config/pfdense-table.json, tuned at 2048 rows)
struct Shape { const char* name; int K, N, cfg, group, small; };

const Shape SHAPES[] = {
    {"wkv", 5120, 512, 5, 1, 6},
    {"wo_a", 4096, 1024, 5, 16, 6},
    {"wq_a", 5120, 1280, 5, 4, 4},
    {"wq_b", 1280, 16384, 0, 4, 0},
    {"wo_b", 4096, 5120, 5, 16, 0},
};
constexpr int NBASE = 3;

}  // namespace

// deterministic hash fill: fp16 ~U(lo, hi), or raw 32-bit words
__global__ void fill_kernel(void* p, size_t n, uint32_t seed, float lo, float hi, int kind) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        if (kind == F_WORDS) {
            static_cast<uint32_t*>(p)[i] = x;
            continue;
        }
        const float v = lo + (hi - lo) * (x & 0xffffff) / 16777216.f;
        static_cast<uint16_t*>(p)[i] = __half_as_ushort(__float2half_rn(v));
    }
}

namespace {

void fill(void* p, size_t n, uint32_t seed, float lo, float hi, int kind) {
    fill_kernel<<<1024, 256>>>(p, n, seed, lo, hi, kind);
    CK(cudaGetLastError());
}

template <typename T> T* alloc(size_t n) {
    T* d;
    CK(cudaMalloc(&d, n * sizeof(T) + 16));
    return d;
}

// elements of two 16-bit device buffers that differ, and the first one
long long diff16(const void* ref, const void* cand, size_t n, long long& first) {
    std::vector<uint16_t> hr(n), hc(n);
    CK(cudaMemcpy(hr.data(), ref, n * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hc.data(), cand, n * 2, cudaMemcpyDeviceToHost));
    long long nd = 0;
    first = -1;
    for (size_t i = 0; i < n; ++i)
        if (hr[i] != hc[i]) { if (first < 0) first = (long long)i; ++nd; }
    return nd;
}

struct Ms { double med, min; };

// median / min ms of `reps` timed runs (one warm-up first), each after a 256 MB write (L2 flushed)
template <class F>
Ms time_ms(F&& go, int reps, void* flush) {
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
        if (i > 0) ms.push_back(m);
    }
    CK(cudaEventDestroy(t0));
    CK(cudaEventDestroy(t1));
    std::sort(ms.begin(), ms.end());
    return {ms[ms.size() / 2], ms[0]};
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

// pfdense.cuh's run / describe at the lanes width k2 (8, 10 or 12)
bool run_lanes(int k2, const dsv41_pfd::Args& a, int cfg) {
    if (k2 == 8) return dsv41_pfd::run<8, true>(a, cfg, 0);
    if (k2 == 10) return dsv41_pfd::run<10, true>(a, cfg, 0);
    if (k2 == 12) return dsv41_pfd::run<12, true>(a, cfg, 0);
    return false;
}
void describe_lanes(int k2, int cfg, long long (&o)[8]) {
    if (k2 == 8) dsv41_pfd::describe<8, true>(cfg, o);
    else if (k2 == 10) dsv41_pfd::describe<10, true>(cfg, o);
    else dsv41_pfd::describe<12, true>(cfg, o);
}

struct Point { int cfg, group; double ms, min, tflops; long long ctas, occ, diffs; double waves; };

struct ShapeBuf {
    half *xh, *svh;
    uint32_t* T;
    uint16_t *ref, *cand;
    long long stride_k, stride_nb;
};

dsv41_pfd::Args args_of(const Shape& s, const ShapeBuf& b, int M, int group, uint16_t* out) {
    dsv41_pfd::Args a{};
    a.xh = b.xh;
    a.T = b.T;
    a.stride_k = b.stride_k;
    a.stride_nb = b.stride_nb;
    a.svh = b.svh;
    a.bias = nullptr;
    a.out = out;
    a.o_stride = s.N;
    a.M = M; a.K = s.K; a.N = s.N; a.group = group; a.out_type = dsv41_pfd::OUT_BF16;
    return a;
}

void print_point(const char* tag, const Point& p) {
    printf("    %-6s cfg %d group %2d  %8.4f ms (min %8.4f)  %6.1f TFLOP/s  %6lld CTAs  occ %lld  %6.2f waves  %s\n",
           tag, p.cfg, p.group, p.ms, p.min, p.tflops, p.ctas, p.occ, p.waves, p.diffs ? "FAIL" : "bytes ok");
}

}  // namespace

int main(int argc, char** argv) {
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    std::vector<int> rows = parse_list("4096,2048");
    std::string shapes = "base";
    int k2 = 10, reps = 15;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--rows") rows = parse_list(argv[i + 1]);
        else if (k == "--k2") k2 = atoi(argv[i + 1]);
        else if (k == "--shapes") shapes = argv[i + 1];
        else if (k == "--reps") reps = atoi(argv[i + 1]);
        else {
            fprintf(stderr, "pfd_sweep_bench [--k2 8|10|12] [--rows 4096,2048] [--shapes base|all] [--reps 15]\n");
            return 2;
        }
    }
    if (argc % 2 == 0) { fprintf(stderr, "pfd_sweep_bench: %s needs a value\n", argv[argc - 1]); return 2; }
    if (rows.empty() || reps < 1 || (k2 != 8 && k2 != 10 && k2 != 12) || (shapes != "base" && shapes != "all")) {
        fprintf(stderr, "need --rows, --reps >= 1, --k2 8, 10 or 12 and --shapes base or all\n");
        return 2;
    }
    for (const int r : rows) if (r < 1) { fprintf(stderr, "bad row count %d\n", r); return 2; }
    const int rmax = *std::max_element(rows.begin(), rows.end());
    const int tune = std::find(rows.begin(), rows.end(), TUNE_ROWS) != rows.end() ? TUNE_ROWS : rmax;
    const int nshape = shapes == "all" ? (int)(sizeof SHAPES / sizeof SHAPES[0]) : NBASE;

    int devi = 0, sms = 0, clk = 0;
    CK(cudaGetDevice(&devi));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, devi));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, devi));
    CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, devi));
    printf("device %s: %d SMs, clock %d MHz; lanes K2 %d, out bf16, reference cfg %d group %d, table rows %d\n",
           prop.name, sms, clk / 1000, k2, REF_CFG, REF_GROUP, tune);
    for (int c = 0; c < DSV41_PFD_NCFG; ++c) {
        long long o[8];
        describe_lanes(k2, c, o);
        printf("  cfg %d: BM %lld WM %lld WN %lld KS %lld NST %lld, %lld threads, %lld B smem, %lld blocks an SM\n", c,
               o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);
    }
    void* flush;
    CK(cudaMalloc(&flush, FLUSH));

    bool bytes_ok = true;
    std::string summary, json;
    for (int si = 0; si < nshape; ++si) {
        const Shape& s = SHAPES[si];
        ShapeBuf b{};
        b.stride_k = 8 * 4 * k2;
        b.stride_nb = (long long)(s.K / 16) * b.stride_k;
        const size_t words = (size_t)(s.N / 128) * b.stride_nb, nout = (size_t)rmax * s.N;
        b.xh = alloc<half>((size_t)rmax * s.K);
        b.svh = alloc<half>(s.N);
        b.T = alloc<uint32_t>(words);
        b.ref = alloc<uint16_t>(nout);
        b.cand = alloc<uint16_t>(nout);
        fill(b.xh, (size_t)rmax * s.K, 11 + si, -0.125f, 0.125f, F_F16);
        fill(b.svh, s.N, 21 + si, 0.5f, 1.5f, F_F16);
        fill(b.T, words, 31 + si, 0.f, 0.f, F_WORDS);
        CK(cudaDeviceSynchronize());
        Point best_t{}, prod_t{};
        for (const int M : rows) {
            const double flop = 2.0 * M * s.K * (double)s.N;
            printf("%s [%d, %d] @ [%d, %d], rows %d:\n", s.name, M, s.K, s.K, s.N, M);
            CK(cudaMemset(b.ref, 0xff, (size_t)M * s.N * 2));
            if (!run_lanes(k2, args_of(s, b, M, REF_GROUP, b.ref), REF_CFG)) { fprintf(stderr, "no cfg\n"); return 2; }
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            std::vector<Point> pts;
            for (int c = 0; c < DSV41_PFD_NCFG; ++c) {
                long long o[8];
                describe_lanes(k2, c, o);
                if (s.K % (16 * o[3])) {
                    printf("  cfg %d skipped: stage K %lld does not divide K %d\n", c, 16 * o[3], s.K);
                    continue;
                }
                for (const int g : GROUPS) {
                    const dsv41_pfd::Args a = args_of(s, b, M, g, b.cand);
                    CK(cudaMemset(b.cand, 0xff, (size_t)M * s.N * 2));
                    if (!run_lanes(k2, a, c)) { fprintf(stderr, "no cfg %d\n", c); return 2; }
                    CK(cudaGetLastError());
                    CK(cudaDeviceSynchronize());
                    long long first = -1;
                    const long long nd = diff16(b.ref, b.cand, (size_t)M * s.N, first);
                    if (nd) {
                        printf("  cfg %d group %d: %lld of %lld bf16 differ from cfg %d group %d (first row %lld col "
                               "%lld) -> FAIL\n", c, g, nd, (long long)M * s.N, REF_CFG, REF_GROUP, first / s.N,
                               first % s.N);
                        bytes_ok = false;
                    }
                    auto go = [&] { return run_lanes(k2, a, c) ? cudaGetLastError() : cudaErrorInvalidValue; };
                    const Ms t = time_ms(go, reps, flush);
                    Point p{c, g, t.med, t.min, flop / t.med / 1e9, 0, o[7], nd, 0.0};
                    p.ctas = (long long)((M + o[0] - 1) / o[0]) * (s.N / 128);
                    p.waves = (double)p.ctas / ((double)sms * std::max(1LL, o[7]));
                    pts.push_back(p);
                }
            }
            std::sort(pts.begin(), pts.end(), [](const Point& x, const Point& y) { return x.ms < y.ms; });
            int rank = 0;
            for (const Point& p : pts) {
                char tag[16];
                snprintf(tag, sizeof tag, "#%d", ++rank);
                print_point(tag, p);
            }
            const auto prod = std::find_if(pts.begin(), pts.end(),
                                           [&](const Point& p) { return p.cfg == s.cfg && p.group == s.group; });
            if (pts.empty() || prod == pts.end()) { fprintf(stderr, "prod pick not measured\n"); return 2; }
            print_point("prod", *prod);
            print_point("best", pts[0]);
            printf("  best vs prod: %.3fx (%+.4f ms)\n", prod->ms / pts[0].ms, prod->ms - pts[0].ms);
            if (M == tune) {
                best_t = pts[0];
                prod_t = *prod;
            }
        }
        char buf[320];
        snprintf(buf, sizeof buf, "%s\"%d,%d\": {\"cfg\": %d, \"group\": %d, \"small\": %d, \"small_rows\": %d, "
                 "\"k2\": %d, \"name\": \"%s\", \"ms\": %.4f, \"tflops_%d\": %.1f}", json.empty() ? "" : ", ", s.K,
                 s.N, best_t.cfg, best_t.group, s.small, SMALL_ROWS, k2, s.name, best_t.ms, tune, best_t.tflops);
        json += buf;
        printf("TABLE %s %d,%d: {\"cfg\": %d, \"group\": %d, \"small\": %d, \"small_rows\": %d}\n", s.name, s.K, s.N,
               best_t.cfg, best_t.group, s.small, SMALL_ROWS);
        snprintf(buf, sizeof buf, " %s best cfg%d g%d %.4fms vs prod cfg%d g%d %.4fms (%.3fx)", s.name, best_t.cfg,
                 best_t.group, best_t.ms, prod_t.cfg, prod_t.group, prod_t.ms, prod_t.ms / best_t.ms);
        summary += buf;
        for (void* p : {(void*)b.xh, (void*)b.svh, (void*)b.T, (void*)b.ref, (void*)b.cand}) CK(cudaFree(p));
    }
    printf("JSON {%s}\n", json.c_str());
    printf("SUMMARY pfd_sweep k2=%d bytes=%s rows=%d:%s\n", k2, bytes_ok ? "PASS" : "FAIL", tune, summary.c_str());
    CK(cudaFree(flush));
    return bytes_ok ? 0 : 1;
}
