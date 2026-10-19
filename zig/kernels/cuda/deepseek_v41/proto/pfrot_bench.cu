// pfrot (rot2_kernel, pfdr_kernel) against prod's rot_in_kernel twice / pfd_kernel x4 + rot_in_kernel: bytes, times.
#include "../linear.cu"
#include "../pfrot.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); exit(1); } } while (0)

namespace {

// wo_a at prod's shape: 4 groups of [M, 4096] @ [4096, 1024] into z [M, 4096] bf16, lanes K2 10, prod's cfg 5 group 16
constexpr int K2 = 10, G = 4, KW = 4096, NW = 1024, ZN = G * NW, CFG = 5, GROUP = 16;
using C5 = dsv41_pfd::Cfg<128, 2, 4, 2, 2>;
using L5 = dsv41_pfd::Lay<K2, true, C5>;
constexpr long long STRIDE_K = 8 * 4 * K2, STRIDE_NB = (long long)(KW / 16) * STRIDE_K;
constexpr size_t FLUSH = 256u << 20;
constexpr double DRAM_GBS = 273.0;
enum Fill : int { F_F16 = 0, F_BF16 = 1, F_WORDS = 2, F_SIGNED = 3 };

}  // namespace

// deterministic hash fill: fp16 / bf16 ~U(lo, hi) (F_SIGNED: a random sign on top), or raw 32-bit words
__global__ void fill_kernel(void* p, size_t n, uint32_t seed, float lo, float hi, int kind) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        if (kind == F_WORDS) {
            static_cast<uint32_t*>(p)[i] = x;
            continue;
        }
        float v = lo + (hi - lo) * (x & 0xffffff) / 16777216.f;
        if (kind == F_SIGNED && (x >> 31)) v = -v;
        uint16_t* q = static_cast<uint16_t*>(p);
        q[i] = kind == F_BF16 ? __bfloat16_as_ushort(__float2bfloat16_rn(v)) : __half_as_ushort(__float2half_rn(v));
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

std::vector<uint16_t> fetch(const void* p, size_t n) {
    std::vector<uint16_t> h(n);
    CK(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost));
    return h;
}

// two [rows, cols] 16-bit buffers element for element: prints the count and the first difference
bool same16(const char* name, const void* ref, const void* cand, int rows, int cols) {
    const size_t n = (size_t)rows * cols;
    const std::vector<uint16_t> hr = fetch(ref, n), hc = fetch(cand, n);
    long long nd = 0, first = -1;
    for (size_t i = 0; i < n; ++i)
        if (hr[i] != hc[i]) { if (first < 0) first = (long long)i; ++nd; }
    printf(" %s %lld", name, nd);
    if (nd) printf(" (first row %lld col %lld: ref 0x%04x cand 0x%04x)", first / cols, first % cols, hr[first],
                   hc[first]);
    return nd == 0;
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

void print_attr(const char* name, const void* k, size_t dyn) {
    cudaFuncAttributes fa{};
    CK(cudaFuncGetAttributes(&fa, k));
    printf("  %s: %d regs/thread, %zu B local, %zu B static smem, %zu B dynamic smem\n", name, fa.numRegs,
           fa.localSizeBytes, fa.sharedSizeBytes, dyn);
}

// prod's input rotation (linear.cu): grid (ceil(K / 512), M), 128 threads, x bf16
cudaError_t rot_in(const void* x, const half* suh, half* xh, int M, int K) {
    tf_exl3_linear::rot_in_kernel<<<dim3((unsigned)((K / 128 + 3) / 4), (unsigned)M), 128, 0, 0>>>(x, 1, suh, xh, K);
    return cudaGetLastError();
}

// the candidate at configuration C: pfd's grid, threads and dynamic shared memory
template <class C>
cudaError_t launch_pfdr(const dsv41_pfd::Args& a, const dsv41_pfrot::RotOut& r) {
    using L = dsv41_pfd::Lay<K2, true, C>;
    auto kern = dsv41_pfrot::pfdr_kernel<K2, true, C>;
    static bool ready = false;
    if (!ready) {
        const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, L::SMEM);
        if (e != cudaSuccess) return e;
        ready = true;
    }
    const int nm = (a.M + C::BM - 1) / C::BM;
    kern<<<(unsigned)(nm * (a.N >> 7)), C::THREADS, L::SMEM, 0>>>(a, r);
    return cudaGetLastError();
}

cudaError_t run_pfdr(int cfg, const dsv41_pfd::Args& a, const dsv41_pfrot::RotOut& r) {
#define PFB_RUN(ID, BM, WM, WN, KS, NST) \
    if (cfg == ID) return launch_pfdr<dsv41_pfd::Cfg<BM, WM, WN, KS, NST>>(a, r);
    DSV41_PFD_CFGS(PFB_RUN)
#undef PFB_RUN
    return cudaErrorInvalidValue;
}

cudaError_t run_pfd(int cfg, const dsv41_pfd::Args& a) {
    if (!dsv41_pfd::run<K2, true>(a, cfg, 0)) return cudaErrorInvalidValue;
    return cudaGetLastError();
}

// part (a): rot2_kernel once against rot_in_kernel twice over one bf16 x [M, K]
struct Rot2 {
    int M, K;
    uint16_t* x;
    half *suh0, *suh1, *r0, *r1, *c0, *c1;
};

bool rot2_part(const Rot2& b, int reps, void* flush, std::string& summary) {
    const int M = b.M, K = b.K;
    const size_t n = (size_t)M * K;
    for (half* p : {b.r0, b.r1, b.c0, b.c1}) CK(cudaMemset(p, 0xff, n * 2));
    auto ref = [&] {
        const cudaError_t e = rot_in(b.x, b.suh0, b.r0, M, K);
        return e != cudaSuccess ? e : rot_in(b.x, b.suh1, b.r1, M, K);
    };
    const dsv41_pfrot::Rot2Args ra{b.x, b.suh0, b.suh1, b.c0, b.c1, 1, M, K};
    auto cand = [&] {
        dsv41_pfrot::rot2_kernel<<<dim3((unsigned)((K / 128 + 3) / 4), (unsigned)M), 128, 0, 0>>>(ra);
        return cudaGetLastError();
    };
    CK(ref());
    CK(cand());
    CK(cudaDeviceSynchronize());
    printf("  rot2 bytes:");
    bool ok = same16("xh0", b.r0, b.c0, M, K);
    ok &= same16("xh1", b.r1, b.c1, M, K);
    printf(" -> %s\n", ok ? "PASS" : "FAIL");
    const Ms tr = time_ms(ref, reps, flush), tc = time_ms(cand, reps, flush);
    const double fr = 4.0 * n * 2 / (DRAM_GBS * 1e6), fc = 3.0 * n * 2 / (DRAM_GBS * 1e6);
    printf("  DRAM floor: ref (x twice + 2 xh) %.1f MB %.4f ms, rot2 (x once + 2 xh) %.1f MB %.4f ms at %.0f GB/s\n",
           4.0 * n * 2 / 1e6, fr, 3.0 * n * 2 / 1e6, fc, DRAM_GBS);
    printf("  ref  2 x rot_in median %.4f ms (min %.4f): %.2fx its floor\n", tr.med, tr.min, tr.med / fr);
    printf("  rot2 1 x rot2   median %.4f ms (min %.4f): %.2fx its floor; saving %+.4f ms (%+.1f%%)\n", tc.med,
           tc.min, tc.med / fc, tr.med - tc.med, (tr.med / tc.med - 1) * 100);
    char buf[120];
    snprintf(buf, sizeof buf, " rot2 %.4fms vs %.4fms", tr.med, tc.med);
    summary += buf;
    return ok;
}

// part (b): wo_a's 4 pfd GEMMs into z, then wo_b's rotation of z, against 4 pfdr launches writing both
struct Worot {
    int M;
    half *xh, *svh, *suh;
    uint32_t* T;
    uint16_t *zr, *zc;
    half *hr, *hc;
};

dsv41_pfd::Args gemm_args(const Worot& b, int g, uint16_t* z) {
    dsv41_pfd::Args a{};
    a.xh = b.xh + (size_t)g * b.M * KW;
    a.T = b.T + (size_t)g * (NW / 128) * STRIDE_NB;
    a.stride_k = STRIDE_K;
    a.stride_nb = STRIDE_NB;
    a.svh = b.svh + g * NW;
    a.bias = nullptr;
    a.out = z + g * NW;
    a.o_stride = ZN;
    a.M = b.M; a.K = KW; a.N = NW; a.group = GROUP; a.out_type = dsv41_pfd::OUT_BF16;
    return a;
}

bool worot_part(const Worot& b, int reps, void* flush, std::string& summary) {
    const int M = b.M;
    const size_t n = (size_t)M * ZN;
    dsv41_pfd::Args ar[G], ac[G];
    dsv41_pfrot::RotOut rc[G];
    for (int g = 0; g < G; ++g) {
        ar[g] = gemm_args(b, g, b.zr);
        ac[g] = gemm_args(b, g, b.zc);
        rc[g] = {b.suh + g * NW, b.hc + g * NW, ZN};
    }
    auto gemms = [&] {
        for (int g = 0; g < G; ++g) {
            const cudaError_t e = run_pfd(CFG, ar[g]);
            if (e != cudaSuccess) return e;
        }
        return cudaSuccess;
    };
    auto ref = [&] {
        const cudaError_t e = gemms();
        return e != cudaSuccess ? e : rot_in(b.zr, b.suh, b.hr, M, ZN);
    };
    auto cand_at = [&](int cfg) {
        for (int g = 0; g < G; ++g) {
            const cudaError_t e = run_pfdr(cfg, ac[g], rc[g]);
            if (e != cudaSuccess) return e;
        }
        return cudaSuccess;
    };
    auto cand = [&] { return cand_at(CFG); };
    CK(cudaMemset(b.zr, 0xff, n * 2));
    CK(cudaMemset(b.hr, 0xff, n * 2));
    CK(ref());
    CK(cudaDeviceSynchronize());
    bool ok = true;
    for (int c = 0; c < DSV41_PFD_NCFG; ++c) {
        int o[5];
        dsv41_pfd::cfg_of(c, o);
        CK(cudaMemset(b.zc, 0xff, n * 2));
        CK(cudaMemset(b.hc, 0xff, n * 2));
        CK(cand_at(c));
        CK(cudaDeviceSynchronize());
        printf("  worot bytes, pfdr cfg %d (BM %d WM %d WN %d KS %d NST %d)%s:", c, o[0], o[1], o[2], o[3], o[4],
               c == CFG ? " prod's" : "");
        bool same = same16("z", b.zr, b.zc, M, ZN);
        same &= same16("xhb", b.hr, b.hc, M, ZN);
        printf(" -> %s\n", same ? "PASS" : "FAIL");
        ok &= same;
    }
    const Ms tg = time_ms(gemms, reps, flush), tr = time_ms(ref, reps, flush), tc = time_ms(cand, reps, flush);
    const double flop = 2.0 * M * KW * (double)ZN;
    printf("  4 x pfd alone        median %.4f ms (min %.4f): %.1f TFLOP/s\n", tg.med, tg.min, flop / tg.med / 1e9);
    printf("  ref  4 x pfd + rot_in median %.4f ms (min %.4f): rot_in adds %.4f ms\n", tr.med, tr.min,
           tr.med - tg.med);
    printf("  cand 4 x pfdr         median %.4f ms (min %.4f): saving %+.4f ms (%+.1f%%), post-pass costs %+.4f ms\n",
           tc.med, tc.min, tr.med - tc.med, (tr.med / tc.med - 1) * 100, tc.med - tg.med);
    char buf[120];
    snprintf(buf, sizeof buf, " worot %.4fms vs %.4fms", tr.med, tc.med);
    summary += buf;
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    std::vector<int> rows = parse_list("4096,2048,300");
    std::string part = "both";
    int K = 5120, reps = 15;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--rows") rows = parse_list(argv[i + 1]);
        else if (k == "--part") part = argv[i + 1];
        else if (k == "--k") K = atoi(argv[i + 1]);
        else if (k == "--reps") reps = atoi(argv[i + 1]);
        else {
            fprintf(stderr, "pfrot_bench [--part rot2|worot|both] [--rows 4096,2048,300] [--k 5120] [--reps 15]\n");
            return 2;
        }
    }
    if (argc % 2 == 0) { fprintf(stderr, "pfrot_bench: %s needs a value\n", argv[argc - 1]); return 2; }
    const bool do_rot2 = part == "rot2" || part == "both", do_worot = part == "worot" || part == "both";
    if (rows.empty() || reps < 1 || K < 128 || K % 128 || !(do_rot2 || do_worot)) {
        fprintf(stderr, "need --rows, --reps >= 1, --k a multiple of 128 and --part rot2, worot or both\n");
        return 2;
    }
    for (const int r : rows) if (r < 1 || r > 65535) { fprintf(stderr, "bad row count %d\n", r); return 2; }
    int o5[5];
    dsv41_pfd::cfg_of(CFG, o5);
    if (o5[0] != C5::BM || o5[1] != C5::WM || o5[2] != C5::WN || o5[3] != C5::KS || o5[4] != C5::NST) {
        fprintf(stderr, "pfdense_args.h's cfg %d is not Cfg<128, 2, 4, 2, 2>: update C5\n", CFG);
        return 2;
    }
    const int rmax = *std::max_element(rows.begin(), rows.end());

    int devi = 0, sms = 0, clk = 0;
    CK(cudaGetDevice(&devi));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, devi));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, devi));
    CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, devi));
    printf("device %s: %d SMs, clock %d MHz; part %s, rot2 K %d, worot %d x [M, %d] @ [%d, %d] lanes K2 %d cfg %d "
           "group %d\n", prop.name, sms, clk / 1000, part.c_str(), K, G, KW, KW, NW, K2, CFG, GROUP);
    void* flush;
    CK(cudaMalloc(&flush, FLUSH));

    Rot2 r2{};
    if (do_rot2) {
        const size_t n = (size_t)rmax * K;
        r2 = {0, K, alloc<uint16_t>(n), alloc<half>(K), alloc<half>(K), alloc<half>(n), alloc<half>(n),
              alloc<half>(n), alloc<half>(n)};
        fill(r2.x, n, 1, -2.f, 2.f, F_BF16);
        fill(r2.suh0, K, 2, 0.9f, 1.1f, F_SIGNED);
        fill(r2.suh1, K, 3, 0.9f, 1.1f, F_SIGNED);
    }
    Worot wr{};
    if (do_worot) {
        const size_t n = (size_t)rmax * ZN, words = (size_t)G * (NW / 128) * STRIDE_NB;
        wr = {0, alloc<half>((size_t)G * rmax * KW), alloc<half>(ZN), alloc<half>(ZN), alloc<uint32_t>(words),
              alloc<uint16_t>(n), alloc<uint16_t>(n), alloc<half>(n), alloc<half>(n)};
        fill(wr.svh, ZN, 5, 0.5f, 1.5f, F_F16);
        fill(wr.suh, ZN, 6, 0.9f, 1.1f, F_SIGNED);
        fill(wr.T, words, 7, 0.f, 0.f, F_WORDS);
    }
    CK(cudaDeviceSynchronize());
    print_attr("rot_in_kernel", (const void*)tf_exl3_linear::rot_in_kernel, 0);
    print_attr("rot2_kernel", (const void*)dsv41_pfrot::rot2_kernel, 0);
    if (do_worot) {
        CK(cudaFuncSetAttribute(dsv41_pfd::pfd_kernel<K2, true, C5>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                L5::SMEM));
        print_attr("pfd_kernel<10, true, cfg 5>", (const void*)dsv41_pfd::pfd_kernel<K2, true, C5>, L5::SMEM);
        print_attr("pfdr_kernel<10, true, cfg 5>", (const void*)dsv41_pfrot::pfdr_kernel<K2, true, C5>, L5::SMEM);
    }

    bool rot2_ok = true, worot_ok = true;
    std::string summary;
    for (const int M : rows) {
        printf("rows %d:\n", M);
        char buf[40];
        snprintf(buf, sizeof buf, " rows=%d:", M);
        summary += buf;
        if (do_rot2) {
            r2.M = M;
            rot2_ok &= rot2_part(r2, reps, flush, summary);
        }
        if (do_worot) {
            wr.M = M;
            fill(wr.xh, (size_t)G * M * KW, 4, -0.125f, 0.125f, F_F16);
            printf("  pfd grid %d CTAs a group (%d row tiles x %d column blocks), %d threads, %d B dynamic smem\n",
                   (M + C5::BM - 1) / C5::BM * (NW / 128), (M + C5::BM - 1) / C5::BM, NW / 128, C5::THREADS,
                   L5::SMEM);
            worot_ok &= worot_part(wr, reps, flush, summary);
        }
    }
    printf("SUMMARY pfrot rot2=%s worot=%s%s\n", do_rot2 ? (rot2_ok ? "PASS" : "FAIL") : "SKIP",
           do_worot ? (worot_ok ? "PASS" : "FAIL") : "SKIP", summary.c_str());
    if (do_rot2)
        for (void* p : {(void*)r2.x, (void*)r2.suh0, (void*)r2.suh1, (void*)r2.r0, (void*)r2.r1, (void*)r2.c0,
                        (void*)r2.c1})
            CK(cudaFree(p));
    if (do_worot)
        for (void* p : {(void*)wr.xh, (void*)wr.svh, (void*)wr.suh, (void*)wr.T, (void*)wr.zr, (void*)wr.zc,
                        (void*)wr.hr, (void*)wr.hc})
            CK(cudaFree(p));
    CK(cudaFree(flush));
    return rot2_ok && worot_ok ? 0 : 1;
}
