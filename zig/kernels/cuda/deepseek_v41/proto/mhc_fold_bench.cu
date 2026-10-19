// mHC prefill fold (dsv41_mhc_fold::fold_kernel) against prod's site_kernel + a separate finish<true>: bytes, oracle.
#include "../mhc_pf_fold.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); exit(1); } } while (0)

namespace {

// one GB10 rank of the prefill boundary: D 5,120, 40 column blocks, 4 streams, world 2 bf16 gathered partials
constexpr int D = dsv41_mhc_fold::D, NB = dsv41_mhc_fold::NB, BM = dsv41_mhc_fold::BM, PW = dsv41_mhc::PW;
constexpr int W = 2, NMIX = 24;
constexpr int FSMEM = dsv41_mhc::MAXR * dsv41_mhc::ROWB + 2 * dsv41_mhc::STAGE4 * 16;
constexpr float EPS = 1e-20f, HC_EPS = 1e-6f, POST_ALPHA = 2.0f;
constexpr int ITERS = 20;
constexpr size_t FLUSH = 256u << 20;
constexpr double DRAM_GBS = 273.0, CTOL = 1e-5, TRITON_FINISH_MS = 0.828;

}  // namespace

// the separate-pass finish (Triton _finish_k's shape): CTA = a 16-row tile, fold_kernel's finish arguments
__global__ void __launch_bounds__(128) ref_finish_kernel(const dsv41_mhc_fold::Args a) {
    extern __shared__ __align__(16) unsigned char sm[];
    const int t = threadIdx.x, r0 = blockIdx.x * BM;
    dsv41_mhc::Args f{};
    f.part = a.s.part + static_cast<long long>(r0) * 4 * NB * PW;
    f.c = a.s.c + static_cast<long long>(r0) * D;
    f.base = a.base; f.scale = a.scale;
    f.opre = a.opre + r0 * 4; f.opost = a.opost + r0 * 4; f.ocomb = a.ocomb + r0 * 16;
    f.nw = a.nw; f.nwbf16 = a.nwbf16;
    f.out = a.out + static_cast<long long>(r0) * D;
    f.R = min(BM, a.s.R - r0);
    f.eps = a.eps; f.hc_eps = a.hc_eps; f.post_alpha = a.post_alpha; f.iters = a.iters;
    dsv41_mhc::finish<true>(f, reinterpret_cast<float*>(sm),
                            reinterpret_cast<float4*>(sm + dsv41_mhc::MAXR * dsv41_mhc::ROWB), t);
}

// deterministic hash fill ~U(lo, hi): bf16 bits (bf) or fp32
__global__ void fill_kernel(void* p, size_t n, uint32_t seed, float lo, float hi, int bf) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        const float v = lo + (hi - lo) * (x & 0xffffff) / 16777216.f;
        if (bf) static_cast<uint16_t*>(p)[i] = static_cast<uint16_t>(dsv41_mhc_pf::bfbits(v));
        else static_cast<float*>(p)[i] = v;
    }
}

namespace {

void fill(void* p, size_t n, uint32_t seed, float lo, float hi, bool bf) {
    fill_kernel<<<1024, 256>>>(p, n, seed, lo, hi, bf ? 1 : 0);
    CK(cudaGetLastError());
}

template <typename T> T* alloc(size_t n) {
    T* d;
    CK(cudaMalloc(&d, n * sizeof(T) + 16));
    return d;
}

// one side's outputs: streams (in place), collapsed row, partials, normed input, the next site's coefficients
struct Side { uint16_t *x, *c, *out; float *part, *opre, *opost, *ocomb; };

Side make_side(int rmax) {
    return {alloc<uint16_t>((size_t)rmax * 4 * D), alloc<uint16_t>((size_t)rmax * D), alloc<uint16_t>((size_t)rmax * D),
            alloc<float>((size_t)rmax * 4 * NB * PW), alloc<float>((size_t)rmax * 4), alloc<float>((size_t)rmax * 4),
            alloc<float>((size_t)rmax * 16)};
}
void free_side(Side& s) {
    for (void* p : {(void*)s.x, (void*)s.c, (void*)s.out, (void*)s.part, (void*)s.opre, (void*)s.opost,
                    (void*)s.ocomb})
        CK(cudaFree(p));
}

struct Buf { const char* name; size_t bytes, elem; void* (*get)(const Side&); };

const Buf BUFS[] = {
    {"xout", 4 * D * 2, 2, [](const Side& s) { return (void*)s.x; }},
    {"c", D * 2, 2, [](const Side& s) { return (void*)s.c; }},
    {"part", 4 * NB * PW * 4, 4, [](const Side& s) { return (void*)s.part; }},
    {"out", D * 2, 2, [](const Side& s) { return (void*)s.out; }},
    {"opre", 4 * 4, 4, [](const Side& s) { return (void*)s.opre; }},
    {"opost", 4 * 4, 4, [](const Side& s) { return (void*)s.opost; }},
    {"ocomb", 16 * 4, 4, [](const Side& s) { return (void*)s.ocomb; }},
};

// the same 0xFF pattern in every output a side writes (the streams are restored from the pristine copy)
void reset_side(const Side& s, const uint16_t* x0, int R) {
    CK(cudaMemcpy(s.x, x0, (size_t)R * 4 * D * 2, cudaMemcpyDeviceToDevice));
    for (const Buf& b : BUFS)
        if (std::string(b.name) != "xout") CK(cudaMemset(b.get(s), 0xff, b.bytes * R));
}

std::vector<uint8_t> fetch(const void* p, size_t n) {
    std::vector<uint8_t> h(n);
    CK(cudaMemcpy(h.data(), p, n, cudaMemcpyDeviceToHost));
    return h;
}

// every output byte for byte, all R rows: elements that differ a buffer (xout in mode 0 only)
bool compare(const Side& ref, const Side& cand, int R, int mode, const char* tag) {
    bool ok = true;
    printf("  bytes (%s):", tag);
    for (const Buf& b : BUFS) {
        if (mode != 0 && std::string(b.name) == "xout") continue;
        const std::vector<uint8_t> hr = fetch(b.get(ref), b.bytes * R), hc = fetch(b.get(cand), b.bytes * R);
        long long nd = 0, first = -1;
        for (size_t i = 0; i < hr.size(); i += b.elem)
            if (memcmp(&hr[i], &hc[i], b.elem)) { if (first < 0) first = (long long)(i / b.elem); ++nd; }
        printf(" %s %lld", b.name, nd);
        if (nd) {
            const size_t row = (size_t)first * b.elem / b.bytes;
            printf(" (first elem %lld, row %zu)", first, row);
        }
        ok &= nd == 0;
    }
    printf(" -> %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool cnt_clean(const int* cnt, int tiles) {
    std::vector<int> h(tiles);
    CK(cudaMemcpy(h.data(), cnt, tiles * 4, cudaMemcpyDeviceToHost));
    const long long nz = std::count_if(h.begin(), h.end(), [](int v) { return v != 0; });
    printf("  cnt after the fold: %lld of %d tiles nonzero -> %s\n", nz, tiles, nz ? "FAIL" : "PASS");
    return nz == 0;
}

uint16_t bf16_rn(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}
float bf2f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

struct Oracle { int rows = 0, out_bad = 0, coef_bad = 0; double coef_max = 0; };

// host fp32 IEEE normed input (bits) and double _finish_k coefficients (1e-5 abs) of sampled rows of the fold
Oracle oracle(const Side& s, int R, const std::vector<float>& nw, const std::vector<float>& base,
              const std::vector<float>& scale) {
    std::vector<int> rows{0, 1, 17, R / 3, R / 2, 2 * R / 3, R % BM ? R - R % BM : R - BM + 5, R - 1};
    for (int& r : rows) r = std::min(std::max(r, 0), R - 1);
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    Oracle o;
    std::vector<float> part(4 * NB * PW), pre(4), post(4), comb(16);
    std::vector<uint16_t> c(D), out(D);
    for (const int r : rows) {
        CK(cudaMemcpy(part.data(), s.part + (size_t)r * 4 * NB * PW, part.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(c.data(), s.c + (size_t)r * D, D * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(out.data(), s.out + (size_t)r * D, D * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pre.data(), s.opre + (size_t)r * 4, 16, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(post.data(), s.opost + (size_t)r * 4, 16, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(comb.data(), s.ocomb + (size_t)r * 16, 64, cudaMemcpyDeviceToHost));
        volatile float ssc = 0.0f;
        for (int b = 0; b < NB; ++b) ssc = ssc + part[b * PW + 25];
        volatile float q = ssc / (float)D;
        volatile float qe = q + EPS;
        volatile float sq = std::sqrt((float)qe);
        const float rn = 1.0f / sq;
        int bad = 0;
        for (int d = 0; d < D; ++d) {
            volatile float cr = bf2f(c[d]) * rn;
            bad += bf16_rn(cr * nw[d]) != out[d];
        }
        o.out_bad += bad;
        double mix[NMIX] = {}, ss = 0;
        for (int e = 0; e < 4 * NB; ++e) {
            for (int m = 0; m < NMIX; ++m) mix[m] += part[e * PW + m];
            ss += part[e * PW + 24];
        }
        const double rinv = 1.0 / std::sqrt(ss / (4.0 * D) + EPS);
        double cp[4], cq[4], cm[4][4], err = 0;
        for (int i = 0; i < 4; ++i) {
            cp[i] = 1.0 / (1.0 + std::exp(-(mix[i] * rinv * scale[0] + base[i]))) + HC_EPS;
            cq[i] = POST_ALPHA / (1.0 + std::exp(-(mix[4 + i] * rinv * scale[1] + base[4 + i])));
        }
        for (int i = 0; i < 4; ++i) {
            double l[4], mx = -INFINITY, sum = 0;
            for (int j = 0; j < 4; ++j) {
                l[j] = mix[8 + 4 * i + j] * rinv * scale[2] + base[8 + 4 * i + j];
                mx = std::max(mx, l[j]);
            }
            for (int j = 0; j < 4; ++j) sum += (l[j] = std::exp(l[j] - mx));
            for (int j = 0; j < 4; ++j) cm[i][j] = l[j] / sum + HC_EPS;
        }
        for (int it = 0; it < ITERS; ++it) {
            if (it > 0)
                for (int i = 0; i < 4; ++i) {
                    const double sr = cm[i][0] + cm[i][1] + cm[i][2] + cm[i][3] + HC_EPS;
                    for (int j = 0; j < 4; ++j) cm[i][j] /= sr;
                }
            for (int j = 0; j < 4; ++j) {
                const double sc = cm[0][j] + cm[1][j] + cm[2][j] + cm[3][j] + HC_EPS;
                for (int i = 0; i < 4; ++i) cm[i][j] /= sc;
            }
        }
        for (int i = 0; i < 4; ++i) {
            err = std::max({err, std::fabs(cp[i] - pre[i]), std::fabs(cq[i] - post[i])});
            for (int j = 0; j < 4; ++j) err = std::max(err, std::fabs(cm[i][j] - comb[4 * i + j]));
        }
        if (!(err <= CTOL)) ++o.coef_bad;
        o.coef_max = std::max(o.coef_max, err);
        ++o.rows;
        printf("    row %d: out %d of %d differ, coefficients max |err| %.3e\n", r, bad, D, err);
    }
    return o;
}

struct Ms { double med, min; };

// median / min ms of `reps` timed runs (one warm-up first), each after `prep` and a 256 MB write (L2 flushed)
template <class P, class F>
Ms time_ms(P&& prep, F&& go, int reps, void* flush) {
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    std::vector<float> ms;
    for (int i = 0; i < reps + 1; ++i) {
        prep();
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

const void* site_fn(int mode) {
    using namespace dsv41_mhc_pf;
    if (mode == 0) return (const void*)site_kernel<true, 2, false, false>;
    if (mode == 1) return (const void*)site_kernel<false, 1, false, false>;
    return (const void*)site_kernel<false, 2, false, false>;
}
const void* fold_fn(int mode) {
    using namespace dsv41_mhc_fold;
    if (mode == 0) return (const void*)fold_kernel<true, 2, false, false>;
    if (mode == 1) return (const void*)fold_kernel<false, 1, false, false>;
    return (const void*)fold_kernel<false, 2, false, false>;
}

void print_attr(const char* name, const void* k, size_t dyn) {
    cudaFuncAttributes fa{};
    CK(cudaFuncGetAttributes(&fa, k));
    printf("  %s: %d regs/thread, %zu B local, %zu B static smem, %zu B dynamic smem\n", name, fa.numRegs,
           fa.localSizeBytes, fa.sharedSizeBytes, dyn);
}

}  // namespace

int main(int argc, char** argv) {
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    // 4096 the prefill segment, 1104 the trace's last-piece finish rows, 4100 a ragged tile, 256 mhc_pf's minimum
    std::vector<int> rows = parse_list("4096,1104,4100,256");
    int mode = 0, reps = 15;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--rows") rows = parse_list(argv[i + 1]);
        else if (k == "--mode") mode = atoi(argv[i + 1]);
        else if (k == "--reps") reps = atoi(argv[i + 1]);
        else {
            fprintf(stderr, "mhc_fold_bench [--mode 0|1|2] [--rows 4096,1104,4100,256] [--reps 15]\n");
            return 2;
        }
    }
    if (argc % 2 == 0) { fprintf(stderr, "mhc_fold_bench: %s needs a value\n", argv[argc - 1]); return 2; }
    if (rows.empty() || reps < 1 || mode < 0 || mode > 2) {
        fprintf(stderr, "need --rows, --reps >= 1 and --mode 0, 1 or 2\n");
        return 2;
    }
    for (const int r : rows) if (r < 1) { fprintf(stderr, "bad row count %d\n", r); return 2; }
    const int rmax = *std::max_element(rows.begin(), rows.end());

    int devi = 0, sms = 0, clk = 0;
    CK(cudaGetDevice(&devi));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, devi));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, devi));
    CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, devi));
    printf("device %s: %d SMs, clock %d MHz; mode %d (%s), world %d bf16 partials, fn bf16, nw fp32\n", prop.name, sms,
           clk / 1000, mode, mode == 0 ? "boundary" : mode == 1 ? "site from stream 0" : "site with a pre-mix", W);

    uint16_t* x0 = alloc<uint16_t>((size_t)rmax * 4 * D);
    uint16_t* g = alloc<uint16_t>((size_t)W * rmax * D);
    uint16_t* fn = alloc<uint16_t>((size_t)NMIX * 4 * D);
    float *post = alloc<float>((size_t)rmax * 4), *comb = alloc<float>((size_t)rmax * 16);
    float *pre = alloc<float>((size_t)rmax * 4), *base = alloc<float>(NMIX), *scale = alloc<float>(3);
    float* nw = alloc<float>(D);
    int* cnt = alloc<int>((size_t)(rmax + BM - 1) / BM);
    void* flush;
    CK(cudaMalloc(&flush, FLUSH));
    fill(x0, (size_t)rmax * 4 * D, 1, -2.f, 2.f, true);
    fill(g, (size_t)W * rmax * D, 2, -1.f, 1.f, true);
    fill(post, (size_t)rmax * 4, 3, 0.2f, 1.8f, false);
    fill(comb, (size_t)rmax * 16, 4, 0.1f, 0.4f, false);
    fill(pre, (size_t)rmax * 4, 5, 0.2f, 1.0f, false);
    fill(fn, (size_t)NMIX * 4 * D, 6, -0.02f, 0.02f, true);
    fill(base, NMIX, 7, -0.5f, 0.5f, false);
    fill(scale, 3, 8, 0.5f, 1.5f, false);
    fill(nw, D, 9, 0.5f, 1.5f, false);
    CK(cudaDeviceSynchronize());
    std::vector<float> hnw(D), hbase(NMIX), hscale(3);
    CK(cudaMemcpy(hnw.data(), nw, D * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hbase.data(), base, NMIX * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hscale.data(), scale, 12, cudaMemcpyDeviceToHost));
    Side ref = make_side(rmax), cand = make_side(rmax);
    print_attr("site_kernel", site_fn(mode), dsv41_mhc_pf::Lay<false>::BYTES);
    print_attr("ref_finish_kernel", (const void*)ref_finish_kernel, FSMEM);
    print_attr("fold_kernel", fold_fn(mode), dsv41_mhc_pf::Lay<false>::BYTES);

    bool bytes_ok = true, oracle_ok = true;
    std::string summary;
    for (const int R : rows) {
        const int tiles = (R + BM - 1) / BM;
        auto args = [&](const Side& s) {
            dsv41_mhc_fold::Args a{};
            a.s.x = s.x; a.s.xs = 4 * D; a.s.xout = s.x;
            a.s.g = g; a.s.gr = (long long)rmax * D; a.s.world = W; a.s.gbf16 = 1;
            a.s.post = post; a.s.comb = comb; a.s.pre = pre; a.s.fn = fn;
            a.s.part = s.part; a.s.c = s.c; a.s.tap = nullptr; a.s.ts = 0; a.s.R = R;
            a.base = base; a.scale = scale; a.opre = s.opre; a.opost = s.opost; a.ocomb = s.ocomb;
            a.nw = nw; a.nwbf16 = 0; a.out = s.out; a.cnt = cnt;
            a.eps = EPS; a.hc_eps = HC_EPS; a.post_alpha = POST_ALPHA; a.iters = ITERS;
            return a;
        };
        const dsv41_mhc_fold::Args ar = args(ref), ac = args(cand);
        auto site = [&] { return dsv41_mhc_pf::dispatch(ar.s, mode, false, false, 0); };
        auto site_fin = [&] {
            const cudaError_t e = site();
            if (e != cudaSuccess) return e;
            ref_finish_kernel<<<tiles, 128, FSMEM, 0>>>(ar);
            return cudaGetLastError();
        };
        auto fold = [&] { return dsv41_mhc_fold::dispatch(ac, mode, 0); };
        printf("rows %d: %d row tiles (last %d rows), grid %d x %d CTAs\n", R, tiles, R - (tiles - 1) * BM, NB, tiles);

        reset_side(ref, x0, R);
        reset_side(cand, x0, R);
        CK(cudaMemset(cnt, 0, (size_t)tiles * 4));
        CK(site_fin());
        CK(fold());
        CK(cudaDeviceSynchronize());
        bool same = compare(ref, cand, R, mode, "fold 1");
        same &= cnt_clean(cnt, tiles);
        const Oracle o = oracle(cand, R, hnw, hbase, hscale);
        reset_side(cand, x0, R);
        CK(fold());
        CK(cudaDeviceSynchronize());
        same &= compare(ref, cand, R, mode, "fold 2");
        same &= cnt_clean(cnt, tiles);
        bytes_ok &= same;
        const bool ook = o.rows > 0 && o.out_bad == 0 && o.coef_bad == 0;
        oracle_ok &= ook;
        printf("  oracle: %d rows, out (fp32 IEEE bits) %d differ, coefficients (double, %.0e abs) %d rows bad, "
               "max |err| %.3e -> %s\n", o.rows, o.out_bad, CTOL, o.coef_bad, o.coef_max, ook ? "PASS" : "FAIL");

        const size_t xb = (size_t)R * 4 * D * 2;
        auto prep_r = [&] { if (mode == 0) CK(cudaMemcpyAsync(ref.x, x0, xb, cudaMemcpyDeviceToDevice, 0)); };
        auto prep_c = [&] { if (mode == 0) CK(cudaMemcpyAsync(cand.x, x0, xb, cudaMemcpyDeviceToDevice, 0)); };
        const Ms ts = time_ms(prep_r, site, reps, flush), tb = time_ms(prep_r, site_fin, reps, flush);
        const Ms tf = time_ms(prep_c, fold, reps, flush);
        const double strm = (mode == 0 ? 2.0 : 1.0) * R * 4 * D * 2, gb = mode == 0 ? (double)W * R * D * 2 : 0.0;
        const double bytes = strm + gb + (double)R * D * 2 + (double)R * 4 * NB * PW * 4;
        const double floor_ms = bytes / (DRAM_GBS * 1e6);
        printf("  DRAM floor of the site: %.1f MB (streams %.1f, g %.1f, c %.1f, part %.1f) -> %.4f ms at %.0f GB/s\n",
               bytes / 1e6, strm / 1e6, gb / 1e6, R * D * 2 / 1e6, R * 4.0 * NB * PW * 4 / 1e6, floor_ms, DRAM_GBS);
        printf("  (a) site          median %.4f ms (min %.4f): %.2fx floor\n", ts.med, ts.min, ts.med / floor_ms);
        printf("  (b) site + finish median %.4f ms (min %.4f)\n", tb.med, tb.min);
        printf("  (c) fold          median %.4f ms (min %.4f): %.2fx floor\n", tf.med, tf.min, tf.med / floor_ms);
        printf("  saving (b - c) %+.4f ms (%+.1f%%); fold vs site alone %+.4f ms (%+.1f%%)\n", tb.med - tf.med,
               (tb.med / tf.med - 1) * 100, tf.med - ts.med, (tf.med / ts.med - 1) * 100);
        printf("  note: prod's Triton _finish_k measured %.3f ms avg at 4096 rows (19 Oct trace): served saving about "
               "site + %.3f - fold = %.4f ms\n", TRITON_FINISH_MS, TRITON_FINISH_MS,
               ts.med + TRITON_FINISH_MS - tf.med);
        char buf[200];
        snprintf(buf, sizeof buf, " rows=%d: site %.4f ms site+finish %.4f ms fold %.4f ms (saving %+.4f ms)", R,
                 ts.med, tb.med, tf.med, tb.med - tf.med);
        summary += buf;
    }
    printf("SUMMARY mode=%d bytes=%s oracle=%s%s\n", mode, bytes_ok ? "PASS" : "FAIL", oracle_ok ? "PASS" : "FAIL",
           summary.c_str());
    free_side(ref);
    free_side(cand);
    for (void* p : {(void*)x0, (void*)g, (void*)fn, (void*)post, (void*)comb, (void*)pre, (void*)base, (void*)scale,
                    (void*)nw, (void*)cnt, flush})
        CK(cudaFree(p));
    return bytes_ok && oracle_ok ? 0 : 1;
}
