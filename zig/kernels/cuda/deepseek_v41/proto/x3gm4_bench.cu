// x3gm v4 M1b (gm4_gu_kernel, x3gm4.cuh) against production v2 gate/up gm2_kernel<2, 2, GM_GU1>: bytes, oracle, times.
#include "x3gm4.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); exit(1); } } while (0)
#define XSTR_(...) #__VA_ARGS__
#define XSTR(...) XSTR_(__VA_ARGS__)

using namespace dsv41_x3gm;

template __global__ void dsv41_x3gm::gm2_kernel<2, 2, GM_GU1>(const dsv41_x3gm::Args);
template __global__ void dsv41_x3gm::dequant_kernel<4>(const uint32_t*, half*, int);
template __global__ void dsv41_x3gm::dequant_kernel<6>(const uint32_t*, half*, int);
template __global__ void dsv41_x3gm::dequant_kernel<8>(const uint32_t*, half*, int);
template __global__ void dsv41_x3gm::dequant_kernel<10>(const uint32_t*, half*, int);

namespace {

// one rank of flash17 (the production trace): 384 experts top-6, D 5,120, I 1,152 (the rank's half of 2,304)
constexpr int E = 384, S = 6, D = 5120, I = 1152, NBLK = I / 128, BM2 = 64, BM4 = 128;
constexpr size_t FLUSH = 256u << 20;
constexpr double DRAM_GBS = 273.0, TENSOR_FLOPS = 100e12, LIMIT = 10.0, TOL = 2e-2;
using V2 = Smem2<2, 2, GM_GU1>;

__global__ void fill_words(uint32_t* p, size_t n, uint32_t seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        p[i] = x;
    }
}
__global__ void fill_half(half* p, size_t n, uint32_t seed, float lo, float hi) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
        p[i] = __float2half(lo + (hi - lo) * (x & 0xffffff) / 16777216.f);
    }
}

using Kern4 = void (*)(const Args, const Args4);
struct L4 { Kern4 k; int threads, smem; const char* name; };

template <int K2, int NS>
L4 make4(const char* name) {
    using C = Cfg4<K2, NS>;
    Kern4 k = gm4_gu_kernel<K2, NS>;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    CK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    return {k, C::THREADS, C::SMEM, name};
}
L4 pick4(int k2, const std::string& cfg) {
    if (cfg == "6a" && k2 == 6) return make4<GM4_CFG_6A>(XSTR(GM4_CFG_6A));
    if (!cfg.empty() && cfg != std::to_string(k2)) {
        fprintf(stderr, "no gm4 config %s at k2 %d (6a needs --k2 6)\n", cfg.c_str(), k2);
        exit(2);
    }
    switch (k2) {
        case 4: return make4<GM4_CFG_4>(XSTR(GM4_CFG_4));
        case 6: return make4<GM4_CFG_6>(XSTR(GM4_CFG_6));
        case 8: return make4<GM4_CFG_8>(XSTR(GM4_CFG_8));
        case 10: return make4<GM4_CFG_10>(XSTR(GM4_CFG_10));
        default: fprintf(stderr, "--k2 takes 4, 6, 8 or 10, not %d\n", k2); exit(2);
    }
}

void dequant(int k2, const uint32_t* t, half* out) {
    const dim3 g(D / 16, I / 16);
    switch (k2) {
        case 4: dequant_kernel<4><<<g, 32>>>(t, out, I); break;
        case 6: dequant_kernel<6><<<g, 32>>>(t, out, I); break;
        case 8: dequant_kernel<8><<<g, 32>>>(t, out, I); break;
        default: dequant_kernel<10><<<g, 32>>>(t, out, I); break;
    }
    CK(cudaGetLastError());
}

// routing as x3gm_bench's route(): 6 distinct experts a row, popularity rank^-zipf over shuffled experts
std::vector<int> gen_picks(int rows, double zipf, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<double> w(E);
    for (int e = 0; e < E; ++e) w[e] = zipf > 0 ? std::pow(e + 1.0, -zipf) : 1.0;
    std::shuffle(w.begin(), w.end(), rng);
    std::discrete_distribution<int> pickd(w.begin(), w.end());
    std::vector<int> pick((size_t)rows * S);
    for (int r = 0; r < rows; ++r) {
        int got = 0;
        while (got < S) {
            const int e = pickd(rng);
            bool dup = false;
            for (int j = 0; j < got; ++j) dup |= pick[(size_t)r * S + j] == e;
            if (!dup) pick[(size_t)r * S + got++] = e;
        }
    }
    return pick;
}

struct Passes { std::vector<int> pe, poff, pcnt; };
struct Route { std::vector<int> order, cnt, start; Passes v2, v4; int live = 0; };

// v2: x3gm_bench's plan (passes of min(64, rest) in expert order); v4: ceil(c / 128) balanced passes, largest first
Route plan(const std::vector<int>& pick, int P, bool lpt) {
    Route r;
    r.order.resize(P);
    std::iota(r.order.begin(), r.order.end(), 0);
    std::stable_sort(r.order.begin(), r.order.end(), [&](int a, int b) { return pick[a] < pick[b]; });
    r.cnt.assign(E, 0);
    r.start.assign(E, 0);
    for (int p = 0; p < P; ++p) ++r.cnt[pick[p]];
    for (int e = 0, off = 0; e < E; off += r.cnt[e++]) {
        r.start[e] = off;
        r.live += r.cnt[e] > 0;
        for (int j = 0; j < r.cnt[e]; j += BM2) {
            r.v2.pe.push_back(e); r.v2.poff.push_back(off + j); r.v2.pcnt.push_back(std::min(BM2, r.cnt[e] - j));
        }
    }
    Passes q;
    for (int e = 0; e < E; ++e) {
        const int c = r.cnt[e];
        if (!c) continue;
        const int n = (c + BM4 - 1) / BM4, s = (c + 8 * n - 1) / (8 * n) * 8;
        for (int i = 0; i < n; ++i) {
            const int sz = i < n - 1 ? s : c - (n - 1) * s;
            if (sz < 1 || sz > BM4) { fprintf(stderr, "bad v4 pass %d of expert %d\n", sz, e); exit(2); }
            q.pe.push_back(e); q.poff.push_back(r.start[e] + i * s); q.pcnt.push_back(sz);
        }
    }
    std::vector<int> idx(q.pe.size());
    std::iota(idx.begin(), idx.end(), 0);
    if (lpt) std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return q.pcnt[a] > q.pcnt[b]; });
    for (int i : idx) { r.v4.pe.push_back(q.pe[i]); r.v4.poff.push_back(q.poff[i]); r.v4.pcnt.push_back(q.pcnt[i]); }
    return r;
}

template <typename T> T* dev(const std::vector<T>& v) {
    T* d;
    CK(cudaMalloc(&d, v.size() * sizeof(T) + 16));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

struct DevPasses { int *pe, *poff, *pcnt, *np; };
DevPasses dev_passes(const Passes& p) {
    return {dev(p.pe), dev(p.poff), dev(p.pcnt), dev(std::vector<int>{(int)p.pe.size()})};
}
void free_passes(DevPasses& d) {
    for (int* p : {d.pe, d.poff, d.pcnt, d.np}) CK(cudaFree(p));
}

double h2d(uint16_t h) {
    const int ex = (h >> 10) & 31, m = h & 1023;
    const double v = ex == 0 ? std::ldexp((double)m, -24)
                   : ex == 31 ? (m ? NAN : INFINITY) : std::ldexp((double)(m | 1024), ex - 25);
    return h >> 15 ? -v : v;
}

// natural (Sylvester) order Walsh-Hadamard of 128 values, unnormalised: out[i] = sum_j (-1)^popc(i & j) v[j]
void wht128(double* v) {
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const double a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
}

std::vector<uint16_t> fetch(const half* p, size_t n) {
    std::vector<uint16_t> h(n);
    CK(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost));
    return h;
}

struct Host { std::vector<uint16_t> svg, svu, sud; };
struct Oracle { int pairs = 0, bad = 0; double max_abs = 0; };

// FP64 gate/up of up to 3 members of 6 experts (the most popular one first) from dequant_kernel's W, against v4's xd
Oracle oracle(const Route& r, int k2, const uint32_t* tg, const uint32_t* tu, size_t se, const half* xg,
              const half* xu, const Host& h, const std::vector<uint16_t>& xd, half* wq0, half* wq1) {
    std::vector<int> live;
    for (int e = 0; e < E; ++e) if (r.cnt[e]) live.push_back(e);
    std::vector<int> ex{(int)(std::max_element(r.cnt.begin(), r.cnt.end()) - r.cnt.begin())};
    for (int q = 0; q < 6 && ex.size() < 6; ++q) {
        const int e = live[(size_t)q * live.size() / 6];
        if (std::find(ex.begin(), ex.end(), e) == ex.end()) ex.push_back(e);
    }
    for (size_t q = 0; q < live.size() && ex.size() < 6; ++q)
        if (std::find(ex.begin(), ex.end(), live[q]) == ex.end()) ex.push_back(live[q]);
    Oracle o;
    std::vector<float> wg((size_t)D * I), wu((size_t)D * I);
    std::vector<double> g(I), u(I), v(128);
    const double had = 1.0 / std::sqrt(128.0);
    for (const int e : ex) {
        dequant(k2, tg + (size_t)e * se, wq0);
        dequant(k2, tu + (size_t)e * se, wq1);
        CK(cudaDeviceSynchronize());
        const std::vector<uint16_t> hg = fetch(wq0, (size_t)D * I), hu = fetch(wq1, (size_t)D * I);
        for (size_t i = 0; i < hg.size(); ++i) { wg[i] = (float)h2d(hg[i]); wu[i] = (float)h2d(hu[i]); }
        const int c = r.cnt[e];
        std::vector<int> mem{0, c / 2, c - 1};
        mem.erase(std::unique(mem.begin(), mem.end()), mem.end());
        for (const int m : mem) {
            const int p = r.order[r.start[e] + m];
            const std::vector<uint16_t> rg = fetch(xg + (size_t)p * D, D), ru = fetch(xu + (size_t)p * D, D);
            std::fill(g.begin(), g.end(), 0.0);
            std::fill(u.begin(), u.end(), 0.0);
            for (int k = 0; k < D; ++k) {
                const double a = h2d(rg[k]), b = h2d(ru[k]);
                const float* pg = &wg[(size_t)k * I];
                const float* pu = &wu[(size_t)k * I];
                for (int n = 0; n < I; ++n) { g[n] += a * pg[n]; u[n] += b * pu[n]; }
            }
            for (int blk = 0; blk < NBLK; ++blk) {
                double* gb = &g[blk * 128];
                double* ub = &u[blk * 128];
                wht128(gb);
                wht128(ub);
                for (int i = 0; i < 128; ++i) {
                    const size_t col = (size_t)e * I + blk * 128 + i;
                    const double gg = std::min(gb[i] * had * h2d(h.svg[col]), LIMIT);
                    const double uu = std::min(std::max(ub[i] * had * h2d(h.svu[col]), -LIMIT), LIMIT);
                    v[i] = gg / (1.0 + std::exp(-gg)) * uu * h2d(h.sud[col]);
                }
                wht128(v.data());
                for (int i = 0; i < 128; ++i) {
                    const double ref = v[i] * had, got = h2d(xd[(size_t)p * I + blk * 128 + i]);
                    const double err = std::fabs(ref - got);
                    if (!(err <= TOL + TOL * std::fabs(ref))) ++o.bad;
                    if (!(err <= o.max_abs)) o.max_abs = err;
                }
            }
            ++o.pairs;
        }
    }
    return o;
}

struct Ms { double med, min; };

// median / min ms of `reps` launches (one warm-up first), each after a 256 MB write (L2 flushed) and a ticket reset
template <class F>
Ms time_ms(F&& go, int reps, void* flush, int* ticket) {
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    std::vector<float> ms;
    for (int i = 0; i < reps + 1; ++i) {
        CK(cudaMemsetAsync(flush, i & 0xff, FLUSH, 0));
        CK(cudaMemsetAsync(ticket, 0, 4, 0));
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

void print_attr(const char* tag, const char* name, const void* k, size_t dyn) {
    cudaFuncAttributes fa{};
    CK(cudaFuncGetAttributes(&fa, k));
    printf("  %-3s %s: %d regs/thread, %zu B local, %zu B static smem, %zu B dynamic smem\n", tag, name, fa.numRegs,
           fa.localSizeBytes, fa.sharedSizeBytes, dyn);
}

}  // namespace

int main(int argc, char** argv) {
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    // 4096 = the measured prefill segment, 2048 = 2K segments; 5 = the decode verify window (prod decode runs x3ld)
    std::vector<int> rows = parse_list("4096,2048,5");
    int k2 = 6, reps = 15, lpt = 1;
    unsigned seed = 1234;
    double zipf = 0;
    std::string cfg, picks_path;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--rows") rows = parse_list(argv[i + 1]);
        else if (k == "--k2") k2 = atoi(argv[i + 1]);
        else if (k == "--cfg") cfg = argv[i + 1];
        else if (k == "--reps") reps = atoi(argv[i + 1]);
        else if (k == "--zipf") zipf = atof(argv[i + 1]);
        else if (k == "--seed") seed = (unsigned)atoi(argv[i + 1]);
        else if (k == "--lpt") lpt = atoi(argv[i + 1]);
        else if (k == "--picks") picks_path = argv[i + 1];
        else {
            fprintf(stderr, "x3gm4_bench [--rows 4096,2048,5] [--k2 6] [--cfg 6|6a] [--reps 15] [--zipf 0] "
                            "[--seed 1234] [--lpt 1] [--picks file.i32]\n");
            return 2;
        }
    }
    if (rows.empty() || reps < 1) { fprintf(stderr, "need --rows and --reps >= 1\n"); return 2; }
    for (const int r : rows) if (r < 1) { fprintf(stderr, "bad row count %d\n", r); return 2; }
    const int rmax = *std::max_element(rows.begin(), rows.end()), pmax = rmax * S;
    std::vector<int> file_picks;
    if (!picks_path.empty()) {
        FILE* f = fopen(picks_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", picks_path.c_str()); return 2; }
        file_picks.resize((size_t)pmax);
        const size_t got = fread(file_picks.data(), 4, file_picks.size(), f);
        fclose(f);
        if (got < file_picks.size()) {
            fprintf(stderr, "%s: %zu expert ids, %d rows need %d\n", picks_path.c_str(), got, rmax, pmax);
            return 2;
        }
        for (const int e : file_picks)
            if (e < 0 || e >= E) { fprintf(stderr, "expert id %d out of range\n", e); return 2; }
    }

    int devi = 0, sms = 0, clk = 0;
    CK(cudaGetDevice(&devi));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, devi));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, devi));
    CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, devi));
    printf("device %s: %d SMs, clock %d MHz\n", prop.name, sms, clk / 1000);
    const L4 g4 = pick4(k2, cfg);
    const auto v2k = gm2_kernel<2, 2, GM_GU1>;
    CK(cudaFuncSetAttribute(v2k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)V2::SMEM));
    CK(cudaFuncSetAttribute(v2k, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    int per2 = 0, per4 = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per2, v2k, V2::THREADS, V2::SMEM));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per4, g4.k, g4.threads, g4.smem));
    if (per2 < 1 || per4 < 1) { fprintf(stderr, "no residency: v2 %d gm4 %d CTAs/SM\n", per2, per4); return 1; }
    printf("k2 %d zipf %.2f seed %u lpt %d picks %s; v2 %d CTAs/SM, gm4 <%s> %d CTA/SM\n", k2, zipf, seed, lpt,
           picks_path.empty() ? "generated" : picks_path.c_str(), per2, g4.name, per4);

    const size_t se = (size_t)(D / 16) * (I / 16) * 4 * k2, mw = (size_t)E * se;
    uint32_t *tg, *tu;
    half *xg, *xu, *svg, *svu, *sud, *xd2, *xd4, *wq0, *wq1;
    int *ticket, *k2e;
    void* flush;
    CK(cudaMalloc(&tg, mw * 4)); CK(cudaMalloc(&tu, mw * 4));
    CK(cudaMalloc(&xg, (size_t)pmax * D * 2)); CK(cudaMalloc(&xu, (size_t)pmax * D * 2));
    CK(cudaMalloc(&svg, (size_t)E * I * 2)); CK(cudaMalloc(&svu, (size_t)E * I * 2));
    CK(cudaMalloc(&sud, (size_t)E * I * 2));
    CK(cudaMalloc(&xd2, (size_t)pmax * I * 2)); CK(cudaMalloc(&xd4, (size_t)pmax * I * 2));
    CK(cudaMalloc(&wq0, (size_t)D * I * 2)); CK(cudaMalloc(&wq1, (size_t)D * I * 2));
    CK(cudaMalloc(&ticket, 8));
    CK(cudaMalloc(&flush, FLUSH));
    k2e = dev(std::vector<int>(E, k2));
    fill_words<<<1024, 256>>>(tg, mw, 1);
    fill_words<<<1024, 256>>>(tu, mw, 2);
    fill_half<<<1024, 256>>>(xg, (size_t)pmax * D, 4, -1.f / 64, 1.f / 64);
    fill_half<<<1024, 256>>>(xu, (size_t)pmax * D, 5, -1.f / 64, 1.f / 64);
    fill_half<<<1024, 256>>>(svg, (size_t)E * I, 6, 0.5f, 1.5f);
    fill_half<<<1024, 256>>>(svu, (size_t)E * I, 7, 0.5f, 1.5f);
    fill_half<<<1024, 256>>>(sud, (size_t)E * I, 8, 0.5f, 1.5f);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    const Host host{fetch(svg, (size_t)E * I), fetch(svu, (size_t)E * I), fetch(sud, (size_t)E * I)};

    bool bytes_ok = true, oracle_ok = true;
    std::string summary;
    for (const int R : rows) {
        const int P = R * S;
        const std::vector<int> pick = picks_path.empty() ? gen_picks(R, zipf, seed)
                                                         : std::vector<int>(file_picks.begin(), file_picks.begin() + P);
        const Route rt = plan(pick, P, lpt != 0);
        int* order = dev(rt.order);
        DevPasses p2 = dev_passes(rt.v2), p4 = dev_passes(rt.v4);
        Args a{};
        a.x0 = xg; a.x1 = xu; a.t0 = tg; a.t1 = tu; a.k2e = k2e; a.order = order;
        a.pe = p2.pe; a.poff = p2.poff; a.pcnt = p2.pcnt; a.npass = p2.np; a.ticket = ticket;
        a.sv0 = svg; a.sv1 = svu; a.sd = sud; a.xd = xd2; a.K = D; a.N = I; a.limit = (float)LIMIT;
        Args a4 = a;
        a4.xd = xd4;
        const Args4 b{p4.pe, p4.poff, p4.pcnt, p4.np, ticket + 1};
        const int n2 = (int)rt.v2.pe.size(), n4 = (int)rt.v4.pe.size();
        const int grid2 = sms * per2, grid4 = std::min(sms, n4 * NBLK);
        auto run2 = [&] { v2k<<<grid2, V2::THREADS, V2::SMEM>>>(a); return cudaGetLastError(); };
        auto run4 = [&] { g4.k<<<grid4, g4.threads, g4.smem>>>(a4, b); return cudaGetLastError(); };
        printf("rows %d: P %d, %d experts live; passes v2 %d / v4 %d (items %d / %d, v4 passes %d..%d members); "
               "grid v2 %d gm4 %d\n", R, P, rt.live, n2, n4, n2 * NBLK, n4 * NBLK,
               *std::min_element(rt.v4.pcnt.begin(), rt.v4.pcnt.end()),
               *std::max_element(rt.v4.pcnt.begin(), rt.v4.pcnt.end()), grid2, grid4);
        print_attr("v2", "gm2_kernel<2, 2, GM_GU1>", (const void*)v2k, V2::SMEM);
        print_attr("gm4", (std::string("gm4_gu_kernel<") + g4.name + ">").c_str(), (const void*)g4.k, g4.smem);

        CK(cudaMemset(xd2, 0xff, (size_t)P * I * 2));
        CK(cudaMemset(xd4, 0xff, (size_t)P * I * 2));
        CK(cudaMemset(ticket, 0, 8));
        CK(run2());
        CK(run4());
        CK(cudaDeviceSynchronize());
        const std::vector<uint16_t> h2 = fetch(xd2, (size_t)P * I), h4 = fetch(xd4, (size_t)P * I);
        long long ndiff = 0, sent2 = 0, sent4 = 0;
        for (size_t i = 0; i < h2.size(); ++i) {
            sent2 += h2[i] == 0xffff;
            sent4 += h4[i] == 0xffff;
            if (h2[i] == h4[i]) continue;
            if (ndiff++ < 5)
                printf("  diff pair %zu col %zu: v2 0x%04x gm4 0x%04x\n", i / I, i % I, h2[i], h4[i]);
        }
        const bool same = ndiff == 0 && sent2 == 0 && sent4 == 0;
        bytes_ok &= same;
        printf("  bytes: %lld of %zu halves differ, sentinels v2 %lld gm4 %lld -> %s\n", ndiff, h2.size(), sent2, sent4,
               same ? "PASS" : "FAIL");
        const Oracle o = oracle(rt, k2, tg, tu, se, xg, xu, host, h4, wq0, wq1);
        const bool ook = o.bad == 0 && o.pairs > 0;
        oracle_ok &= ook;
        printf("  oracle (FP64, |err| <= %.0e + %.0e |ref|): %d pairs x %d columns, %d bad, max abs err %.3e -> %s\n",
               TOL, TOL, o.pairs, I, o.bad, o.max_abs, ook ? "PASS" : "FAIL");

        const Ms t2 = time_ms(run2, reps, flush, ticket), t4 = time_ms(run4, reps, flush, ticket + 1);
        const double bytes = (double)rt.live * 2 * D * I * k2 / 16 + 2.0 * P * D * 2 + (double)P * I * 2;
        const double dram_ms = bytes / (DRAM_GBS * 1e6), tensor_ms = 2.0 * 2 * P * D * I / TENSOR_FLOPS * 1e3;
        const double floor_ms = std::max(dram_ms, tensor_ms), sp = (t2.med / t4.med - 1) * 100;
        printf("  floors: %.1f MB -> DRAM %.4f ms at %.0f GB/s, tensor %.4f ms at %.0f TFLOP/s\n", bytes / 1e6, dram_ms,
               DRAM_GBS, tensor_ms, TENSOR_FLOPS / 1e12);
        printf("  v2  median %.4f ms (min %.4f): %.1f GB/s, %.2fx floor\n", t2.med, t2.min, bytes / (t2.med * 1e6),
               t2.med / floor_ms);
        printf("  gm4 median %.4f ms (min %.4f): %.1f GB/s, %.2fx floor; speedup %+.1f%%\n", t4.med, t4.min,
               bytes / (t4.med * 1e6), t4.med / floor_ms, sp);
        char buf[160];
        snprintf(buf, sizeof buf, " rows=%d: v2 %.4f ms gm4 %.4f ms (%+.1f%%)", R, t2.med, t4.med, sp);
        summary += buf;
        CK(cudaFree(order));
        free_passes(p2);
        free_passes(p4);
    }
    printf("SUMMARY k2=%d zipf=%.2f bytes=%s oracle=%s%s cfg=%s lpt=%d\n", k2, zipf, bytes_ok ? "PASS" : "FAIL",
           oracle_ok ? "PASS" : "FAIL", summary.c_str(), g4.name, lpt);
    for (void* p : {(void*)tg, (void*)tu, (void*)xg, (void*)xu, (void*)svg, (void*)svu, (void*)sud, (void*)xd2,
                    (void*)xd4, (void*)wq0, (void*)wq1, (void*)ticket, (void*)k2e, flush})
        CK(cudaFree(p));
    return bytes_ok && oracle_ok ? 0 : 1;
}
