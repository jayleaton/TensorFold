// RV32 integer library; program_compile supplies memory operations, putchar and startup.
typedef unsigned int tf_wu;
typedef unsigned long long tf_wd;
struct tf_wint { tf_wu v[8]; int neg; };

// ABI: nodes are topological, child indices precede parents, n <= 64, and node n-1 is the output root.
struct tf_node { int op, l, r, k; const char *text; };

// Ops 0..10 are stable; append 11 gcd, 12 isqrt, 13 factorial, 14 range-product, 15 digit-sum.
static struct tf_wint tf_wvalues[64];

// The parent proves every admitted input and node result has magnitude below 10^62; scratch products use 512 bits.
static int tf_wbad;

static struct tf_wint tf_wsmall(tf_wu x) {
    struct tf_wint a;
    for (int i = 0; i < 8; ++i) a.v[i] = 0;
    a.v[0] = x;
    a.neg = 0;
    return a;
}

static int tf_wused(const struct tf_wint *a) {
    int n = 8;
    while (n && !a->v[n - 1]) --n;
    return n;
}

static void tf_wnormalize(struct tf_wint *a) {
    if (!tf_wused(a)) a->neg = 0;
}

static int tf_wcmp(const struct tf_wint *a, const struct tf_wint *b) {
    for (int i = 7; i >= 0; --i) {
        if (a->v[i] != b->v[i]) return a->v[i] > b->v[i] ? 1 : -1;
    }
    return 0;
}

static void tf_wbound(const struct tf_wint *a) {
    const struct tf_wint limit = { { 0u, 1073741824u, 4155282634u, 576837993u, 895677624u, 3947553080u, 15930u, 0u }, 0 };
    if (tf_wcmp(a, &limit) >= 0) tf_wbad = 1;
}

static struct tf_wint tf_waddmag(const struct tf_wint *a, const struct tf_wint *b) {
    struct tf_wint c = tf_wsmall(0);
    tf_wd carry = 0;
    for (int i = 0; i < 8; ++i) {
        tf_wd s = (tf_wd)a->v[i] + b->v[i] + carry;
        c.v[i] = (tf_wu)s;
        carry = s >> 32;
    }
    if (carry) tf_wbad = 1;
    return c;
}

// Magnitude subtraction also supports the wrapped low words of a 257-bit division remainder.
static struct tf_wint tf_wsubmag(const struct tf_wint *a, const struct tf_wint *b) {
    struct tf_wint c = tf_wsmall(0);
    tf_wd borrow = 0;
    for (int i = 0; i < 8; ++i) {
        tf_wd t = (tf_wd)b->v[i] + borrow;
        c.v[i] = a->v[i] - (tf_wu)t;
        borrow = (tf_wd)a->v[i] < t;
    }
    return c;
}

static struct tf_wint tf_wadd(struct tf_wint a, struct tf_wint b) {
    struct tf_wint c;
    if (a.neg == b.neg) {
        c = tf_waddmag(&a, &b);
        c.neg = a.neg;
    } else if (tf_wcmp(&a, &b) >= 0) {
        c = tf_wsubmag(&a, &b);
        c.neg = a.neg;
    } else {
        c = tf_wsubmag(&b, &a);
        c.neg = b.neg;
    }
    tf_wnormalize(&c);
    return c;
}

static void tf_wproduct(tf_wu w[16], const struct tf_wint *a, const struct tf_wint *b) {
    for (int i = 0; i < 16; ++i) w[i] = 0;
    int na = tf_wused(a), nb = tf_wused(b);
    for (int i = 0; i < na; ++i) {
        tf_wd carry = 0;
        for (int j = 0; j < nb; ++j) {
            tf_wd t = (tf_wd)a->v[i] * b->v[j] + w[i + j] + carry;
            w[i + j] = (tf_wu)t;
            carry = t >> 32;
        }
        w[i + nb] = (tf_wu)carry;
    }
}

static struct tf_wint tf_wmul(const struct tf_wint *a, const struct tf_wint *b) {
    tf_wu w[16];
    tf_wproduct(w, a, b);
    struct tf_wint c = tf_wsmall(0);
    for (int i = 0; i < 8; ++i) {
        c.v[i] = w[i];
        if (w[i + 8]) tf_wbad = 1;
    }
    c.neg = a->neg != b->neg;
    tf_wnormalize(&c);
    tf_wbound(&c);
    return c;
}

// All divisions use 32-bit instructions; the small-divisor path handles two halfwords per limb.
static tf_wu tf_wdivsmall(tf_wu *w, int n, tf_wu d) {
    tf_wu rem = 0;
    for (int i = n - 1; i >= 0; --i) {
        tf_wu x = w[i];
        if (d <= 65535u) {
            tf_wu t = (rem << 16) | (x >> 16);
            tf_wu q = t / d;
            t = ((t % d) << 16) | (x & 65535u);
            w[i] = (q << 16) | (t / d);
            rem = t % d;
        } else if (!rem) {
            w[i] = x / d;
            rem = x % d;
        } else {
            tf_wu q = 0;
            for (int bit = 31; bit >= 0; --bit) {
                tf_wu high = rem >> 31;
                rem = (rem << 1) | ((x >> bit) & 1u);
                if (high || rem >= d) {
                    rem -= d;
                    q |= 1u << bit;
                }
            }
            w[i] = q;
        }
    }
    return rem;
}

static int tf_wbits(const tf_wu *w, int n) {
    while (n && !w[n - 1]) --n;
    if (!n) return 0;
    tf_wu x = w[n - 1];
    int bits = (n - 1) * 32;
    while (x) { ++bits; x >>= 1; }
    return bits;
}

static int tf_wstep(struct tf_wint *r, tf_wu bit, const struct tf_wint *d) {
    tf_wu carry = bit;
    for (int i = 0; i < 8; ++i) {
        tf_wu high = r->v[i] >> 31;
        r->v[i] = (r->v[i] << 1) | carry;
        carry = high;
    }
    if (!carry && tf_wcmp(r, d) < 0) return 0;
    *r = tf_wsubmag(r, d);
    return 1;
}

static void tf_wdivmag(const struct tf_wint *a, const struct tf_wint *b,
                       struct tf_wint *q, struct tf_wint *r) {
    *q = tf_wsmall(0);
    *r = tf_wsmall(0);
    int nb = tf_wused(b);
    if (!nb) { tf_wbad = 1; return; }
    if (nb == 1) {
        *q = *a;
        q->neg = 0;
        r->v[0] = tf_wdivsmall(q->v, 8, b->v[0]);
        return;
    }
    for (int bit = tf_wbits(a->v, 8) - 1; bit >= 0; --bit) {
        if (tf_wstep(r, (a->v[bit / 32] >> (bit % 32)) & 1u, b))
            q->v[bit / 32] |= 1u << (bit % 32);
    }
}

static void tf_wdivmod(const struct tf_wint *a, const struct tf_wint *b,
                       struct tf_wint *q, struct tf_wint *r) {
    tf_wdivmag(a, b, q, r);
    if (tf_wbad) return;
    if (a->neg != b->neg && tf_wused(r)) {
        struct tf_wint one = tf_wsmall(1);
        *q = tf_waddmag(q, &one);
        *r = tf_wsubmag(b, r);
    }
    q->neg = a->neg != b->neg;
    r->neg = b->neg;
    tf_wnormalize(q);
    tf_wnormalize(r);
}

static struct tf_wint tf_wmodmul(const struct tf_wint *a, const struct tf_wint *b,
                                const struct tf_wint *m) {
    tf_wu w[16];
    tf_wproduct(w, a, b);
    struct tf_wint r = tf_wsmall(0);
    if (tf_wused(m) == 1) {
        r.v[0] = tf_wdivsmall(w, 16, m->v[0]);
        return r;
    }
    for (int bit = tf_wbits(w, 16) - 1; bit >= 0; --bit)
        tf_wstep(&r, (w[bit / 32] >> (bit % 32)) & 1u, m);
    return r;
}

static struct tf_wint tf_wpow(struct tf_wint a, const struct tf_wint *e,
                             const struct tf_wint *mod) {
    struct tf_wint result = tf_wsmall(1), m = tf_wsmall(0), unused;
    if (e->neg) { tf_wbad = 1; return m; }
    if (mod) {
        m = *mod;
        m.neg = 0;
        if (!tf_wused(&m)) { tf_wbad = 1; return m; }
        if (tf_wused(&m) == 1 && m.v[0] == 1) return tf_wsmall(0);
        int negative = a.neg;
        struct tf_wint reduced;
        tf_wdivmag(&a, &m, &unused, &reduced);
        a = reduced;
        if (negative && tf_wused(&a)) a = tf_wsubmag(&m, &a);
    }
    int bits = tf_wbits(e->v, 8);
    for (int bit = 0; bit < bits && !tf_wbad; ++bit) {
        if ((e->v[bit / 32] >> (bit % 32)) & 1u)
            result = mod ? tf_wmodmul(&result, &a, &m) : tf_wmul(&result, &a);
        if (bit + 1 < bits) a = mod ? tf_wmodmul(&a, &a, &m) : tf_wmul(&a, &a);
    }
    if (mod && mod->neg && tf_wused(&result)) {
        result = tf_wsubmag(&m, &result);
        result.neg = 1;
    }
    return result;
}

static struct tf_wint tf_wgcd(struct tf_wint a, struct tf_wint b) {
    a.neg = b.neg = 0;
    while (tf_wused(&b) && !tf_wbad) {
        struct tf_wint q, r;
        tf_wdivmag(&a, &b, &q, &r);
        a = b;
        b = r;
    }
    return a;
}

static struct tf_wint tf_wisqrt(const struct tf_wint *a) {
    struct tf_wint x = tf_wsmall(0), q, r;
    if (a->neg) { tf_wbad = 1; return x; }
    int bits = tf_wbits(a->v, 8);
    if (!bits) return x;
    int shift = (bits + 1) / 2;
    x.v[shift / 32] = 1u << (shift % 32);
    tf_wdivmag(a, &x, &q, &r);
    struct tf_wint y = tf_waddmag(&x, &q);
    tf_wdivsmall(y.v, 8, 2);
    while (tf_wcmp(&y, &x) < 0 && !tf_wbad) {
        x = y;
        tf_wdivmag(a, &x, &q, &r);
        y = tf_waddmag(&x, &q);
        tf_wdivsmall(y.v, 8, 2);
    }
    return x;
}

static struct tf_wint tf_wfactorial(const struct tf_wint *a) {
    struct tf_wint result = tf_wsmall(1), maximum = tf_wsmall(48);
    if (a->neg || tf_wcmp(a, &maximum) > 0) { tf_wbad = 1; return result; }
    for (tf_wu i = 2; i <= a->v[0] && !tf_wbad; ++i) {
        struct tf_wint factor = tf_wsmall(i);
        result = tf_wmul(&result, &factor);
    }
    return result;
}

static int tf_wsignedcmp(const struct tf_wint *a, const struct tf_wint *b) {
    if (a->neg != b->neg) return a->neg ? -1 : 1;
    return a->neg ? -tf_wcmp(a, b) : tf_wcmp(a, b);
}

static struct tf_wint tf_wrange(struct tf_wint a, const struct tf_wint *stop,
                               const struct tf_wint *step) {
    struct tf_wint result = tf_wsmall(1);
    if (!tf_wused(step)) { tf_wbad = 1; return result; }
    if (step->neg ? tf_wsignedcmp(&a, stop) <= 0 : tf_wsignedcmp(&a, stop) >= 0) return result;
    struct tf_wint negative = step->neg ? *stop : a;
    negative.neg = !negative.neg;
    struct tf_wint distance = tf_wadd(step->neg ? a : *stop, negative);
    struct tf_wint stride = *step, length, remainder;
    stride.neg = 0;
    tf_wdivmod(&distance, &stride, &length, &remainder);
    if (tf_wused(&remainder)) length = tf_wadd(length, tf_wsmall(1));
    struct tf_wint maximum = tf_wsmall(10000);
    if (tf_wbad || tf_wcmp(&length, &maximum) > 0) { tf_wbad = 1; return result; }
    int count = 0;
    while (!tf_wbad && (step->neg ? tf_wsignedcmp(&a, stop) > 0 : tf_wsignedcmp(&a, stop) < 0)) {
        if (++count > 10000) { tf_wbad = 1; break; }
        tf_wbound(&a);
        result = tf_wmul(&result, &a);
        if (!tf_wused(&result)) return result;
        a = tf_wadd(a, *step);
    }
    return result;
}

static struct tf_wint tf_wdigitsum(struct tf_wint a) {
    tf_wu sum = 0;
    if (a.neg) tf_wbad = 1;
    while (tf_wused(&a)) sum += tf_wdivsmall(a.v, 8, 10);
    return tf_wsmall(sum);
}

// Literal text contains signed digits without a radix prefix; k is radix 2, 10, or 16, not a child index.
static struct tf_wint tf_wparse(const char *s, int base) {
    struct tf_wint a = tf_wsmall(0);
    if (!s || (base != 2 && base != 10 && base != 16)) { tf_wbad = 1; return a; }
    int negative = *s == '-';
    if (*s == '-' || *s == '+') ++s;
    if (!*s) tf_wbad = 1;
    for (; *s; ++s) {
        int digit;
        if (*s >= '0' && *s <= '9') digit = *s - '0';
        else if (*s >= 'a' && *s <= 'f') digit = *s - 'a' + 10;
        else if (*s >= 'A' && *s <= 'F') digit = *s - 'A' + 10;
        else { tf_wbad = 1; break; }
        if (digit >= base) { tf_wbad = 1; break; }
        tf_wd carry = (tf_wu)digit;
        for (int i = 0; i < 8; ++i) {
            tf_wd t = (tf_wd)a.v[i] * (tf_wu)base + carry;
            a.v[i] = (tf_wu)t;
            carry = t >> 32;
        }
        if (carry) { tf_wbad = 1; break; }
    }
    a.neg = negative;
    tf_wnormalize(&a);
    return a;
}

static int tf_wformat(struct tf_wint a, char text[79]) {
    char digits[78];
    int count = 0, len = 0;
    do { digits[count++] = (char)('0' + tf_wdivsmall(a.v, 8, 10)); } while (tf_wused(&a));
    if (a.neg) text[len++] = '-';
    while (count) text[len++] = digits[--count];
    return len;
}

static void tf_wprint(const struct tf_wint *a) {
    char text[79];
    int len = tf_wformat(*a, text);
    for (int i = 0; i < len; ++i) putchar(text[i]);
}

// Append the static node array and main calling tf_run(nodes, N); errors return 1 before emitting any output.
int tf_run(const volatile struct tf_node *nodes, int n) {
    tf_wbad = 0;
    if (!nodes || n < 1 || n > 64) return 1;
    struct tf_wint remainder = tf_wsmall(0);
    for (int i = 0; i < n; ++i) {
        const volatile struct tf_node *node = &nodes[i];
        struct tf_wint a = tf_wsmall(0), b = tf_wsmall(0), c;
        int op = node->op;
        if (op < 0 || op > 15) return 1;
        if (op) {
            if (node->l < 0 || node->l >= i) return 1;
            a = tf_wvalues[node->l];
        }
        if ((op >= 2 && op <= 8) || op == 10 || op == 11 || op == 14) {
            if (node->r < 0 || node->r >= i) return 1;
            b = tf_wvalues[node->r];
        }
        switch (op) {
        case 0: c = tf_wparse(node->text, node->k); break;
        case 1:
            c = a;
            c.neg = !c.neg;
            tf_wnormalize(&c);
            break;
        case 2: c = tf_wadd(a, b); break;
        case 3:
            b.neg = !b.neg;
            c = tf_wadd(a, b);
            break;
        case 4: c = tf_wmul(&a, &b); break;
        case 5: case 6: case 10:
            if (op == 10 && i != n - 1) return 1;
            tf_wdivmod(&a, &b, &c, &remainder);
            if (op == 6) c = remainder;
            break;
        case 7: c = tf_wpow(a, &b, (const struct tf_wint *)0); break;
        case 8:
            if (node->k < 0 || node->k >= i) return 1;
            c = tf_wpow(a, &b, &tf_wvalues[node->k]);
            break;
        case 9: {
            char text[79];
            c = tf_wsmall((tf_wu)tf_wformat(a, text));
            break;
        }
        case 11: c = tf_wgcd(a, b); break;
        case 12: c = tf_wisqrt(&a); break;
        case 13: c = tf_wfactorial(&a); break;
        case 14:
            if (node->k < 0 || node->k >= i) return 1;
            c = tf_wrange(a, &b, &tf_wvalues[node->k]);
            break;
        case 15: c = tf_wdigitsum(a); break;
        default: return 1;
        }
        tf_wbound(&c);
        if (tf_wbad) return 1;
        tf_wvalues[i] = c;
    }
    if (nodes[n - 1].op == 10) putchar('(');
    tf_wprint(&tf_wvalues[n - 1]);
    if (nodes[n - 1].op == 10) {
        putchar(',');
        putchar(' ');
        tf_wprint(&remainder);
        putchar(')');
    }
    putchar('\n');
    return 0;
}
