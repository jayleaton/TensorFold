static const char message[] = "circuit drom bytes\n";
static const unsigned char sign_probe[] = {129, 0};

__attribute__((noinline)) static int signed_bytes(const volatile signed char *p) {
    int c;
    while ((c = *p++) != 0) putchar(c);
    return *(const volatile signed char *)sign_probe;
}

__attribute__((noinline)) static unsigned unsigned_bytes(const volatile unsigned char *p) {
    unsigned c;
    while ((c = *p++) != 0) putchar(c);
    return *(const volatile unsigned char *)sign_probe;
}

int main(void) {
    int s = signed_bytes((const volatile signed char *)message);
    unsigned u = unsigned_bytes((const volatile unsigned char *)message);
    return s != -127 || u != 129;
}
