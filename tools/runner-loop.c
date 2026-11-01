#include "rt.h"

_Noreturn void exit(int code) {
    for (;;) {
        rv_ecall2(93, code, 0, 0);
    }
}

static void print_unsigned(unsigned value) {
    char digits[10];
    unsigned count = 0;
    do {
        digits[count++] = (char)('0' + value % 10);
        value /= 10;
    } while (value);
    while (count) {
        rv_putchar(digits[--count]);
    }
    rv_putchar('\n');
}

int main(void) {
    volatile unsigned limit = 10;
    unsigned total = 0;
    for (unsigned i = 0; i < limit; ++i) {
        total += i * i;
    }
    print_unsigned(total);
    return 0;
}
