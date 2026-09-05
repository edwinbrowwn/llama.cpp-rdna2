#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    bool ok = true;

    for (int i = 0; i < GGML_OP_COUNT; ++i) {
        const char * name = ggml_op_name((ggml_op) i);
        const char * sym  = ggml_op_symbol((ggml_op) i);

        if (name == nullptr || name[0] == '\0') {
            std::fprintf(stderr, "missing ggml op name for op %d\n", i);
            ok = false;
        }

        if (sym == nullptr || sym[0] == '\0') {
            std::fprintf(stderr, "missing ggml op symbol for op %d (%s)\n", i, name ? name : "?");
            ok = false;
        }
    }

    if (!ok) {
        return 1;
    }

    std::printf("ggml op coverage check passed: all %d ops have names and symbols\n", GGML_OP_COUNT);
    return 0;
}