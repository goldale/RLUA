#include "l0.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* The callback sees only the arguments of this L0 call, indexed from zero. */
static int c_add(L0State *state) {
    int32_t left;
    int32_t right;
    if (!l0_to_i32(state, 0, &left) || !l0_to_i32(state, 1, &right)) {
        return 1;
    }

    int32_t result = left + right;
    printf("C callback: %d + %d = %d\n", left, right, result);
    l0_push_i32(state, result);
    return 0;
}

static char *read_source(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) return NULL;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return NULL; }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) { fclose(file); return NULL; }
    char *source = malloc((size_t)length + 1);
    if (source == NULL || fread(source, 1, (size_t)length, file) != (size_t)length) {
        free(source);
        fclose(file);
        return NULL;
    }
    source[length] = '\0';
    fclose(file);
    return source;
}

int main(void) {
    L0State *state = l0_new_state();
    char *source = read_source("examples/host_functions/c_host.l0");
    if (state == NULL || source == NULL) {
        fprintf(stderr, "could not initialize the C host example\n");
        free(source);
        l0_free_state(state);
        return 1;
    }

    if (!l0_register_i32_function(state, "c_add", c_add, 2) || !l0_execute(state, source)) {
        fprintf(stderr, "L0 execution failed\n");
        free(source);
        l0_free_state(state);
        return 1;
    }

    free(source);
    l0_free_state(state);
    return 0;
}
