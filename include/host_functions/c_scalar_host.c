#include "l0.h"

#include <stdio.h>

static int c_calculate_distance(L0State *state) {
    float distance = 0.0f;
    int is_boosted = 0;
    if (!l0_to_f32(state, 0, &distance) ||
        !l0_to_bool(state, 1, &is_boosted)) {
        return 1;
    }

    if (is_boosted) distance *= 1.5f;
    l0_push_f32(state, distance);
    return 0;
}

int main(void) {
    L0State *state = l0_new_state();
    if (state == NULL) return 1;

    const L0TypeId argument_types[] = { L0_TYPE_F32, L0_TYPE_BOOL };
    const char *source =
        "let d: f32 = calc_dist(100.5, 1 == 1)\n"
        "printf(\"Distance: {}\", d)";

    int ok = l0_register_c_function(state, "calc_dist", c_calculate_distance,
                                    argument_types, 2, L0_TYPE_F32) &&
             l0_execute(state, source);
    l0_free_state(state);
    return ok ? 0 : 1;
}
