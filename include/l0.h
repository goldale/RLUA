#ifndef L0_H
#define L0_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct L0State L0State;
typedef int (*L0CFunction)(L0State *state);

uint32_t l0_abi_version(void);
L0State *l0_new_state(void);
void l0_free_state(L0State *state);
/* Register an i32 -> i32 host function under an L0 identifier. */
int l0_register_i32_function(L0State *state, const char *name,
                             L0CFunction function, size_t argument_count);
/* Compile and run a UTF-8, NUL-terminated L0 source unit in this state. */
int l0_execute(L0State *state, const char *source);
void l0_push_i32(L0State *state, int32_t value);
/* During an external callback, index addresses its arguments from zero. */
int l0_to_i32(L0State *state, size_t index, int32_t *out);

#ifdef __cplusplus
}
#endif
#endif
