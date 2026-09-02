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
void l0_push_i32(L0State *state, int32_t value);
int l0_to_i32(L0State *state, size_t index, int32_t *out);

#ifdef __cplusplus
}
#endif
#endif
