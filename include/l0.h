#ifndef L0_H
#define L0_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct L0State L0State;
typedef int (*L0CFunction)(L0State *state);

/* Stable IDs for scalar types accepted by l0_register_c_function. */
typedef enum {
    L0_TYPE_I8 = 0,
    L0_TYPE_I16,
    L0_TYPE_I32,
    L0_TYPE_I64,
    L0_TYPE_U8,
    L0_TYPE_U16,
    L0_TYPE_U32,
    L0_TYPE_U64,
    /* IEEE-754 binary16 bits, passed through uint16_t helpers. */
    L0_TYPE_F16,
    L0_TYPE_F32,
    L0_TYPE_F64,
    L0_TYPE_BOOL
} L0TypeId;

uint32_t l0_abi_version(void);
L0State *l0_new_state(void);
void l0_free_state(L0State *state);
/*
 * Register a scalar host function under an L0 identifier. arg_types must point
 * to argument_count IDs when argument_count is nonzero. Returns 1 on success.
 */
int l0_register_c_function(L0State *state, const char *name,
                           L0CFunction function,
                           const L0TypeId *arg_types, size_t argument_count,
                           L0TypeId result_type);
/* Backward-compatible shorthand for i32... -> i32 callbacks. */
int l0_register_i32_function(L0State *state, const char *name,
                             L0CFunction function, size_t argument_count);
/* Compile and run a UTF-8, NUL-terminated L0 source unit in this state. */
int l0_execute(L0State *state, const char *source);
/* During an external callback, index addresses its arguments from zero. */
void l0_push_i8(L0State *state, int8_t value);
void l0_push_i16(L0State *state, int16_t value);
void l0_push_i32(L0State *state, int32_t value);
void l0_push_i64(L0State *state, int64_t value);
void l0_push_u8(L0State *state, uint8_t value);
void l0_push_u16(L0State *state, uint16_t value);
void l0_push_u32(L0State *state, uint32_t value);
void l0_push_u64(L0State *state, uint64_t value);
void l0_push_f16(L0State *state, uint16_t ieee754_bits);
void l0_push_f32(L0State *state, float value);
void l0_push_f64(L0State *state, double value);
void l0_push_bool(L0State *state, int value);
int l0_to_i8(L0State *state, size_t index, int8_t *out);
int l0_to_i16(L0State *state, size_t index, int16_t *out);
int l0_to_i32(L0State *state, size_t index, int32_t *out);
int l0_to_i64(L0State *state, size_t index, int64_t *out);
int l0_to_u8(L0State *state, size_t index, uint8_t *out);
int l0_to_u16(L0State *state, size_t index, uint16_t *out);
int l0_to_u32(L0State *state, size_t index, uint32_t *out);
int l0_to_u64(L0State *state, size_t index, uint64_t *out);
int l0_to_f16(L0State *state, size_t index, uint16_t *out_ieee754_bits);
int l0_to_f32(L0State *state, size_t index, float *out);
int l0_to_f64(L0State *state, size_t index, double *out);
int l0_to_bool(L0State *state, size_t index, int *out);

#ifdef __cplusplus
}
#endif
#endif
