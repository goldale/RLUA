# RLUA language standard — draft 0.1

This document specifies the implemented core and the approved implementation order for the next runtime layers. Every value has an explicit static type; tables are data containers, not objects.

## Values and structures

Scalar types are `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`, `f64`, and `bool`. `vector<T>` is a homogeneous, zero-indexed packed vector of scalar `T` values. `table<T>` maps integer or named keys to values of one declared type `T`.

`struct` is the sole object-oriented mechanism. Its zero-argument member functions may access fields by name; a field hides an outer name of the same spelling. `this` is the whole current instance and permits calls to another member function: `this.advance()`.

Structure fields are typed values and may include a `vector<T>`:

```l0
struct Board { cells: vector<i8>; score: i8; }
let board: Board = Board { cells = [0, 0, 0], score = 0 }
```

Tables never gain methods as part of this model. Object behaviour belongs only to `struct` and its member functions.

## Expressions and control flow

`&&` and `||` use short-circuit evaluation: the right-hand expression is
evaluated only when its value is needed. This makes guarded accesses safe:

```l0
if ready && board[0] == 1 then
    print("winner")
end
```

The `+` operator concatenates two `string` values. Array literals infer their
scalar element type when no surrounding declaration supplies one; an empty
literal still needs an explicit type. `len(vector)` returns an `i32` length.

`while` loops support `break` and `continue`. The initial `for` form is an
inclusive `i32` range:

```l0
for i = 0, len(cells) - 1 do
    if cells[i] == 0 then
        continue
    end
    print(i)
end
```

## Modules

A source file run through the RLUA executable establishes its own directory as the module root. `require` accepts a string literal path relative to that root; an omitted extension means `.l0`. Absolute paths and paths that resolve outside the root are rejected. A canonical module is initialized once per VM and then returned from the module cache.

The receiving binding has type `module`:

```l0
let game: module = require("game")
```

Only explicit exports form a module's public interface:

```l0
export let score: i32 = 0
export struct Board { cells: vector<i8>; }
export function init()
    score = score + 1
    print(score)
end
```

An exported value is read through the namespace and an exported procedure is called only through that namespace. Procedures are deliberately zero-argument in this first slice:

```l0
game.init()
print(game.score)
```

An exported structure is qualified by the binding used for `require`:

```l0
let board: game.Board = game.Board { cells = [0, 0, 0] }
```

The first module slice exports a structure's data layout. Cross-module member method dispatch for exported structures is reserved for the next object-model revision; declare behaviour as an exported module procedure in the meantime.

## Host functions

The host can register a function before compiling a source unit. Its argument and result types become part of compilation, so a call from L0 is checked just like a built-in function and compiled to an external-call opcode.

Rust hosts use `Vm::register_rust_function(name, argument_types, result_type, callback)` and then `vm.execute(source)`. The callback receives the typed L0 values and returns one typed value.

The C ABI provides the same path for `i32` functions: call `l0_register_i32_function`, then `l0_execute`. During the callback, `l0_to_i32(state, 0, ...)` and subsequent indices read the call arguments; the callback must push exactly one `i32` result with `l0_push_i32` and return zero. A non-zero return status aborts the L0 call.

## Runtime roadmap

1. **Current stage:** typed vector fields in structures and cached filesystem modules with explicit exports. Bytecode records the static numeric operand family, avoiding a runtime `Type` dispatch for binary arithmetic and comparisons. Named table keys are interned by the compiler and retained as shared `Rc<str>` values in bytecode and tables.
2. **Current runtime layer — garbage collector:** vectors, tables, structures, and strings are heap objects addressed by stable `HeapRef` handles, so copying a `Value` preserves object identity instead of copying its payload. Loaded module VMs are long-lived roots. The non-moving mark-and-sweep arena traces the operand stack, local slots, and those module roots; collection runs automatically after an allocation threshold and can also be requested explicitly through `Vm::collect_garbage()`. This makes cycles collectible without reference counting.
3. **Later stage:** extend the C callback helpers beyond `i32`, add richer module interfaces, cross-module struct method dispatch, parameter and return-value design, and standard libraries.

The GC stage must preserve the observable semantics above and collect cyclic tables and structures only when they become unreachable from all roots.
