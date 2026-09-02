# Calling host code from L0

Both examples register a function before compiling the `.l0` source. The
signature is known by the compiler, so incorrect argument count or type is
rejected before the VM starts execution.

## C callback

`c_host.c` registers `c_add(i32, i32) -> i32`, then executes `c_host.l0`.
The C callback reads its two arguments with `l0_to_i32`, pushes one result with
`l0_push_i32`, and returns `0` for success.

From the repository root:

```sh
cargo build --release
cc -std=c11 examples/host_functions/c_host.c -Iinclude -Ltarget/release -ll0 -o c_host
LD_LIBRARY_PATH=target/release ./c_host
```

On macOS, use `DYLD_LIBRARY_PATH=target/release`; on Windows, put
`l0.dll` beside `c_host.exe`.

## Rust callback

`rust_host.rs` registers `rust_multiply(i32, i32) -> i32` and executes
`rust_host.l0`. It is registered as a Cargo example:

```sh
cargo run --example rust_host
```

The expected result in both cases is `42`; the C example also prints the
calculation from inside its callback.
