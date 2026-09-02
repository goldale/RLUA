# FFI and complex L0 values

## Status

The current FFI supports calls from L0 scripts into host code, but it does
**not** yet provide direct access to the contents of managed L0 values.

`vector`, `table`, `struct`, and `string` live in the VM's non-moving
mark-and-sweep `Heap`. An L0 `Value` for one of these types holds a stable
`HeapRef`, not a copy of the object data. Copying such a value therefore
preserves object identity and permits cyclic object graphs to be collected.

Rust callbacks are currently declared as:

```rust
pub type L0RustFunction = fn(&[Value]) -> Result<Value, Error>;
```

Consequently, a Rust callback can receive and return a reference value, but it
cannot resolve its `HeapRef`: it has neither the `Vm` nor the `Heap`. Direct
reading or mutation of a vector, table, structure, or string is therefore not
supported by the current Rust FFI.

The current C ABI is narrower. It exports only the `i32... -> i32` route:
`l0_register_i32_function`, `l0_to_i32`, and `l0_push_i32`. C callbacks cannot
currently receive reference values at all.

## Proposed Rust heap-access extension

The following is a proposed API change. It is documented here so that the
extension can be implemented consistently; it is **not implemented in the
current release**.

### 1. Pass the heap to a Rust callback

Change the Rust callback type in `src/lib.rs` to accept mutable heap access:

```rust
pub type L0RustFunction = fn(&[Value], &mut Heap) -> Result<Value, Error>;
```

Then pass the heap at the Rust branch of `Vm::call_external`:

```rust
ExternalFunction::Rust(function) => {
    let arguments = &self.stack[base..];
    let mut heap = self.heap.borrow_mut();
    function(arguments, &mut heap)?
},
```

The callback receives a borrow that is valid only for that call. It must not
attempt to retain it. Because it receives `&mut Heap`, the Rust type system
also prevents two Rust callbacks from mutating the heap concurrently through
this API.

### 2. Expose a minimal public heap API

The following types and functions must be public for a host crate to inspect
the argument it received:

```rust
pub enum HeapObject {
    Array { bytes: Vec<u8>, element: Type },
    String(String),
    Table { entries: HashMap<TableKey, Value>, element: Type },
    Struct { values: Vec<Value>, layout: StructLayout },
}

pub struct Heap {
    // Internal fields remain private.
}

impl Heap {
    pub fn get(&self, reference: HeapRef) -> Result<&HeapObject, Error> { /* ... */ }
    pub fn get_mut(&mut self, reference: HeapRef) -> Result<&mut HeapObject, Error> { /* ... */ }
}

pub fn scalar_size(ty: &Type) -> Result<usize, Error> { /* ... */ }
pub fn encode_scalar(value: &Value, element: &Type, bytes: &mut Vec<u8>) -> Result<(), Error> { /* ... */ }
pub fn decode_scalar(bytes: &[u8], index: usize, element: &Type) -> Result<Value, Error> { /* ... */ }
```

`HeapRef` should remain an opaque handle: host code may obtain one by matching
a `Value`, but must not manufacture arbitrary references. Hosts modifying an
array should use `encode_scalar` rather than write bytes directly. This keeps
the packed scalar representation and its little-endian encoding valid.

Exposing raw `HeapObject` variants is a small initial API. A later stable API
should preferably provide checked operations such as `array_len`, `array_get`,
and `array_set`, so host code does not depend on the heap's internal storage
format.

### 3. Example: modify a vector from Rust

Once the extension above is implemented, a Rust callback can change an L0
vector in place:

```rust
use l0::{encode_scalar, scalar_size, Error, Heap, HeapObject, Value};

fn rust_modify_array(args: &[Value], heap: &mut Heap) -> Result<Value, Error> {
    let [Value::Array(heap_ref, _expected_type), Value::I32(index), new_value] = args else {
        return Err(Error::Runtime("expected: vector<i32>, i32, i32".into()));
    };

    if *index < 0 {
        return Err(Error::Runtime("array index must not be negative".into()));
    }

    let HeapObject::Array { bytes, element } = heap.get_mut(*heap_ref)? else {
        return Err(Error::Runtime("expected an array heap object".into()));
    };

    let width = scalar_size(element)?;
    let start = (*index as usize)
        .checked_mul(width)
        .ok_or_else(|| Error::Runtime("array index is too large".into()))?;
    let end = start
        .checked_add(width)
        .ok_or_else(|| Error::Runtime("array index is too large".into()))?;
    if end > bytes.len() {
        return Err(Error::Runtime("array index is out of bounds".into()));
    }

    let mut encoded = Vec::with_capacity(width);
    encode_scalar(new_value, element, &mut encoded)?;
    bytes[start..end].copy_from_slice(&encoded);
    Ok(Value::Bool(true))
}
```

The corresponding L0 script is:

```l0
let my_data: vector<i32> = [10, 20, 30]
let changed: bool = rust_modify_array(my_data, 1, 999)
print(my_data[1])
```

It prints `999`. `changed` is intentionally bound because an external call is
an expression in the current L0 grammar; a standalone call statement is not
yet supported.

## Current patterns when direct access is unnecessary

### Handle / ID: data owned by Rust

For large data sets that are primarily processed by host code, keep the data
in Rust and expose a scalar handle to L0. L0 controls the host object through
small typed functions instead of passing an L0 vector to Rust.

```rust
use std::sync::Mutex;
use l0::{Error, Value};

static GAME_BOARDS: Mutex<Vec<Vec<i8>>> = Mutex::new(Vec::new());

fn create_board(_args: &[Value]) -> Result<Value, Error> {
    let mut boards = GAME_BOARDS.lock().map_err(|_| Error::Runtime("board store is unavailable".into()))?;
    let id = i32::try_from(boards.len()).map_err(|_| Error::Runtime("too many boards".into()))?;
    boards.push(vec![0; 81]);
    Ok(Value::I32(id))
}

fn set_cell(args: &[Value]) -> Result<Value, Error> {
    let [Value::I32(board_id), Value::I32(index), Value::I8(value)] = args else {
        return Err(Error::Runtime("expected: board id, index, i8 value".into()));
    };
    let mut boards = GAME_BOARDS.lock().map_err(|_| Error::Runtime("board store is unavailable".into()))?;
    let board = boards
        .get_mut(usize::try_from(*board_id).map_err(|_| Error::Runtime("invalid board id".into()))?)
        .ok_or_else(|| Error::Runtime("unknown board id".into()))?;
    let cell = board
        .get_mut(usize::try_from(*index).map_err(|_| Error::Runtime("invalid cell index".into()))?)
        .ok_or_else(|| Error::Runtime("cell index is out of bounds".into()))?;
    *cell = *value;
    Ok(Value::Bool(true))
}
```

```l0
let board_id: i32 = create_board()
let changed: bool = set_cell(board_id, 42, 1)
```

The lifetime of a host-side handle must be explicit. A production integration
should provide a host function to release the object, or define when the host
clears its registry.

### Flattening: pass scalar fields separately

For a small record such as a position, pass individual scalar fields instead
of a structure:

```l0
struct Position { x: f32; y: f32; z: f32; }

let p: Position = Position { x = 10.0, y = 20.0, z = 5.0 }
let updated: bool = rust_update_position(p.x, p.y, p.z)
```

This pattern works with the current Rust FFI for scalar signatures and is also
the only pattern currently available through the C ABI.

## C API direction

The Rust `Heap` is an internal Rust type and must not be exposed as a C ABI
pointer. To support complex values from C, add explicit ABI-safe operations
instead: opaque value/heap handles, type inspection, checked array/table/
structure accessors, and clearly defined lifetime rules while a callback is
active. That is a separate API design task from the Rust extension above.
