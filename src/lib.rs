//! L0: experimental typed Lua-like language with a stack bytecode VM.

pub mod compiler;
pub mod ffi;
pub mod ext;

#[cfg(test)]
mod tests;

pub use compiler::*;
pub use ffi::*;

use std::fmt;
use std::fs;
use std::rc::Rc;
use std::cell::RefCell;
use std::io::{self, Write};
use std::collections::{HashMap, VecDeque};
use std::path::Path;

pub const ABI_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringId(pub u32);

#[derive(Clone, Debug, Default)]
pub struct StringInterner {
    pub strings: Vec<Rc<str>>,
    pub lookup: HashMap<Rc<str>, StringId>,
}
impl StringInterner {
    pub fn new() -> Self { Self::default() }
    pub fn intern(&mut self, s: &str) -> StringId {
        if let Some(&id) = self.lookup.get(s) { return id; }
        let id = StringId(self.strings.len() as u32);
        let text: Rc<str> = Rc::from(s);
        self.strings.push(text.clone());
        self.lookup.insert(text, id);
        id
    }
    pub fn resolve(&self, id: StringId) -> &str { &self.strings[id.0 as usize] }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    I8, I16, I32, I64, U8, U16, U32, U64, F16, BF16, F32, F64, Bool, String,
    Array(Box<Type>), Tensor(Box<Type>, usize), Table(Box<Type>), TableKey, TableKeys,
    Struct(String), Module(String), DArray, DTensor,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I8 => write!(f, "i8"), Self::I16 => write!(f, "i16"),
            Self::I32 => write!(f, "i32"), Self::I64 => write!(f, "i64"),
            Self::U8 => write!(f, "u8"), Self::U16 => write!(f, "u16"),
            Self::U32 => write!(f, "u32"), Self::U64 => write!(f, "u64"),
            Self::F16 => write!(f, "f16"), Self::BF16 => write!(f, "bf16"), Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"), Self::Bool => write!(f, "bool"),
            Self::String => write!(f, "string"),
            Self::Array(inner) => write!(f, "vector<{}>", inner),
            Self::Tensor(inner, rank) => write!(f, "tensor<{}, {}>", inner, rank),
            Self::Table(inner) => write!(f, "table<{}>", inner),
            Self::TableKey => write!(f, "table_key"),
            Self::TableKeys => write!(f, "table_keys"),
            Self::Struct(name) => write!(f, "{name}"),
            Self::Module(_) => write!(f, "module"),
            Self::DArray => write!(f, "dArray"),
            Self::DTensor => write!(f, "dTensor"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeapRef(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    I8(i8), I16(i16), I32(i32), I64(i64), U8(u8), U16(u16), U32(u32), U64(u64),
    F16(u16), BF16(u16), F32(f32), F64(f64), Bool(bool), String(HeapRef),
    Array(HeapRef, Box<Type>),
    Tensor(HeapRef, Rc<Type>),
    Table(HeapRef, Box<Type>),
    TableKey(Rc<TableKey>),
    TableKeys(HeapRef),
    Struct(HeapRef, Rc<StructLayout>),
    Module(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TableKey { Index(i128), Name(Rc<str>) }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructField { pub name: String, pub ty: Type, pub index: usize }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructLayout { pub name: String, pub fields: Vec<StructField> }
impl Value {
    pub fn ty(&self) -> Type {
        match self {
            Self::I8(_) => Type::I8, Self::I16(_) => Type::I16, Self::I32(_) => Type::I32,
            Self::I64(_) => Type::I64, Self::U8(_) => Type::U8, Self::U16(_) => Type::U16,
            Self::U32(_) => Type::U32, Self::U64(_) => Type::U64, Self::F16(_) => Type::F16,
            Self::BF16(_) => Type::BF16, Self::F32(_) => Type::F32, Self::F64(_) => Type::F64, Self::Bool(_) => Type::Bool,
            Self::String(_) => Type::String,
            Self::Array(_, element) => Type::Array(element.clone()),
            Self::Table(_, element) => Type::Table(element.clone()),
            Self::TableKey(_) => Type::TableKey,
            Self::TableKeys(_) => Type::TableKeys,
            Self::Struct(_, layout) => Type::Struct(layout.name.clone()),
            Self::Module(id) => Type::Module(id.clone()),
            Self::Tensor(_, ty) => (**ty).clone(),
        }
    }
    pub fn pack_array(values: Vec<Value>, element: &Type) -> Result<Vec<u8>, Error> {
        let element_size = scalar_size(element)?;
        let mut bytes = Vec::with_capacity(values.len().checked_mul(element_size).ok_or_else(|| Error::Runtime("array is too large".into()))?);
        for value in values { encode_scalar(&value, element, &mut bytes)?; }
        Ok(bytes)
    }
}
#[derive(Clone, Debug)]
pub enum HeapObject {
    Array { bytes: Vec<u8>, element: Type },
    Struct { values: Vec<Value>, layout: Rc<StructLayout> },
    Tensor { bytes: Vec<u8>, element: Type, shape: Vec<usize> },
    Table { entries: HashMap<TableKey, Value>, element: Type },
    TableKeys(Vec<TableKey>),
    String(String),
}
#[derive(Clone, Debug)]
pub enum HeapSlot {
    Free { next_free: Option<usize> },
    Occupied { marked: bool, object: HeapObject },
}
#[derive(Debug)]
pub struct Heap {
    pub slots: Vec<HeapSlot>,
    pub free_head: Option<usize>,
    pub allocated_bytes: usize,
    pub threshold_bytes: usize,
}

impl Default for Heap {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            allocated_bytes: 0,
            threshold_bytes: 64 * 1024,
        }
    }
}

impl Heap {
    fn object_size(object: &HeapObject) -> usize {
        use std::mem::size_of;

        size_of::<HeapSlot>() + match object {
            HeapObject::Array { bytes, .. } | HeapObject::Tensor { bytes, .. } => bytes.capacity(),
            HeapObject::String(text) => text.capacity(),
            HeapObject::Table { entries, .. } => entries.capacity() * size_of::<(TableKey, Value)>(),
            HeapObject::TableKeys(keys) => keys.capacity() * size_of::<TableKey>(),
            HeapObject::Struct { values, layout } => {
                values.capacity() * size_of::<Value>()
                    + layout.fields.capacity() * size_of::<StructField>()
                    + layout.name.capacity()
            },
        }
    }
    pub fn allocate(&mut self, object: HeapObject) -> HeapRef {
        self.allocated_bytes = self.allocated_bytes.saturating_add(Self::object_size(&object));
        if let Some(index) = self.free_head {
            if let HeapSlot::Free { next_free } = self.slots[index] {
                self.free_head = next_free;
                self.slots[index] = HeapSlot::Occupied { marked: false, object };
                return HeapRef(index);
            }
        }
        let index = self.slots.len();
        self.slots.push(HeapSlot::Occupied { marked: false, object });
        HeapRef(index)
    }
    pub fn get(&self, reference: HeapRef) -> Result<&HeapObject, Error> {
        match self.slots.get(reference.0) {
            Some(HeapSlot::Occupied { object, .. }) => Ok(object),
            _ => Err(Error::Runtime("dangling heap reference".into())),
        }
    }
    pub fn get_mut(&mut self, reference: HeapRef) -> Result<&mut HeapObject, Error> {
        match self.slots.get_mut(reference.0) {
            Some(HeapSlot::Occupied { object, .. }) => Ok(object),
            _ => Err(Error::Runtime("dangling heap reference".into())),
        }
    }
    pub fn heap_ref(value: &Value) -> Option<HeapRef> {
        match value {
            Value::Array(reference, _) | Value::Tensor(reference, _) | Value::String(reference)
            | Value::Table(reference, _) | Value::TableKeys(reference) | Value::Struct(reference, _) => Some(*reference),
            _ => None,
        }
    }
    fn mark_reference(&mut self, root: HeapRef) {
        let mut work = vec![root];
        while let Some(reference) = work.pop() {
            let newly_marked = match self.slots.get_mut(reference.0) {
                Some(HeapSlot::Occupied { marked, .. }) => {
                    if *marked { false } else { *marked = true; true }
                }
                _ => false,
            };
            if !newly_marked { continue; }

            match &self.slots[reference.0] {
                HeapSlot::Occupied { object, .. } => match object {
                    HeapObject::Table { entries, .. } => {
                        work.extend(entries.values().filter_map(Self::heap_ref));
                    }
                    HeapObject::Struct { values, .. } => {
                        work.extend(values.iter().filter_map(Self::heap_ref));
                    }
                    HeapObject::Array { .. } | HeapObject::Tensor { .. } | HeapObject::String(_) | HeapObject::TableKeys(_) => {}
                },
                _ => {}
            }
        }
    }
    pub fn collect(&mut self, roots: impl IntoIterator<Item = HeapRef>) -> usize {
        for root in roots { self.mark_reference(root); }
        let mut reclaimed = 0;

        for slot in self.slots.iter_mut() {
            if let HeapSlot::Occupied { marked, object } = slot {
                if *marked {
                    *marked = false;
                } else {
                    self.allocated_bytes = self.allocated_bytes.saturating_sub(Self::object_size(object));
                    *slot = HeapSlot::Free { next_free: None };
                    reclaimed += 1;
                }
            }
        }
        while let Some(HeapSlot::Free { .. }) = self.slots.last() {
            self.slots.pop();
        }
        self.slots.shrink_to_fit();

        self.free_head = None;
        for index in (0..self.slots.len()).rev() {
            if let HeapSlot::Free { next_free } = &mut self.slots[index] {
                *next_free = self.free_head;
                self.free_head = Some(index);
            }
        }
        self.threshold_bytes = self.allocated_bytes.saturating_mul(2).max(64 * 1024);
        reclaimed
    }
    pub fn should_collect(&self) -> bool { self.allocated_bytes >= self.threshold_bytes }
}

pub fn table_key_display(key: &TableKey) -> String { match key { TableKey::Index(index) => format!("[{index}]"), TableKey::Name(name) => name.to_string() } }
pub fn table_key_from_value(heap: &Heap, value: &Value) -> Result<TableKey, Error> {
    let index = match value {
        Value::I8(v) => *v as i128, Value::I16(v) => *v as i128, Value::I32(v) => *v as i128, Value::I64(v) => *v as i128,
        Value::U8(v) => *v as i128, Value::U16(v) => *v as i128, Value::U32(v) => *v as i128, Value::U64(v) => *v as i128,
        Value::TableKey(key) => return Ok((**key).clone()),
        Value::String(reference) => return match heap.get(*reference)? {
            HeapObject::String(name) => Ok(TableKey::Name(Rc::from(name.as_str()))),
            _ => Err(Error::Runtime("string heap invariant broken".into())),
        },
        _ => return Err(Error::Runtime("table key must be an integer or string".into())),
    };
    Ok(TableKey::Index(index))
}

pub fn type_size(ty: &Type) -> Option<usize> {
    match ty {
        Type::I8 | Type::U8 | Type::Bool => Some(1),
        Type::I16 | Type::U16 | Type::F16 | Type::BF16 => Some(2),
        Type::I32 | Type::U32 | Type::F32 => Some(4),
        Type::I64 | Type::U64 | Type::F64 => Some(8),
        Type::Array(_) | Type::Tensor(_, _) | Type::Table(_)
            | Type::TableKey | Type::TableKeys | Type::Struct(_)
            | Type::String | Type::Module(_) | Type::DArray
            | Type::DTensor => None,
    }
}

pub fn scalar_size(ty: &Type) -> Result<usize, Error> { type_size(ty).ok_or_else(|| Error::Type("packed storage supports scalar field and element types only".into())) }

pub fn encode_scalar(value: &Value, element: &Type, bytes: &mut Vec<u8>) -> Result<(), Error> {
    if &value.ty() != element { return Err(Error::Runtime("VM array type invariant broken".into())); }
    match value {
        Value::I8(v) => bytes.push(*v as u8), Value::U8(v) => bytes.push(*v),
        Value::Bool(v) => bytes.push(u8::from(*v)),
        Value::I16(v) => bytes.extend(v.to_le_bytes()),Value::U16(v) | Value::F16(v) | Value::BF16(v) => bytes.extend(v.to_le_bytes()),
        Value::I32(v) => bytes.extend(v.to_le_bytes()), Value::U32(v) => bytes.extend(v.to_le_bytes()),
        Value::F32(v) => bytes.extend(v.to_bits().to_le_bytes()),
        Value::I64(v) => bytes.extend(v.to_le_bytes()), Value::U64(v) => bytes.extend(v.to_le_bytes()),
        Value::F64(v) => bytes.extend(v.to_bits().to_le_bytes()),
        _ => return Err(Error::Type("packed arrays can contain only scalar values".into())),
    }
    Ok(())
}

pub fn decode_scalar(bytes: &[u8], index: usize, element: &Type) -> Result<Value, Error> {
    let size = scalar_size(element)?;
    let offset = index.checked_mul(size).ok_or_else(|| Error::Runtime("array index too large".into()))?;
    if offset + size > bytes.len() {
        return Err(Error::Runtime(format!("array index {} is out of bounds", index)));
    }

    let cell = &bytes[offset..offset + size];
    match element {
        Type::I8 => Ok(Value::I8(cell[0] as i8)), Type::U8 => Ok(Value::U8(cell[0])),
        Type::Bool => Ok(Value::Bool(cell[0] != 0)),
        Type::I16 => Ok(Value::I16(i16::from_le_bytes(cell.try_into().unwrap()))),
        Type::U16 => Ok(Value::U16(u16::from_le_bytes(cell.try_into().unwrap()))),
        Type::F16 => Ok(Value::F16(u16::from_le_bytes(cell.try_into().unwrap()))),
        Type::BF16 => Ok(Value::BF16(u16::from_le_bytes(cell.try_into().unwrap()))),
        Type::I32 => Ok(Value::I32(i32::from_le_bytes(cell.try_into().unwrap()))),
        Type::U32 => Ok(Value::U32(u32::from_le_bytes(cell.try_into().unwrap()))),
        Type::F32 => Ok(Value::F32(f32::from_bits(u32::from_le_bytes(cell.try_into().unwrap())))),
        Type::I64 => Ok(Value::I64(i64::from_le_bytes(cell.try_into().unwrap()))),
        Type::U64 => Ok(Value::U64(u64::from_le_bytes(cell.try_into().unwrap()))),
        Type::F64 => Ok(Value::F64(f64::from_bits(u64::from_le_bytes(cell.try_into().unwrap())))),
        _ => Err(Error::Type("not a scalar type".into())),
    }
}
pub fn write_scalar(bytes: &mut [u8], index: usize, value: &Value, element: &Type) -> Result<(), Error> {
    let size = scalar_size(element)?;
    let offset = index.checked_mul(size).ok_or_else(|| Error::Runtime("array index too large".into()))?;
    if offset + size > bytes.len() {
        return Err(Error::Runtime(format!("array index {} is out of bounds", index)));
    }
    if &value.ty() != element { return Err(Error::Runtime("VM array type invariant broken".into())); }
    let cell = &mut bytes[offset..offset + size];
    match value {
        Value::U8(v) => cell[0] = *v,
        Value::I8(v) => cell[0] = *v as u8,
        Value::Bool(v) => cell[0] = u8::from(*v),
        Value::I16(v) => cell.copy_from_slice(&v.to_le_bytes()),
        Value::U16(v) | Value::F16(v) | Value::BF16(v) => cell.copy_from_slice(&v.to_le_bytes()),
        Value::I32(v) => cell.copy_from_slice(&v.to_le_bytes()),
        Value::U32(v) => cell.copy_from_slice(&v.to_le_bytes()),
        Value::F32(v) => cell.copy_from_slice(&v.to_bits().to_le_bytes()),
        Value::I64(v) => cell.copy_from_slice(&v.to_le_bytes()),
        Value::U64(v) => cell.copy_from_slice(&v.to_le_bytes()),
        Value::F64(v) => cell.copy_from_slice(&v.to_bits().to_le_bytes()),
        _ => return Err(Error::Type("packed arrays can contain only scalar values".into())),
    }
    Ok(())
}
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I8(v) => write!(f, "{v}"),
            Self::I16(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"),
            Self::I64(v) => write!(f, "{v}"),
            Self::U8(v) => write!(f, "{v}"),
            Self::U16(v) => write!(f, "{v}"),
            Self::U32(v) => write!(f, "{v}"),
            Self::U64(v) => write!(f, "{v}"),
            Self::F16(v) => write!(f, "{}", f16_to_f32(*v)),
            Self::F32(v) => write!(f, "{v}"),
            Self::BF16(v) => write!(f, "{}", bf16_to_f32(*v)),
            Self::F64(v) => write!(f, "{v}"), Self::Bool(v) => write!(f, "{v}"),
            Self::String(reference) => write!(f, "string@{}", reference.0),
            Self::Array(reference, element) => write!(f, "vector<{}>@{}", element, reference.0),
            Self::Tensor(reference, ty) => write!(f, "{}@{}", ty, reference.0),
            Self::Table(reference, element) => write!(f, "table<{}>@{}", element, reference.0),
            Self::TableKey(key) => write!(f, "{}", table_key_display(key)),
            Self::TableKeys(reference) => write!(f, "table_keys@{}", reference.0),
            Self::Struct(reference, layout) => write!(f, "{}@{}", layout.name, reference.0),
            Self::Module(id) => write!(f, "module({id})")
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceLocation { pub offset: usize, pub line: usize, pub column: usize }

impl SourceLocation {
    pub fn at(chars: &[char], offset: usize) -> Self {
        let mut line = 1; let mut column = 1;
        for ch in &chars[..offset.min(chars.len())] {
            if *ch == '\n' { line += 1; column = 1; } else { column += 1; }
        }
        Self { offset, line, column }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Lex(String), Parse(String), Type(String), Runtime(String),
    Located { source: Box<Error>, location: SourceLocation }
}
impl Error {
    pub fn at(self, location: SourceLocation) -> Self {
        match self {
            Self::Located { .. } => self,
            source => Self::Located { source: Box::new(source), location }
        }
    }
    pub fn location(&self) -> Option<SourceLocation> {
        match self { Self::Located { location, .. } => Some(*location), _ => None }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lex(message) => write!(f, "lex error: {message}"),
            Self::Parse(message) => write!(f, "parse error: {message}"),
            Self::Type(message) => write!(f, "type error: {message}"),
            Self::Runtime(message) => write!(f, "runtime error: {message}"),
            Self::Located { source, location } => write!(f, "{source} at line {}, column {}", location.line, location.column),
        }
    }
}
impl std::error::Error for Error {}

pub fn types_compatible(expected: &Type, found: &Type) -> bool {
    if expected == found { return true; }
    match (expected, found) {
        (Type::Module(expected_id), Type::Module(_)) if expected_id.is_empty() => true,
        (Type::DArray, Type::Array(_)) | (Type::Array(_), Type::DArray) => true,
        (Type::DTensor, Type::Tensor(_, _)) | (Type::Tensor(_, _), Type::DTensor) => true,
        (Type::F32, Type::F16 | Type::BF16) => true,
        (Type::F64, Type::F16 | Type::BF16 | Type::F32) => true,
        (Type::I16, Type::I8 | Type::U8) => true,
        (Type::I32, Type::I8 | Type::U8 | Type::I16 | Type::U16) => true,
        (Type::I64, Type::I8 | Type::U8 | Type::I16 | Type::U16 | Type::I32 | Type::U32) => true,
        (Type::U16, Type::U8) => true,
        (Type::U32, Type::U8 | Type::U16) => true,
        (Type::U64, Type::U8 | Type::U16 | Type::U32) => true,
        _ => false
    }
}
pub fn is_numeric(t: &Type) -> bool {
    !matches!(t, Type::Bool|Type::String|Type::Array(_)|Type::Tensor(_, _)|Type::Table(_)|Type::TableKey|Type::TableKeys|Type::Struct(_)|Type::Module(_)|Type::DArray|Type::DTensor)
}
pub fn is_integer(t: &Type) -> bool {
    matches!(t, Type::I8|Type::I16|Type::I32|Type::I64|Type::U8|Type::U16|Type::U32|Type::U64)
}
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    half::bf16::from_bits(bits).to_f32()
}
#[inline]
pub fn f32_to_bf16(value: f32) -> u16 {
    half::bf16::from_f32(value).to_bits()
}
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}
pub fn int_value(n: i128, ty: &Type) -> Result<Value, Error> {
    match ty {
        Type::I8 => i8::try_from(n).map(Value::I8).map_err(|_| Error::Type(format!("{n} does not fit in i8"))),
        Type::I16 => i16::try_from(n).map(Value::I16).map_err(|_| Error::Type(format!("{n} does not fit in i16"))),
        Type::I32 => i32::try_from(n).map(Value::I32).map_err(|_| Error::Type(format!("{n} does not fit in i32"))),
        Type::I64 => i64::try_from(n).map(Value::I64).map_err(|_| Error::Type(format!("{n} does not fit in i64"))),
        Type::U8 => u8::try_from(n).map(Value::U8).map_err(|_| Error::Type(format!("{n} does not fit in u8"))),
        Type::U16 => u16::try_from(n).map(Value::U16).map_err(|_| Error::Type(format!("{n} does not fit in u16"))),
        Type::U32 => u32::try_from(n).map(Value::U32).map_err(|_| Error::Type(format!("{n} does not fit in u32"))),
        Type::U64 => u64::try_from(n).map(Value::U64).map_err(|_| Error::Type(format!("{n} does not fit in u64"))),
        _ => Err(Error::Type(format!("integer literal cannot initialize {ty}"))),
    }
}
pub fn float_value(n: f64, ty: &Type) -> Value {
    match ty {
        Type::F16 => Value::F16(f32_to_f16(n as f32)),
        Type::BF16 => Value::BF16(f32_to_bf16(n as f32)),
        Type::F32 => Value::F32(n as f32),
        _ => Value::F64(n)
    }
}
pub type L0RustFunction = fn(&[Value], &RefCell<Heap>) -> Result<Value, Error>;

#[derive(Clone)]
pub enum ExternalFunction {
    Rust(L0RustFunction),
    C(crate::ffi::L0CFunction),
}
#[derive(Clone)]
pub struct RegisteredExternal {
    pub signature: HostSignature,
    pub function: ExternalFunction
}
pub struct FfiCall { pub arguments: Vec<Value>, pub results: Vec<Value> }
pub struct ModuleInstance { pub artifact: ModuleArtifact, pub vm: Vm }
#[derive(Clone, Copy, Debug)]
pub struct ActiveDestructor { pub slot: usize, pub target: usize }
pub struct Vm {
    pub stack: Vec<Value>,
    pub stack_ptr: usize,
    pub locals: Vec<Value>,
    pub output: Vec<String>,
    pub interactive: bool,
    pub input: VecDeque<String>,
    pub modules: HashMap<String, ModuleInstance>,
    pub extern_functions: HashMap<String, RegisteredExternal>,
    pub heap: Rc<RefCell<Heap>>,
    pub gc_owner: bool,
    pub callback_state: Option<*mut crate::ffi::L0State>,
    pub random_state: u64,
    pub active_destructors: Vec<ActiveDestructor>,

    pub native_modules: HashMap<String, Box<dyn crate::ext::NativeModule>>,
}

impl Default for Vm {
    fn default() -> Self {
        let mut vm = Self {
            stack: vec![Value::Bool(false); 4096],
            stack_ptr: 0,
            locals: Vec::with_capacity(64),
            output: Vec::new(),
            interactive: false,
            input: VecDeque::new(),
            modules: HashMap::new(),
            extern_functions: HashMap::new(),
            heap: Rc::new(RefCell::new(Heap::default())),
            gc_owner: true,
            callback_state: None,
            random_state: 0x5EED_CAFE_D15C_A11E,
            active_destructors: Vec::new(),
            native_modules: HashMap::new(),
        };

        for ext in crate::ext::available_extensions() {
            vm.native_modules.insert(ext.name().to_string(), ext);
        }
        vm
    }
}

impl Vm {
    pub fn with_shared_heap(
        heap: Rc<RefCell<Heap>>,
        extern_functions: HashMap<String, RegisteredExternal>,
        callback_state: Option<*mut crate::ffi::L0State>
    ) -> Self {
        Self { heap, extern_functions, gc_owner: false, callback_state, ..Self::default() }
    }

    pub fn set_interactive(&mut self, interactive: bool) {
        self.interactive = interactive;
    }

    #[inline(always)]
    fn heap_ref(&self) -> &Heap { unsafe { &*self.heap.as_ptr() } }

    #[inline(always)]
    fn heap_mut(&mut self) -> &mut Heap { unsafe { &mut *self.heap.as_ptr() } }

    pub fn register_rust_function(
        &mut self, name: impl Into<String>, arguments: Vec<Type>, result: Type, function: L0RustFunction
    ) -> Result<(), Error> {
        self.register_external(name.into(), HostSignature { arguments, result }, ExternalFunction::Rust(function))
    }

    pub fn register_c_i32_function(&mut self, name: impl Into<String>, argument_count: usize, function: crate::ffi::L0CFunction) -> Result<(), Error> {
        self.register_external(name.into(), HostSignature { arguments: vec![Type::I32; argument_count], result: Type::I32 }, ExternalFunction::C(function))
    }

    pub fn register_external(&mut self, name: String, signature: HostSignature, function: ExternalFunction) -> Result<(), Error> {
        if name.is_empty() { return Err(Error::Type("external function name cannot be empty".into())); }
        if self.extern_functions.insert(name.clone(), RegisteredExternal { signature, function }).is_some() {
            return Err(Error::Type(format!("external function '{name}' is already registered")));
        }
        Ok(())
    }

    fn external_signatures(&self) -> HashMap<String, HostSignature> {
        self.extern_functions.iter().map(|(name, registered)| (name.clone(), registered.signature.clone())).collect()
    }

    pub fn execute(&mut self, source: &str) -> Result<Vec<String>, Error> {
        let (program, strings) = Parser::new(lex(source)?).into_program()?;
        let code = Compiler::with_extern_functions(self.external_signatures()).with_strings(strings).compile(program)?;
        self.output.clear();
        Ok(self.run(&code)?.to_vec())
    }

    pub fn push_input(&mut self, data: String) {
        self.input.push_back(data);
    }

    pub fn execute_file(&mut self, path: impl AsRef<Path>) -> Result<Vec<String>, Error> {
        let path = fs::canonicalize(path.as_ref())
            .map_err(|error| Error::Runtime(format!("cannot open source file: {error}")))?;
        let root = path.parent()
            .ok_or_else(|| Error::Runtime("source file has no parent directory".into()))?
            .to_path_buf();
        let source = fs::read_to_string(&path)
            .map_err(|error| Error::Runtime(format!("cannot read source file: {error}")))?;
        let (program, strings) = Parser::new(lex(&source)?).into_program()?;
        let mut compiler = Compiler::with_module_root(root).with_strings(strings);
        compiler.extern_functions = self.external_signatures();
        let code = compiler.compile(program)?;
        self.output.clear();
        Ok(self.run(&code)?.to_vec())
    }

    fn roots(&self) -> Vec<HeapRef> {
        let mut roots = Vec::with_capacity(self.stack_ptr + self.locals.len());
        roots.extend(self.stack[..self.stack_ptr].iter().filter_map(Heap::heap_ref));
        roots.extend(self.locals.iter().filter_map(Heap::heap_ref));
        for module in self.modules.values() { roots.extend(module.vm.roots()); }
        roots
    }

    pub fn collect_garbage(&mut self) -> usize {
        let roots = self.roots();
        self.heap_mut().collect(roots)
    }

    fn collect_if_needed(&mut self) {
        if self.gc_owner && self.heap_ref().should_collect() { self.collect_garbage(); }
    }

    pub fn allocate(&mut self, object: HeapObject) -> HeapRef { self.heap_mut().allocate(object) }

    fn tensor_shape_from_value(&self, value: &Value) -> Result<Vec<usize>, Error> {
        let Value::Array(reference, element) = value else { return Err(Error::Runtime("tensor shape must be vector<u64>".into())); };
        if element.as_ref() != &Type::U64 { return Err(Error::Runtime("tensor shape must be vector<u64>".into())); }
        let bytes = match self.heap_ref().get(*reference)? { HeapObject::Array { bytes, element } if element == &Type::U64 => bytes.clone(), _ => return Err(Error::Runtime("tensor shape heap invariant broken".into())) };
        let mut shape = Vec::with_capacity(bytes.len() / 8);
        for chunk in bytes.chunks_exact(8) {
            shape.push(usize::try_from(u64::from_le_bytes(chunk.try_into().expect("exact chunk"))).map_err(|_| Error::Runtime("tensor dimension is too large".into()))?);
        }
        Ok(shape)
    }

    fn tensor_offset(shape: &[usize], indices: &[usize]) -> Result<usize, Error> {
        if shape.len() != indices.len() { return Err(Error::Runtime(format!("tensor rank {} requires {} index(es)", shape.len(), shape.len()))); }
        let mut offset = 0usize;
        for (&dimension, &index) in shape.iter().zip(indices) {
            if index >= dimension { return Err(Error::Runtime(format!("tensor index {index} is out of bounds for dimension {dimension}"))); }
            offset = offset.checked_mul(dimension).and_then(|value| value.checked_add(index)).ok_or_else(|| Error::Runtime("tensor index is too large".into()))?;
        }
        Ok(offset)
    }

    fn pop_tensor_indices(&mut self, rank: usize) -> Result<Vec<usize>, Error> {
        let mut indices = Vec::with_capacity(rank);
        for _ in 0..rank { indices.push(integer_to_usize(&self.pop()?)?); }
        indices.reverse();
        Ok(indices)
    }

    fn next_random_unit(&mut self) -> f64 {
        let mut state = self.random_state;
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        self.random_state = state;
        (state >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    fn make_tensor_bytes(&mut self, init: TensorInit, element: &Type, shape: &[usize]) -> Result<Vec<u8>, Error> {
        let element_size = scalar_size(element)?;
        let count = shape.iter().try_fold(1usize, |count, dimension| count.checked_mul(*dimension)).ok_or_else(|| Error::Runtime("tensor is too large".into()))?;
        let total_bytes = count.checked_mul(element_size).ok_or_else(|| Error::Runtime("tensor is too large".into()))?;
        if matches!(init, TensorInit::Zeros) {
            return Ok(vec![0; total_bytes]);
        }
        let mut bytes = Vec::with_capacity(total_bytes);
        for _ in 0..count {
            let value = match init {
                TensorInit::Random => match element {
                    Type::F16 => Value::F16(f32_to_f16(self.next_random_unit() as f32)),
                    Type::BF16 => Value::BF16(f32_to_bf16(self.next_random_unit() as f32)),
                    Type::F32 => Value::F32(self.next_random_unit() as f32),
                    Type::F64 => Value::F64(self.next_random_unit()),
                    _ => return Err(Error::Runtime("random tensor element invariant broken".into())),
                },
                TensorInit::Zeros => unreachable!("zero tensors return before scalar encoding"),
            };
            encode_scalar(&value, element, &mut bytes)?;
        }
        Ok(bytes)
    }

    fn format_value(&self, value: &Value) -> Result<String, Error> {
        match value {
            Value::Array(reference, element) => match self.heap_ref().get(*reference)? {
                HeapObject::Array { bytes, element: stored_element } if stored_element == element.as_ref() => {
                    let size = scalar_size(element)?;
                    let mut values = Vec::with_capacity(bytes.len() / size);
                    for index in 0..bytes.len() / size { values.push(decode_scalar(bytes, index, element)?.to_string()); }
                    Ok(format!("[{}]", values.join(", ")))
                },
                _ => Err(Error::Runtime("array heap invariant broken".into())),
            },
            Value::Tensor(reference, ty) => match self.heap_ref().get(*reference)? {
                HeapObject::Tensor { shape, .. } => Ok(format!("{}[{}]", ty, shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", "))),
                _ => Err(Error::Runtime("tensor heap invariant broken".into())),
            },
            Value::String(reference) => match self.heap_ref().get(*reference)? { HeapObject::String(text) => Ok(text.clone()), _ => Err(Error::Runtime("string heap invariant broken".into())) },
            Value::Table(reference, element) => match self.heap_ref().get(*reference)? { HeapObject::Table { entries, .. } => Ok(format!("table<{}>({})", element, entries.len())), _ => Err(Error::Runtime("table heap invariant broken".into())) },
            Value::TableKey(key) => Ok(table_key_display(key)),
            Value::TableKeys(reference) => match self.heap_ref().get(*reference)? { HeapObject::TableKeys(keys) => Ok(format!("[{}]", keys.iter().map(table_key_display).collect::<Vec<_>>().join(", "))), _ => Err(Error::Runtime("table_keys heap invariant broken".into())) },
            Value::Struct(reference, _) => {
                let (values, layout) = match self.heap_ref().get(*reference)? { HeapObject::Struct { values, layout } => (values.clone(), layout.clone()), _ => return Err(Error::Runtime("struct heap invariant broken".into())) };
                let mut fields = Vec::with_capacity(layout.fields.len());
                for field in &layout.fields { fields.push(format!("{}: {}", field.name, self.format_value(values.get(field.index).ok_or_else(|| Error::Runtime("invalid struct field index".into()))?)?)); }
                Ok(format!("{}{{{}}}", layout.name, fields.join(", ")))
            },
            _ => Ok(value.to_string()),
        }
    }

    fn call_external(&mut self, name: &str, argument_count: usize) -> Result<(), Error> {
        let registered = self.extern_functions.get(name).cloned().ok_or_else(|| Error::Runtime(format!("external function '{name}' is not registered")))?;
        if registered.signature.arguments.len() != argument_count || self.stack_ptr < argument_count { return Err(Error::Runtime("external call stack invariant broken".into())); }
        let base = self.stack_ptr - argument_count;
        for (value, expected) in self.stack[base..self.stack_ptr].iter().zip(registered.signature.arguments.iter()) {
            if !types_compatible(&value.ty(), expected) { return Err(Error::Runtime(format!("external function '{name}' received an invalid argument type"))); }
        }
        let result = match registered.function {
            ExternalFunction::Rust(function) => function(&self.stack[base..self.stack_ptr], &self.heap)?,
            ExternalFunction::C(function) => {
                let state = self.callback_state.ok_or_else(|| Error::Runtime(format!("C function '{name}' requires execution through L0State")))?;
                unsafe { (*state).ffi_call = Some(FfiCall { arguments: self.stack[base..self.stack_ptr].to_vec(), results: Vec::new() }); }
                let status = unsafe { function(state) };
                let ffi_call = unsafe { (*state).ffi_call.take().ok_or_else(|| Error::Runtime("missing C call context".into()))? };
                if status != 0 { self.stack_ptr = base; return Err(Error::Runtime(format!("C function '{name}' failed with status {status}"))); }
                if ffi_call.results.len() != 1 { self.stack_ptr = base; return Err(Error::Runtime(format!("C function '{name}' must push exactly one result"))); }
                ffi_call.results.into_iter().next().expect("checked external result")
            },
        };
        if !types_compatible(&registered.signature.result, &result.ty()) {
            return Err(Error::Runtime(format!("external function '{name}' returned {}; expected {}", result.ty(), registered.signature.result)));
        }
        let final_result = if is_numeric(&registered.signature.result) && result.ty() != registered.signature.result {
            cast_numeric(result, &registered.signature.result)?
        } else {
            result
        };
        self.stack_ptr = base;
        self.push(final_result);
        Ok(())
    }

    pub fn run(&mut self, code: &FlatBytecode) -> Result<&[String], Error> { self.run_from(code, 0, false) }

    pub fn run_from(&mut self, code: &FlatBytecode, pc: usize, terminal_return: bool) -> Result<&[String], Error> {
        match self.run_from_inner(code, pc, terminal_return, None) {
            Ok(_) => Ok(&self.output),
            Err(error) => {
                self.unwind_destructors(code);
                Err(error)
            },
        }
    }

    fn unwind_destructors(&mut self, code: &FlatBytecode) {
        while let Some(destructor) = self.active_destructors.pop() {
            let _ = self.run_from_inner(code, destructor.target, true, Some(destructor.slot));
        }
    }

    fn run_from_inner(&mut self, code: &FlatBytecode, mut pc: usize, terminal_return: bool, mut current_receiver: Option<usize>) -> Result<&[String], Error> {
        let mut call_stack: Vec<(usize, Option<usize>)> = Vec::new();

        while pc < code.words.len() {
            let (instruction, next_pc) = code.decode(pc)?;
            match instruction {
            DecodedOp::AddI32 => {
                let right = self.pop_compiled();
                let left = self.pop_compiled();
                let (Value::I32(left), Value::I32(right)) = (left, right) else {
                    return Err(Error::Runtime("VM i32 addition invariant broken".into()));
                };
                let result = left.checked_add(right)
                    .ok_or_else(|| Error::Runtime("addition overflow".into()))?;
                self.push(Value::I32(result));
            },
            DecodedOp::AddF32 => {
                let right = self.pop_compiled();
                let left = self.pop_compiled();
                let (Value::F32(left), Value::F32(right)) = (left, right) else {
                    return Err(Error::Runtime("VM f32 addition invariant broken".into()));
                };
                self.push(Value::F32(left + right));
            },
            DecodedOp::AddF64 => {
                let right = self.pop_compiled();
                let left = self.pop_compiled();
                let (Value::F64(left), Value::F64(right)) = (left, right) else {
                    return Err(Error::Runtime("VM f64 addition invariant broken".into()));
                };
                self.push(Value::F64(left + right));
            },
            DecodedOp::Push(v) => self.push(v.clone()),
            DecodedOp::MakeString(s) => { let reference = self.allocate(HeapObject::String(s.to_string())); self.push(Value::String(reference)); self.collect_if_needed(); },
            DecodedOp::Input(ty) => { let val = self.read_input(ty)?; self.push(val); self.collect_if_needed(); },
            DecodedOp::Require(module) => self.load_module((**module).clone())?,
            DecodedOp::Load(slot) => self.push(self.locals.get(slot).cloned().ok_or_else(|| Error::Runtime("invalid local slot".into()))?),
            DecodedOp::LoadCurrentReceiver => { let receiver = current_receiver.ok_or_else(|| Error::Runtime("this is available only inside a method".into()))?; self.push(self.locals.get(receiver).cloned().ok_or_else(|| Error::Runtime("invalid method receiver".into()))?); },
            DecodedOp::LoadCurrentField(field) => { let receiver = current_receiver.ok_or_else(|| Error::Runtime("field access outside method".into()))?; let Value::Struct(reference, _) = self.locals.get(receiver).ok_or_else(|| Error::Runtime("invalid method receiver".into()))? else { return Err(Error::Runtime("VM method receiver invariant broken".into())); }; let value = match self.heap_ref().get(*reference)? { HeapObject::Struct { values, .. } => values.get(field.index).cloned().ok_or_else(|| Error::Runtime("invalid struct field index".into()))?, _ => return Err(Error::Runtime("struct heap invariant broken".into())) }; self.push(value); },
            DecodedOp::Store(slot) => {
                let v = self.pop()?;
                if slot >= self.locals.len() { self.locals.resize(slot + 1, Value::Bool(false)); }
                self.locals[slot] = v;
            },
            DecodedOp::StoreIndex(slot, element) => {
                let value = self.pop()?;
                let index = integer_to_usize(&self.pop()?)?;
                let reference = match self.locals.get(slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? {
                    Value::Array(reference, _) => *reference,
                    _ => return Err(Error::Runtime("VM array slot invariant broken".into())),
                };
                match self.heap_mut().get_mut(reference)? {
                    HeapObject::Array { bytes, .. } => {
                        write_scalar(bytes, index, &value, element.as_ref())?;
                    },
                    _ => return Err(Error::Runtime("array heap invariant broken".into())),
                }
            },
            DecodedOp::StoreTableIndex(slot, _element) => { let value = self.pop()?; let key_value = self.pop()?; let key = table_key_from_value(self.heap_ref(), &key_value)?; let reference = match self.locals.get(slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? { Value::Table(reference, _) => *reference, _ => return Err(Error::Runtime("VM table slot invariant broken".into())), }; match self.heap_mut().get_mut(reference)? { HeapObject::Table { entries, .. } => { entries.insert(key, value); }, _ => return Err(Error::Runtime("table heap invariant broken".into())), } },
            DecodedOp::StoreTensorIndex(slot, element, rank) => {
                let value = self.pop()?;
                let indices = self.pop_tensor_indices(rank)?;
                let Value::Tensor(reference, _) = self.locals.get(slot).cloned().ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM tensor slot invariant broken".into())); };
                match self.heap_mut().get_mut(reference)? {
                    HeapObject::Tensor { bytes, shape, .. } if shape.len() == rank => {
                        let offset = Self::tensor_offset(shape, &indices)?;
                        write_scalar(bytes, offset, &value, element.as_ref())?;
                    },
                    _ => return Err(Error::Runtime("tensor heap invariant broken".into())),
                }
            },
            DecodedOp::StoreTensorIndexF32(slot, rank) => {
                let value = self.pop()?;
                let indices = self.pop_tensor_indices(rank)?;
                let Value::Tensor(reference, _) = self.locals.get(slot).cloned().ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM tensor slot invariant broken".into())); };
                match self.heap_mut().get_mut(reference)? {
                    HeapObject::Tensor { bytes, shape, .. } => {
                        let offset = Self::tensor_offset(shape, &indices)?;
                        let start = offset.checked_mul(4).ok_or_else(|| Error::Runtime("tensor offset is too large".into()))?;
                        if start + 4 > bytes.len() { return Err(Error::Runtime("tensor offset out of bounds".into())); }
                        if let Value::F32(v) = value {
                            bytes[start..start + 4].copy_from_slice(&v.to_bits().to_le_bytes());
                        }
                    },
                    _ => return Err(Error::Runtime("tensor heap invariant broken".into())),
                }
            },
            DecodedOp::StoreField(slot, field) => { let new_value = self.pop()?; let reference = match self.locals.get(slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? { Value::Struct(reference, _) => *reference, _ => return Err(Error::Runtime("VM struct slot invariant broken".into())), }; match self.heap_mut().get_mut(reference)? { HeapObject::Struct { values, .. } => { *values.get_mut(field.index).ok_or_else(|| Error::Runtime("invalid struct field index".into()))? = new_value; }, _ => return Err(Error::Runtime("struct heap invariant broken".into())), } },
            DecodedOp::StoreFieldIndex(slot, field, element) => { let value = self.pop()?; let index = integer_to_usize(&self.pop()?)?; let Value::Struct(struct_reference, _) = self.locals.get(slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM struct slot invariant broken".into())); }; let array_reference = match self.heap_ref().get(*struct_reference)? { HeapObject::Struct { values, .. } => match values.get(field.index) { Some(Value::Array(reference, _)) => *reference, _ => return Err(Error::Runtime("VM struct vector field invariant broken".into())), }, _ => return Err(Error::Runtime("struct heap invariant broken".into())), }; match self.heap_mut().get_mut(array_reference)? { HeapObject::Array { bytes, .. } => write_scalar(bytes, index, &value, element.as_ref())?, _ => return Err(Error::Runtime("array heap invariant broken".into())), } },
            DecodedOp::StoreTableField(slot, name, _element) => { let value = self.pop()?; let reference = match self.locals.get(slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? { Value::Table(reference, _) => *reference, _ => return Err(Error::Runtime("VM table slot invariant broken".into())), }; match self.heap_mut().get_mut(reference)? { HeapObject::Table { entries, .. } => { entries.insert(TableKey::Name(name.clone()), value); }, _ => return Err(Error::Runtime("table heap invariant broken".into())), } },
            DecodedOp::StoreCurrentField(field) => { let new_value = self.pop()?; let receiver = current_receiver.ok_or_else(|| Error::Runtime("field assignment outside method".into()))?; let reference = match self.locals.get(receiver).ok_or_else(|| Error::Runtime("invalid method receiver".into()))? { Value::Struct(reference, _) => *reference, _ => return Err(Error::Runtime("VM method receiver invariant broken".into())), }; match self.heap_mut().get_mut(reference)? { HeapObject::Struct { values, .. } => { *values.get_mut(field.index).ok_or_else(|| Error::Runtime("invalid struct field index".into()))? = new_value; }, _ => return Err(Error::Runtime("struct heap invariant broken".into())), } },
            DecodedOp::MakeArray(len, ty) => { if self.stack_ptr < len { return Err(Error::Runtime("stack underflow".into())); } let at = self.stack_ptr - len; let values = self.stack[at..self.stack_ptr].to_vec(); self.stack_ptr = at; let bytes = Value::pack_array(values, ty.as_ref())?; let reference = self.allocate(HeapObject::Array { bytes, element: ty.as_ref().clone() }); self.push(Value::Array(reference, Box::new(ty.as_ref().clone()))); self.collect_if_needed(); },
            DecodedOp::MakeTensor(init, element, rank) => {
                let shape_value = self.pop()?;
                let shape = self.tensor_shape_from_value(&shape_value)?;
                if shape.len() != rank { return Err(Error::Runtime(format!("tensor rank {rank} requires {rank} dimensions, got {}", shape.len()))); }
                let bytes = self.make_tensor_bytes(init, element.as_ref(), &shape)?;
                let reference = self.allocate(HeapObject::Tensor { bytes, element: element.as_ref().clone(), shape });
                self.push(Value::Tensor(reference, Rc::new(Type::Tensor(Box::new(element.as_ref().clone()), rank))));
                self.collect_if_needed();
            },
            DecodedOp::MakeTable(entries, element) => { let value_count = entries.iter().map(|entry| match entry { TableEntry::Index => 2usize, TableEntry::Name(_) => 1usize }).sum(); if self.stack_ptr < value_count { return Err(Error::Runtime("stack underflow".into())); } let at = self.stack_ptr - value_count; let values = self.stack[at..self.stack_ptr].to_vec(); self.stack_ptr = at; let mut cursor = 0; let mut table = HashMap::with_capacity(entries.len()); for entry in entries.iter() { let key = match entry { TableEntry::Index => { let key = table_key_from_value(self.heap_ref(), values.get(cursor).ok_or_else(|| Error::Runtime("stack underflow".into()))?)?; cursor += 1; key }, TableEntry::Name(name) => TableKey::Name(name.clone()), }; let value = values.get(cursor).cloned().ok_or_else(|| Error::Runtime("stack underflow".into()))?; cursor += 1; if table.insert(key.clone(), value).is_some() { return Err(Error::Runtime(format!("table key {} is declared more than once", table_key_display(&key)))); } } let reference = self.allocate(HeapObject::Table { entries: table, element: element.as_ref().clone() }); self.push(Value::Table(reference, Box::new(element.as_ref().clone()))); self.collect_if_needed(); },
            DecodedOp::MakeStruct(layout) => { if self.stack_ptr < layout.fields.len() { return Err(Error::Runtime("stack underflow".into())); } let at = self.stack_ptr - layout.fields.len(); let values = self.stack[at..self.stack_ptr].to_vec(); self.stack_ptr = at; let reference = self.allocate(HeapObject::Struct { values, layout: layout.clone() }); self.push(Value::Struct(reference, layout.clone())); self.collect_if_needed(); },
            DecodedOp::Index => { let index = integer_to_usize(&self.pop()?)?; let object = self.pop()?; let Value::Array(reference, element) = object else { return Err(Error::Runtime("VM array invariant broken".into())); }; let value = match self.heap_ref().get(reference)? { HeapObject::Array { bytes, element: stored_element } if stored_element == element.as_ref() => decode_scalar(bytes, index, &element)?, _ => return Err(Error::Runtime("array heap invariant broken".into())) }; self.push(value); },
            DecodedOp::TensorIndex(element, rank) => {
                let indices = self.pop_tensor_indices(rank)?;
                let object = self.pop()?;
                let Value::Tensor(reference, _) = object else { return Err(Error::Runtime("VM tensor invariant broken".into())); };
                let value = match self.heap_ref().get(reference)? {
                    HeapObject::Tensor { bytes, element: stored_element, shape } if stored_element == element.as_ref() && shape.len() == rank => decode_scalar(bytes, Self::tensor_offset(shape, &indices)?, element.as_ref())?,
                    _ => return Err(Error::Runtime("tensor heap invariant broken".into())),
                };
                self.push(value);
            },
            DecodedOp::TensorIndexF32(rank) => {
                let indices = self.pop_tensor_indices(rank)?;
                let object = self.pop()?;
                let Value::Tensor(reference, _) = object else { return Err(Error::Runtime("VM tensor invariant broken".into())); };
                let value = match self.heap_ref().get(reference)? {
                    HeapObject::Tensor { bytes, shape, .. } => {
                        let offset = Self::tensor_offset(shape, &indices)?;
                        let start = offset.checked_mul(4).ok_or_else(|| Error::Runtime("tensor offset is too large".into()))?;
                        let raw: [u8; 4] = bytes.get(start..start + 4).ok_or_else(|| Error::Runtime("tensor offset out of bounds".into()))?.try_into().expect("exact f32 cell");
                        Value::F32(f32::from_bits(u32::from_le_bytes(raw)))
                    },
                    _ => return Err(Error::Runtime("tensor heap invariant broken".into())),
                };
                self.push(value);
            },
            DecodedOp::TableIndex => { let key_value = self.pop()?; let key = table_key_from_value(self.heap_ref(), &key_value)?; let object = self.pop()?; let Value::Table(reference, _) = object else { return Err(Error::Runtime("VM table invariant broken".into())); }; let value = match self.heap_ref().get(reference)? { HeapObject::Table { entries, .. } => entries.get(&key).cloned().ok_or_else(|| Error::Runtime(format!("table has no key {}", table_key_display(&key))))?, _ => return Err(Error::Runtime("table heap invariant broken".into())) }; self.push(value); },
            DecodedOp::TableKeys => { let object = self.pop()?; let Value::Table(reference, _) = object else { return Err(Error::Runtime("VM table invariant broken".into())); }; let mut keys = match self.heap_ref().get(reference)? { HeapObject::Table { entries, .. } => entries.keys().cloned().collect::<Vec<_>>(), _ => return Err(Error::Runtime("table heap invariant broken".into())) }; keys.sort_by(|left, right| match (left, right) { (TableKey::Index(a), TableKey::Index(b)) => a.cmp(b), (TableKey::Name(a), TableKey::Name(b)) => a.cmp(b), (TableKey::Index(_), TableKey::Name(_)) => std::cmp::Ordering::Less, (TableKey::Name(_), TableKey::Index(_)) => std::cmp::Ordering::Greater, }); let reference = self.allocate(HeapObject::TableKeys(keys)); self.push(Value::TableKeys(reference)); self.collect_if_needed(); },
            DecodedOp::TableKeysIndex => { let index = integer_to_usize(&self.pop()?)?; let object = self.pop()?; let Value::TableKeys(reference) = object else { return Err(Error::Runtime("VM table_keys invariant broken".into())); }; let key = match self.heap_ref().get(reference)? { HeapObject::TableKeys(keys) => keys.get(index).cloned().ok_or_else(|| Error::Runtime(format!("table key index {index} is out of bounds")))?, _ => return Err(Error::Runtime("table_keys heap invariant broken".into())) }; self.push(Value::TableKey(Rc::new(key))); },
            DecodedOp::TableRemove => { let key_value = self.pop()?; let key = table_key_from_value(self.heap_ref(), &key_value)?; let object = self.pop()?; let Value::Table(reference, _) = object else { return Err(Error::Runtime("VM table invariant broken".into())); }; let removed = match self.heap_mut().get_mut(reference)? { HeapObject::Table { entries, .. } => entries.remove(&key).is_some(), _ => return Err(Error::Runtime("table heap invariant broken".into())) }; self.push(Value::Bool(removed)); },
            DecodedOp::Field(field) => { let object = self.pop()?; let Value::Struct(reference, _) = object else { return Err(Error::Runtime("VM struct invariant broken".into())); }; let value = match self.heap_ref().get(reference)? { HeapObject::Struct { values, .. } => values.get(field.index).cloned().ok_or_else(|| Error::Runtime("invalid struct field index".into()))?, _ => return Err(Error::Runtime("struct heap invariant broken".into())) }; self.push(value); },
            DecodedOp::TableField(name) => { let object = self.pop()?; let Value::Table(reference, _) = object else { return Err(Error::Runtime("VM table invariant broken".into())); }; let value = match self.heap_ref().get(reference)? { HeapObject::Table { entries, .. } => entries.get(&TableKey::Name(name.clone())).cloned().ok_or_else(|| Error::Runtime(format!("table has no key {name}")))?, _ => return Err(Error::Runtime("table heap invariant broken".into())) }; self.push(value); },
            DecodedOp::ModuleField(name) => { let Value::Module(id) = self.pop()? else { return Err(Error::Runtime("VM module invariant broken".into())); }; let value = { let instance = self.modules.get(&id).ok_or_else(|| Error::Runtime("loaded module is missing".into()))?; let ModuleExport::Value { slot, .. } = instance.artifact.exports.get(name.as_ref()).ok_or_else(|| Error::Runtime(format!("module has no exported value '{name}'")))? else { return Err(Error::Runtime(format!("'{name}' is not an exported module value"))); }; instance.vm.locals.get(*slot).cloned().ok_or_else(|| Error::Runtime("invalid module export slot".into()))? }; self.push(value); },
            DecodedOp::Binary(op) => {
                let right = self.pop_compiled();
                let left = self.pop_compiled();
                let result = {
                    let heap = self.heap_ref();
                    evaluate_binary(&heap, left, right, op)?
                };
                self.push(result);
            },
            DecodedOp::Unary(op, ty) => { let val = self.pop()?; self.push(evaluate_unary(val, op, ty.as_ref())?); },
            DecodedOp::Cast(target_ty) => {
                let val = self.pop()?;
                self.push(cast_numeric(val, target_ty.as_ref())?);
            },
            DecodedOp::Len => {
                let value = self.pop()?;
                let length = match value {
                    Value::Array(reference, element) => match self.heap_ref().get(reference)? {
                        HeapObject::Array { bytes, element: stored_element } if stored_element == element.as_ref() => bytes.len() / scalar_size(&element)?,
                        _ => return Err(Error::Runtime("array heap invariant broken".into())),
                    },
                    Value::String(reference) => match self.heap_ref().get(reference)? {
                        HeapObject::String(text) => text.chars().count(),
                        _ => return Err(Error::Runtime("string heap invariant broken".into())),
                    },
                    Value::Table(reference, _) => match self.heap_ref().get(reference)? {
                        HeapObject::Table { entries, .. } => entries.len(),
                        _ => return Err(Error::Runtime("table heap invariant broken".into())),
                    },
                    Value::TableKeys(reference) => match self.heap_ref().get(reference)? {
                        HeapObject::TableKeys(keys) => keys.len(),
                        _ => return Err(Error::Runtime("table_keys heap invariant broken".into())),
                    },
                    Value::Tensor(reference, _) => match self.heap_ref().get(reference)? {
                        HeapObject::Tensor { shape, .. } => shape.iter().try_fold(1usize, |total, dimension| total.checked_mul(*dimension)).ok_or_else(|| Error::Runtime("tensor is too large".into()))?,
                        _ => return Err(Error::Runtime("tensor heap invariant broken".into())),
                    },
                    _ => return Err(Error::Runtime("VM len invariant broken".into())),
                };
                let length = i32::try_from(length).map_err(|_| Error::Runtime("vector length exceeds i32".into()))?;
                self.push(Value::I32(length));
            },
            DecodedOp::ConcatString => {
                let right = self.pop()?;
                let left = self.pop()?;
                let Value::String(left_ref) = left else { return Err(Error::Runtime("VM string invariant broken".into())); };
                let Value::String(right_ref) = right else { return Err(Error::Runtime("VM string invariant broken".into())); };

                let combined = {
                    let heap = self.heap_ref();
                    match (heap.get(left_ref)?, heap.get(right_ref)?) {
                        (HeapObject::String(l), HeapObject::String(r)) => {
                            let capacity = l.len().checked_add(r.len()).ok_or_else(|| Error::Runtime("string concatenation is too large".into()))?;
                            let mut text = String::with_capacity(capacity);
                            text.push_str(l);
                            text.push_str(r);
                            text
                        },
                        _ => return Err(Error::Runtime("string heap invariant broken".into())),
                    }
                };
                let reference = self.allocate(HeapObject::String(combined));
                self.push(Value::String(reference));
                self.collect_if_needed();
            },
            DecodedOp::Builtin1(name, _ty) => { let arg = self.pop()?; self.push(evaluate_builtin1(name, arg)?); },
            DecodedOp::Builtin2(name, ty) => { let arg2 = self.pop_compiled(); let arg1 = self.pop_compiled(); self.push(evaluate_builtin2(name, arg1, arg2, ty.as_ref())?); },
            DecodedOp::CallExternal(name, argument_count) => self.call_external(name.as_ref(), argument_count)?,
            DecodedOp::JumpIfFalse(target) => { match self.pop()? { Value::Bool(false) => { pc = target; continue; }, Value::Bool(true) => {}, _ => return Err(Error::Runtime("VM condition invariant broken".into())), } },
            DecodedOp::Jump(target) => { pc = target; continue; },
            DecodedOp::JumpIfFalseKeep(target) => {
                if self.stack_ptr == 0 { return Err(Error::Runtime("stack underflow".into())); }
                let val = &self.stack[self.stack_ptr - 1];
                if let Value::Bool(false) = val {
                    pc = target; continue;
                } else if let Value::Bool(true) = val {
                    self.pop()?;
                } else {
                    return Err(Error::Runtime("VM condition invariant broken".into()));
                }
            },
            DecodedOp::JumpIfTrueKeep(target) => {
                if self.stack_ptr == 0 { return Err(Error::Runtime("stack underflow".into())); }
                let val = &self.stack[self.stack_ptr - 1];
                if let Value::Bool(true) = val {
                    pc = target; continue;
                } else if let Value::Bool(false) = val {
                    self.pop()?;
                } else {
                    return Err(Error::Runtime("VM condition invariant broken".into()));
                }
            },
            DecodedOp::CallMethod(receiver, target) => { call_stack.push((next_pc, current_receiver)); current_receiver = Some(receiver); pc = target; continue; },
            DecodedOp::CallCurrentMethod(target) => { let receiver = current_receiver.ok_or_else(|| Error::Runtime("this is available only inside a method".into()))?; call_stack.push((next_pc, current_receiver)); current_receiver = Some(receiver); pc = target; continue; },
            DecodedOp::CallModule(slot, name) => { self.call_module_function(slot, name.as_ref())?; },
            DecodedOp::TrackDestructor(slot, target) => {
                if !matches!(self.locals.get(slot), Some(Value::Struct(_, _))) {
                    return Err(Error::Runtime("VM destructor slot invariant broken".into()));
                }
                self.active_destructors.push(ActiveDestructor { slot, target });
            },
            DecodedOp::UntrackDestructor(slot) => {
                let position = self.active_destructors.iter().rposition(|destructor| destructor.slot == slot)
                    .ok_or_else(|| Error::Runtime("VM destructor tracking invariant broken".into()))?;
                self.active_destructors.remove(position);
            },
            DecodedOp::Return => { if let Some((return_pc, previous_receiver)) = call_stack.pop() { current_receiver = previous_receiver; pc = return_pc; continue; } if terminal_return { return Ok(&self.output); } return Err(Error::Runtime("return outside method".into())); },
            DecodedOp::Print => { let value = self.pop()?; let text = self.format_value(&value)?; self.emit(text); },
            DecodedOp::Printf(num_args) => {
                let format_val = self.pop()?;
                let mut arg_vals = Vec::with_capacity(num_args);
                for _ in 0..num_args { arg_vals.push(self.pop()?); }
                arg_vals.reverse();
                let format_str = if let Value::String(reference) = format_val { match self.heap_ref().get(reference)? { HeapObject::String(text) => text.clone(), _ => return Err(Error::Runtime("string heap invariant broken".into())) } } else { return Err(Error::Runtime("printf format is not a string".into())); };

                let mut result = String::new();
                let mut arg_idx = 0;
                let mut chars = format_str.chars().peekable();

                while let Some(c) = chars.next() {
                    match c {
                        '{' => match chars.next() {
                            Some('}') => {
                                let value = arg_vals.get(arg_idx).ok_or_else(|| Error::Runtime("printf has more placeholders than arguments".into()))?;
                                result.push_str(&self.format_value(value)?);
                                arg_idx += 1;
                            }
                            Some('{') => result.push('{'),
                            _ => return Err(Error::Runtime("printf has an invalid '{' escape; use '{{' or '{}'".into())),
                        },
                        '}' => match chars.next() {
                            Some('}') => result.push('}'),
                            _ => return Err(Error::Runtime("printf has an invalid '}' escape; use '}}'".into())),
                        },
                        _ => result.push(c),
                    }
                }
                if arg_idx != arg_vals.len() { return Err(Error::Runtime("printf has more arguments than placeholders".into())); }
                if self.interactive { print!("{}", result); let _ = io::stdout().flush(); }
                self.output.push(result);
            },
            DecodedOp::Putc => {
                let value = self.pop()?;
                let codepoint = u32::try_from(integer_to_usize(&value)?)
                    .map_err(|_| Error::Runtime("putc code point exceeds u32".into()))?;
                let c = char::from_u32(codepoint)
                    .ok_or_else(|| Error::Runtime(format!("putc requires a valid Unicode scalar value, got {codepoint}")))?;
                if self.interactive { print!("{c}"); let _ = std::io::stdout().flush(); }
            },
        } pc = next_pc; } Ok(&self.output)
    }

    fn load_module(&mut self, artifact: crate::compiler::ModuleArtifact) -> Result<(), Error> {
        let id = artifact.id.clone();

        if let Some(ext) = self.native_modules.remove(&id) {
            ext.register(self)?;
            self.push(Value::Module(id));
            return Ok(());
        }

        if !self.modules.contains_key(&id) {
            let mut vm = Vm::with_shared_heap(self.heap.clone(), self.extern_functions.clone(), self.callback_state);
            vm.run(&artifact.code)?;
            let output = std::mem::take(&mut vm.output);
            self.modules.insert(id.clone(), ModuleInstance { artifact, vm });
            for line in output { self.emit(line); }
        }
        self.push(Value::Module(id));
        Ok(())
    }

    fn call_module_function(&mut self, slot: usize, name: &str) -> Result<(), Error> {
        let Value::Module(id) = self.locals.get(slot).cloned().ok_or_else(|| Error::Runtime("invalid module slot".into()))? else { return Err(Error::Runtime("VM module slot invariant broken".into())); };
        let output = {
            let instance = self.modules.get_mut(&id).ok_or_else(|| Error::Runtime("loaded module is missing".into()))?;
            let entry = match instance.artifact.exports.get(name).ok_or_else(|| Error::Runtime(format!("module has no exported function '{name}'")))? {
                ModuleExport::Function { entry } => *entry,
                _ => return Err(Error::Runtime(format!("'{name}' is not an exported module function"))),
            };
            let code = Rc::clone(&instance.artifact.code);
            instance.vm.run_from(&code, entry, true)?;
            std::mem::take(&mut instance.vm.output)
        };
        for line in output { self.emit(line); } Ok(())
    }

    #[inline(always)]
    pub fn push(&mut self, value: Value) {
        if self.stack_ptr >= self.stack.len() { self.stack.resize((self.stack_ptr + 1) * 2, Value::Bool(false)); }
        unsafe { self.push_unchecked(value) }
    }

    #[inline(always)]
    pub unsafe fn push_unchecked(&mut self, value: Value) {
        *self.stack.get_unchecked_mut(self.stack_ptr) = value;
        self.stack_ptr += 1;
    }

    #[inline(always)]
    pub unsafe fn pop_unchecked(&mut self) -> Value {
        self.stack_ptr -= 1;
        self.stack.get_unchecked(self.stack_ptr).clone()
    }

    #[inline(always)]
    pub fn pop_compiled(&mut self) -> Value {
        debug_assert!(self.stack_ptr > 0, "compiler emitted a stack underflow");
        unsafe { self.pop_unchecked() }
    }

    pub fn pop(&mut self) -> Result<Value, Error> {
        if self.stack_ptr == 0 { return Err(Error::Runtime("stack underflow".into())); }
        Ok(unsafe { self.pop_unchecked() })
    }

    pub fn emit(&mut self, s: String) {
        if self.interactive {
            println!("{s}");
            let _ = io::stdout().flush();
        }
        self.output.push(s);
    }

    pub fn read_input(&mut self, ty: &Type) -> Result<Value, Error> {
        let line = match self.input.pop_front() {
            Some(line) => line,
            None => {
                let mut line = String::new();
                if io::stdin().read_line(&mut line).map_err(|error| Error::Runtime(format!("failed to read input: {error}")))? == 0 { return Err(Error::Runtime("input ended before a value was read".into())); }
                line
            }
        };
        let text = line.trim();
        if *ty == Type::String {
            let text = line.trim_end_matches(&['\r', '\n'][..]).to_owned();
            Ok(Value::String(self.allocate(HeapObject::String(text))))
        } else if is_integer(ty) {
            let n: i128 = text.parse().map_err(|_| Error::Runtime(format!("invalid integer: {text}")))?;
            int_value(n, ty)
        } else if is_numeric(ty) {
            let n: f64 = text.parse().map_err(|_| Error::Runtime(format!("invalid float: {text}")))?;
            Ok(float_value(n, ty))
        } else {
            Err(Error::Runtime("VM input type invariant broken".into()))
        }
    }
}

pub fn integer_to_usize(v: &Value) -> Result<usize, Error> { match v { Value::I8(x) if *x>=0 => Ok(*x as usize),Value::I16(x) if *x>=0 => Ok(*x as usize),Value::I32(x) if *x>=0 => Ok(*x as usize),Value::I64(x) if *x>=0 => Ok(*x as usize),Value::U8(x)=>Ok(*x as usize),Value::U16(x)=>Ok(*x as usize),Value::U32(x)=>Ok(*x as usize),Value::U64(x)=>usize::try_from(*x).map_err(|_|Error::Runtime("array index too large".into())),_=>Err(Error::Runtime("array index must be non-negative integer".into())) } }

pub fn evaluate_builtin1(builtin: BuiltinFn, arg: Value) -> Result<Value, Error> {
    if builtin == BuiltinFn::Abs {
        match arg {
            Value::I8(v) => return v.checked_abs().map(Value::I8).ok_or_else(|| Error::Runtime("abs overflow".into())),
            Value::I16(v) => return v.checked_abs().map(Value::I16).ok_or_else(|| Error::Runtime("abs overflow".into())),
            Value::I32(v) => return v.checked_abs().map(Value::I32).ok_or_else(|| Error::Runtime("abs overflow".into())),
            Value::I64(v) => return v.checked_abs().map(Value::I64).ok_or_else(|| Error::Runtime("abs overflow".into())),
            Value::U8(_) | Value::U16(_) | Value::U32(_) | Value::U64(_) => return Ok(arg),
            Value::F16(v) => return Ok(Value::F16(f32_to_f16(f16_to_f32(v).abs()))),
            Value::BF16(v) => return Ok(Value::BF16(f32_to_bf16(bf16_to_f32(v).abs()))),
            Value::F32(v) => return Ok(Value::F32(v.abs())),
            Value::F64(v) => return Ok(Value::F64(v.abs())),
            _ => return Err(Error::Runtime("invalid type for abs".into())),
        }
    }
    macro_rules! float_math1 {
        ($v:ident, $typecast:ty, $variant:ident, $is_f16:expr) => {
            {
                let val = if $is_f16 { f16_to_f32(*$v as u16) as $typecast } else { *$v as $typecast };
                let res = match builtin {
                    BuiltinFn::Sqrt => val.sqrt(), BuiltinFn::Sin => val.sin(), BuiltinFn::Cos => val.cos(),
                    BuiltinFn::Tan => val.tan(), BuiltinFn::Asin => val.asin(), BuiltinFn::Acos => val.acos(),
                    BuiltinFn::Atan => val.atan(), BuiltinFn::Floor => val.floor(), BuiltinFn::Ceil => val.ceil(),
                    BuiltinFn::Round => val.round(),
                    _ => return Err(Error::Runtime("invalid unary built-in".into())),
                };
                if $is_f16 { Ok(Value::F16(f32_to_f16(res as f32))) } else { Ok(Value::$variant(res as _)) }
            }
        }
    }
    match &arg {
        Value::F32(v) => float_math1!(v, f32, F32, false),
        Value::F64(v) => float_math1!(v, f64, F64, false),
        Value::F16(v) => float_math1!(v, f32, F16, true),
        Value::BF16(v) => {
            let val = bf16_to_f32(*v);
            let res = match builtin {
                BuiltinFn::Sqrt => val.sqrt(), BuiltinFn::Sin => val.sin(), BuiltinFn::Cos => val.cos(),
                BuiltinFn::Tan => val.tan(), BuiltinFn::Asin => val.asin(), BuiltinFn::Acos => val.acos(),
                BuiltinFn::Atan => val.atan(), BuiltinFn::Floor => val.floor(), BuiltinFn::Ceil => val.ceil(),
                BuiltinFn::Round => val.round(),
                _ => return Err(Error::Runtime("invalid unary built-in".into())),
            };
            Ok(Value::BF16(f32_to_bf16(res)))
        },
        _ => Err(Error::Runtime("built-in requires a float".into()))
    }
}

pub fn evaluate_builtin2(builtin: BuiltinFn, a: Value, b: Value, ty: &Type) -> Result<Value, Error> {
    if matches!(builtin, BuiltinFn::Min | BuiltinFn::Max) {
        macro_rules! min_max {
            ($x:ident, $l:ident, $r:ident) => {
                if builtin == BuiltinFn::Min { Ok(Value::$x(if $l < $r { *$l } else { *$r })) }
                else { Ok(Value::$x(if $l > $r { *$l } else { *$r })) }
            }
        }
        match (&a, &b) {
            (Value::I8(l), Value::I8(r)) => return min_max!(I8, l, r),
            (Value::I16(l), Value::I16(r)) => return min_max!(I16, l, r),
            (Value::I32(l), Value::I32(r)) => return min_max!(I32, l, r),
            (Value::I64(l), Value::I64(r)) => return min_max!(I64, l, r),
            (Value::U8(l), Value::U8(r)) => return min_max!(U8, l, r),
            (Value::U16(l), Value::U16(r)) => return min_max!(U16, l, r),
            (Value::U32(l), Value::U32(r)) => return min_max!(U32, l, r),
            (Value::U64(l), Value::U64(r)) => return min_max!(U64, l, r),
            (Value::F32(l), Value::F32(r)) => return min_max!(F32, l, r),
            (Value::F64(l), Value::F64(r)) => return min_max!(F64, l, r),
            (Value::F16(l_raw), Value::F16(r_raw)) => {
                let l = f16_to_f32(*l_raw); let r = f16_to_f32(*r_raw);
                if builtin == BuiltinFn::Min { return Ok(Value::F16(f32_to_f16(if l < r { l } else { r }))); }
                else { return Ok(Value::F16(f32_to_f16(if l > r { l } else { r }))); }
            },
            (Value::BF16(l_raw), Value::BF16(r_raw)) => {
                let l = bf16_to_f32(*l_raw); let r = bf16_to_f32(*r_raw);
                if builtin == BuiltinFn::Min { return Ok(Value::BF16(f32_to_bf16(if l < r { l } else { r }))); }
                else { return Ok(Value::BF16(f32_to_bf16(if l > r { l } else { r }))); }
            },
            _ => return Err(Error::Runtime("invalid type for min/max".into()))
        }
    }
    if builtin == BuiltinFn::Pow && is_integer(ty) {
        macro_rules! int_pow {
            ($x:ident) => {
                if let (Value::$x(l), Value::$x(r)) = (&a, &b) {
                    let exp = u32::try_from(*r).map_err(|_| Error::Runtime("pow exponent out of bounds for int".into()))?;
                    return l.checked_pow(exp).map(Value::$x).ok_or_else(|| Error::Runtime("pow overflow".into()));
                }
            }
        }
        match ty {
            Type::I8 => { int_pow!(I8); }, Type::I16 => { int_pow!(I16); },
            Type::I32 => { int_pow!(I32); }, Type::I64 => { int_pow!(I64); },
            Type::U8 => { int_pow!(U8); }, Type::U16 => { int_pow!(U16); },
            Type::U32 => { int_pow!(U32); }, Type::U64 => { int_pow!(U64); },
            _ => {}
        }
    }
    macro_rules! float_math2 {
        ($l:ident, $r:ident, $typecast:ty, $variant:ident, $is_f16:expr) => {
            {
                let l = if $is_f16 { f16_to_f32(*$l as u16) as $typecast } else { *$l as $typecast };
                let r = if $is_f16 { f16_to_f32(*$r as u16) as $typecast } else { *$r as $typecast };
                let res = match builtin {
                    BuiltinFn::Pow => l.powf(r), BuiltinFn::Atan2 => l.atan2(r),
                    _ => return Err(Error::Runtime("invalid binary built-in".into())),
                };
                if $is_f16 { Ok(Value::F16(f32_to_f16(res as f32))) } else { Ok(Value::$variant(res as _)) }
            }
        }
    }
    match (&a, &b) {
        (Value::F32(l), Value::F32(r)) => float_math2!(l, r, f32, F32, false),
        (Value::F64(l), Value::F64(r)) => float_math2!(l, r, f64, F64, false),
        (Value::F16(l), Value::F16(r)) => float_math2!(l, r, f32, F16, true),
        (Value::BF16(l), Value::BF16(r)) => {
            let l = bf16_to_f32(*l);
            let r = bf16_to_f32(*r);
            let res = match builtin {
                BuiltinFn::Pow => l.powf(r), BuiltinFn::Atan2 => l.atan2(r),
                _ => return Err(Error::Runtime("invalid binary built-in".into())),
            };
            Ok(Value::BF16(f32_to_bf16(res)))
        },
        _ => Err(Error::Runtime("built-in requires matching floats".into()))
    }
}

pub fn evaluate_binary(heap: &Heap, a: Value, b: Value, opcode: &BinaryOp) -> Result<Value, Error> {
    if matches!(opcode, BinaryOp::Equal | BinaryOp::NotEqual) {
        let equal = match (&a, &b) {
            (Value::String(left), Value::String(right)) => match (heap.get(*left)?, heap.get(*right)?) {
                (HeapObject::String(left), HeapObject::String(right)) => left == right,
                _ => return Err(Error::Runtime("string heap invariant broken".into())),
            },
            _ => a == b,
        };
        return Ok(Value::Bool(if matches!(opcode, BinaryOp::Equal) { equal } else { !equal }));
    }
    macro_rules! int_op {
        ($x:ident, $op:expr) => {
            if let (Value::$x(l), Value::$x(r)) = (a, b) {
                match $op {
                    BinOp::Add => return l.checked_add(r).map(Value::$x).ok_or_else(|| Error::Runtime("addition overflow".into())),
                    BinOp::Sub => return l.checked_sub(r).map(Value::$x).ok_or_else(|| Error::Runtime("subtraction overflow".into())),
                    BinOp::Mul => return l.checked_mul(r).map(Value::$x).ok_or_else(|| Error::Runtime("multiplication overflow".into())),
                    BinOp::Div => return l.checked_div(r).map(Value::$x).ok_or_else(|| Error::Runtime("division by zero".into())),
                    BinOp::Mod => return l.checked_rem(r).map(Value::$x).ok_or_else(|| Error::Runtime("modulo by zero".into())),
                    BinOp::BitAnd => return Ok(Value::$x(l & r)),
                    BinOp::BitOr  => return Ok(Value::$x(l | r)),
                    BinOp::BitXor => return Ok(Value::$x(l ^ r)),
                    BinOp::Shl    => return l.checked_shl(r as u32).map(Value::$x).ok_or_else(|| Error::Runtime("shift overflow".into())),
                    BinOp::Shr    => return l.checked_shr(r as u32).map(Value::$x).ok_or_else(|| Error::Runtime("shift overflow".into())),
                    BinOp::Lt     => return Ok(Value::Bool(l < r)),
                    BinOp::Le     => return Ok(Value::Bool(l <= r)),
                    BinOp::Gt     => return Ok(Value::Bool(l > r)),
                    BinOp::Ge     => return Ok(Value::Bool(l >= r)),
                    _ => return Err(Error::Runtime("VM operator invariant broken".into()))
                }
            }
        }
    }
    macro_rules! float_op {
        ($x:ident, $typecast:ty, $op:expr) => {
            if let (Value::$x(l_raw), Value::$x(r_raw)) = (a, b) {
                let l = l_raw as $typecast;
                let r = r_raw as $typecast;
                match $op {
                    BinOp::Add => return Ok(Value::$x((l + r) as _)),
                    BinOp::Sub => return Ok(Value::$x((l - r) as _)),
                    BinOp::Mul => return Ok(Value::$x((l * r) as _)),
                    BinOp::Div => return Ok(Value::$x((l / r) as _)),
                    BinOp::Mod => return Ok(Value::$x((l % r) as _)),
                    BinOp::Lt  => return Ok(Value::Bool(l < r)),
                    BinOp::Le  => return Ok(Value::Bool(l <= r)),
                    BinOp::Gt  => return Ok(Value::Bool(l > r)),
                    BinOp::Ge  => return Ok(Value::Bool(l >= r)),
                    _ => return Err(Error::Runtime("VM float operator invariant broken".into()))
                }
            }
        }
    }
    match opcode {
        BinaryOp::I8(op) => { int_op!(I8, op); }, BinaryOp::I16(op) => { int_op!(I16, op); },
        BinaryOp::I32(op) => { int_op!(I32, op); }, BinaryOp::I64(op) => { int_op!(I64, op); },
        BinaryOp::U8(op) => { int_op!(U8, op); }, BinaryOp::U16(op) => { int_op!(U16, op); },
        BinaryOp::U32(op) => { int_op!(U32, op); }, BinaryOp::U64(op) => { int_op!(U64, op); },
        BinaryOp::F32(op) => { float_op!(F32, f32, op); },
        BinaryOp::F64(op) => { float_op!(F64, f64, op); },
        BinaryOp::F16(op) => {
            if let (Value::F16(l_raw), Value::F16(r_raw)) = (a, b) {
                let l = f16_to_f32(l_raw);
                let r = f16_to_f32(r_raw);
                match op {
                    BinOp::Add => return Ok(Value::F16(f32_to_f16(l + r))),
                    BinOp::Sub => return Ok(Value::F16(f32_to_f16(l - r))),
                    BinOp::Mul => return Ok(Value::F16(f32_to_f16(l * r))),
                    BinOp::Div => return Ok(Value::F16(f32_to_f16(l / r))),
                    BinOp::Mod => return Ok(Value::F16(f32_to_f16(l % r))),
                    BinOp::Lt  => return Ok(Value::Bool(l < r)),
                    BinOp::Le  => return Ok(Value::Bool(l <= r)),
                    BinOp::Gt  => return Ok(Value::Bool(l > r)),
                    BinOp::Ge  => return Ok(Value::Bool(l >= r)),
                    _ => return Err(Error::Runtime("VM float operator invariant broken".into()))
                }
            }
        },
        BinaryOp::BF16(op) => {
            if let (Value::BF16(l_raw), Value::BF16(r_raw)) = (a, b) {
                let l = bf16_to_f32(l_raw);
                let r = bf16_to_f32(r_raw);
                match op {
                    BinOp::Add => return Ok(Value::BF16(f32_to_bf16(l + r))),
                    BinOp::Sub => return Ok(Value::BF16(f32_to_bf16(l - r))),
                    BinOp::Mul => return Ok(Value::BF16(f32_to_bf16(l * r))),
                    BinOp::Div => return Ok(Value::BF16(f32_to_bf16(l / r))),
                    BinOp::Mod => return Ok(Value::BF16(f32_to_bf16(l % r))),
                    BinOp::Lt  => return Ok(Value::Bool(l < r)),
                    BinOp::Le  => return Ok(Value::Bool(l <= r)),
                    BinOp::Gt  => return Ok(Value::Bool(l > r)),
                    BinOp::Ge  => return Ok(Value::Bool(l >= r)),
                    _ => return Err(Error::Runtime("VM float operator invariant broken".into()))
                }
            }
        },
        BinaryOp::Equal | BinaryOp::NotEqual => unreachable!(),
    }
    Err(Error::Runtime("VM execution invariant broken: unsupported binary op".into()))
}

pub fn evaluate_unary(a: Value, op: &UnOp, ty: &Type) -> Result<Value, Error> {
    match op {
        UnOp::Neg => {
            match ty {
                Type::I8 => if let Value::I8(v) = a { return v.checked_neg().map(Value::I8).ok_or_else(|| Error::Runtime("negation overflow".into())); },
                Type::I16 => if let Value::I16(v) = a { return v.checked_neg().map(Value::I16).ok_or_else(|| Error::Runtime("negation overflow".into())); },
                Type::I32 => if let Value::I32(v) = a { return v.checked_neg().map(Value::I32).ok_or_else(|| Error::Runtime("negation overflow".into())); },
                Type::I64 => if let Value::I64(v) = a { return v.checked_neg().map(Value::I64).ok_or_else(|| Error::Runtime("negation overflow".into())); },
                Type::U8 | Type::U16 | Type::U32 | Type::U64 => return Err(Error::Runtime("cannot negate unsigned integer".into())),
                Type::F32 => if let Value::F32(v) = a { return Ok(Value::F32(-v)); },
                Type::F64 => if let Value::F64(v) = a { return Ok(Value::F64(-v)); },
                Type::F16 => if let Value::F16(v) = a { return Ok(Value::F16(f32_to_f16(-f16_to_f32(v)))); },
                Type::BF16 => if let Value::BF16(v) = a { return Ok(Value::BF16(f32_to_bf16(-bf16_to_f32(v)))); },
                _ => {}
            }
        },
        UnOp::Not => {
            if let Value::Bool(v) = a { return Ok(Value::Bool(!v)); }
        }
    }
    Err(Error::Runtime("VM execution invariant broken: unsupported unary op".into()))
}

pub fn cast_numeric(val: Value, target_ty: &Type) -> Result<Value, Error> {
    macro_rules! cast_macro {
        ($v:ident) => {
            match target_ty {
                Type::I8 => Value::I8(*$v as i8),
                Type::I16 => Value::I16(*$v as i16),
                Type::I32 => Value::I32(*$v as i32),
                Type::I64 => Value::I64(*$v as i64),
                Type::U8 => Value::U8(*$v as u8),
                Type::U16 => Value::U16(*$v as u16),
                Type::U32 => Value::U32(*$v as u32),
                Type::U64 => Value::U64(*$v as u64),
                Type::F32 => Value::F32(*$v as f32),
                Type::F64 => Value::F64(*$v as f64),
                Type::F16 => Value::F16(f32_to_f16(*$v as f32)),
                Type::BF16 => Value::BF16(f32_to_bf16(*$v as f32)),
                _ => return Err(Error::Runtime("invalid cast target".into())),
            }
        }
    }
    Ok(match val {
        Value::I8(ref v) => cast_macro!(v),
        Value::I16(ref v) => cast_macro!(v),
        Value::I32(ref v) => cast_macro!(v),
        Value::I64(ref v) => cast_macro!(v),
        Value::U8(ref v) => cast_macro!(v),
        Value::U16(ref v) => cast_macro!(v),
        Value::U32(ref v) => cast_macro!(v),
        Value::U64(ref v) => cast_macro!(v),
        Value::F32(ref v) => cast_macro!(v),
        Value::F64(ref v) => cast_macro!(v),
        Value::F16(v_raw) => { let val = f16_to_f32(v_raw); let v = &val; cast_macro!(v) },
        Value::BF16(v_raw) => { let val = bf16_to_f32(v_raw); let v = &val; cast_macro!(v) },
        _ => return Err(Error::Runtime("invalid cast source".into())),
    })
}

pub fn execute(source: &str) -> Result<Vec<String>, Error> {
    let (program, strings) = Parser::new(lex(source)?).into_program()?;
    let code = Compiler::default().with_strings(strings).compile(program)?;
    let mut vm = Vm::default();
    Ok(vm.run(&code)?.to_vec())
}

pub fn execute_with_input<I, S>(source: &str, input: I) -> Result<Vec<String>, Error>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let (program, strings) = Parser::new(lex(source)?).into_program()?;
    let code = Compiler::default().with_strings(strings).compile(program)?;
    let mut vm = Vm::default();
    vm.input.extend(input.into_iter().map(Into::into));
    Ok(vm.run(&code)?.to_vec())
}

pub fn execute_interactive(source: &str) -> Result<(), Error> {
    let (program, strings) = Parser::new(lex(source)?).into_program()?;
    let code = Compiler::default().with_strings(strings).compile(program)?;
    let mut vm = Vm { interactive: true, ..Vm::default() };
    vm.run(&code)?;
    Ok(())
}
pub fn execute_file(path: impl AsRef<Path>) -> Result<Vec<String>, Error> {
    Vm::default().execute_file(path)
}
pub fn execute_interactive_file(path: impl AsRef<Path>) -> Result<(), Error> {
    let path = fs::canonicalize(path.as_ref()).map_err(|error| Error::Runtime(format!("cannot open source file: {error}")))?;
    let root = path.parent().ok_or_else(|| Error::Runtime("source file has no parent directory".into()))?.to_path_buf();
    let source = fs::read_to_string(&path).map_err(|error| Error::Runtime(format!("cannot read source file: {error}")))?;
    let (program, strings) = Parser::new(lex(&source)?).into_program()?;
    let code = Compiler::with_module_root(root).with_strings(strings).compile(program)?;
    let mut vm = Vm { interactive: true, ..Vm::default() };
    vm.run(&code)?;
    Ok(())
}
