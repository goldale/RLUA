//! L0: experimental typed Lua-like language with a stack bytecode VM.
//! The public FFI boundary is C ABI, so it is callable both from C and Rust.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{CStr, CString};
use std::fs;
use std::fmt;
use std::io::{self, Write};
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use half::f16;

pub const ABI_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringId(pub u32);

#[derive(Clone, Debug, Default)]
pub struct StringInterner {
    strings: Vec<Rc<str>>,
    lookup: std::collections::HashMap<Rc<str>, StringId>,
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
    I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, Bool, String,
    Array(Box<Type>), Tensor(Box<Type>, usize), Table(Box<Type>), TableKey, TableKeys, Struct(String), Module(String),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I8 => write!(f, "i8"), Self::I16 => write!(f, "i16"),
            Self::I32 => write!(f, "i32"), Self::I64 => write!(f, "i64"),
            Self::U8 => write!(f, "u8"), Self::U16 => write!(f, "u16"),
            Self::U32 => write!(f, "u32"), Self::U64 => write!(f, "u64"),
            Self::F16 => write!(f, "f16"), Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"), Self::Bool => write!(f, "bool"),
            Self::String => write!(f, "string"),
            Self::Array(inner) => write!(f, "vector<{}>", inner),
            Self::Tensor(inner, rank) => write!(f, "tensor<{}, {}>", inner, rank),
            Self::Table(inner) => write!(f, "table<{}>", inner),
            Self::TableKey => write!(f, "table_key"),
            Self::TableKeys => write!(f, "table_keys"),
            Self::Struct(name) => write!(f, "{name}"),
            Self::Module(_) => write!(f, "module"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeapRef(usize);

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    I8(i8), I16(i16), I32(i32), I64(i64), U8(u8), U16(u16), U32(u32), U64(u64),
    /// IEEE-754 binary16 bits. Arithmetic widens to f32 and rounds back to f16.
    F16(u16), F32(f32), F64(f64), Bool(bool), String(HeapRef),
    /// A handle to homogeneous scalar values packed in canonical little-endian form.
    Array(HeapRef, Box<Type>),
    /// An opaque handle to CPU tensor storage.  The rank is static; dimensions live
    /// in the heap object so future hardware backends can replace the storage.
    Tensor(HeapRef, Box<Type>, usize),
    Table(HeapRef, Box<Type>),
    /// `TableKey::Index(i128)` has 16-byte alignment; boxing it prevents that
    /// uncommon key representation from raising the alignment of every stack
    /// value.
    TableKey(Rc<TableKey>),
    TableKeys(HeapRef),
    /// Layouts are immutable compiler metadata.  Keeping a shared reference
    /// here avoids embedding the `String` and `Vec` headers in every VM slot.
    Struct(HeapRef, Rc<StructLayout>),
    Module(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TableKey { Index(i128), Name(Rc<str>) }

#[derive(Clone, Debug, PartialEq, Eq)]
struct StructField { name: String, ty: Type, index: usize }
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructLayout { name: String, fields: Vec<StructField> }

impl Value {
    pub fn ty(&self) -> Type {
        match self {
            Self::I8(_) => Type::I8, Self::I16(_) => Type::I16, Self::I32(_) => Type::I32,
            Self::I64(_) => Type::I64, Self::U8(_) => Type::U8, Self::U16(_) => Type::U16,
            Self::U32(_) => Type::U32, Self::U64(_) => Type::U64, Self::F16(_) => Type::F16,
            Self::F32(_) => Type::F32, Self::F64(_) => Type::F64, Self::Bool(_) => Type::Bool,
            Self::String(_) => Type::String,
            Self::Array(_, element) => Type::Array(element.clone()),
            Self::Tensor(_, element, rank) => Type::Tensor(element.clone(), *rank),
            Self::Table(_, element) => Type::Table(element.clone()),
            Self::TableKey(_) => Type::TableKey,
            Self::TableKeys(_) => Type::TableKeys,
            Self::Struct(_, layout) => Type::Struct(layout.name.clone()),
            Self::Module(id) => Type::Module(id.clone()),
        }
    }

    fn pack_array(values: Vec<Value>, element: &Type) -> Result<Vec<u8>, Error> {
        let element_size = scalar_size(element)?;
        let mut bytes = Vec::with_capacity(values.len().checked_mul(element_size).ok_or_else(|| Error::Runtime("array is too large".into()))?);
        for value in values { encode_scalar(&value, element, &mut bytes)?; }
        Ok(bytes)
    }

}

#[derive(Clone, Debug)]
pub enum HeapObject {
    Array { bytes: Vec<u8>, element: Type },
    Tensor { bytes: Vec<u8>, element: Type, shape: Vec<usize> },
    String(String),
    Table { entries: HashMap<TableKey, Value>, element: Type },
    TableKeys(Vec<TableKey>),
    Struct { values: Vec<Value>, layout: Rc<StructLayout> },
}

#[derive(Clone, Debug)]
pub enum HeapSlot {
    Free { next_free: Option<usize> },
    Occupied { marked: bool, object: HeapObject },
}

/// Non-moving mark-and-sweep storage for every reference value in the VM.
#[derive(Debug)]
pub struct Heap {
    pub slots: Vec<HeapSlot>,
    pub free_head: Option<usize>,
    /// Estimated bytes owned by live heap objects. This is deliberately based
    /// on payload capacity rather than object count, since one tensor can be
    /// substantially larger than many small objects.
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

    fn heap_ref(value: &Value) -> Option<HeapRef> {
        match value {
            Value::Array(reference, _) | Value::Tensor(reference, _, _) | Value::String(reference)
            | Value::Table(reference, _) | Value::TableKeys(reference) | Value::Struct(reference, _) => Some(*reference),
            _ => None,
        }
    }

    fn mark_reference(&mut self, root: HeapRef) {
        // Heap graphs can be arbitrarily deep. Mark iteratively so a valid
        // script cannot exhaust the host thread stack during collection. The
        // worklist contains only compact HeapRef values. In particular, we do
        // not clone a table's or struct's values into a temporary vector.
        let mut work = vec![root];
        while let Some(reference) = work.pop() {
            let newly_marked = match self.slots.get_mut(reference.0) {
                Some(HeapSlot::Occupied { marked, .. }) => {
                    if *marked { false } else { *marked = true; true }
                }
                _ => false,
            };
            if !newly_marked { continue; }

            // The mutable borrow used to mark the slot has ended. Extending
            // the worklist directly keeps traversal allocation-free except
            // for the worklist's own amortized growth.
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
    fn collect(&mut self, roots: impl IntoIterator<Item = HeapRef>) -> usize {
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
    fn should_collect(&self) -> bool { self.allocated_bytes >= self.threshold_bytes }
}

fn table_key_display(key: &TableKey) -> String { match key { TableKey::Index(index) => format!("[{index}]"), TableKey::Name(name) => name.to_string() } }
fn table_key_from_value(heap: &Heap, value: &Value) -> Result<TableKey, Error> {
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
        Type::I16 | Type::U16 | Type::F16 => Some(2),
        Type::I32 | Type::U32 | Type::F32 => Some(4),
        Type::I64 | Type::U64 | Type::F64 => Some(8),
        Type::Array(_) | Type::Tensor(_, _) | Type::Table(_) | Type::TableKey | Type::TableKeys | Type::Struct(_) | Type::String | Type::Module(_) => None,
    }
}

fn scalar_size(ty: &Type) -> Result<usize, Error> { type_size(ty).ok_or_else(|| Error::Type("packed storage supports scalar field and element types only".into())) }

fn encode_scalar(value: &Value, element: &Type, bytes: &mut Vec<u8>) -> Result<(), Error> {
    if &value.ty() != element { return Err(Error::Runtime("VM array type invariant broken".into())); }
    match value {
        Value::I8(v) => bytes.push(*v as u8), Value::U8(v) => bytes.push(*v),
        Value::Bool(v) => bytes.push(u8::from(*v)),
        Value::I16(v) => bytes.extend(v.to_le_bytes()), Value::U16(v) | Value::F16(v) => bytes.extend(v.to_le_bytes()),
        Value::I32(v) => bytes.extend(v.to_le_bytes()), Value::U32(v) => bytes.extend(v.to_le_bytes()),
        Value::F32(v) => bytes.extend(v.to_bits().to_le_bytes()),
        Value::I64(v) => bytes.extend(v.to_le_bytes()), Value::U64(v) => bytes.extend(v.to_le_bytes()),
        Value::F64(v) => bytes.extend(v.to_bits().to_le_bytes()),
        _ => return Err(Error::Type("packed arrays can contain only scalar values".into())),
    }
    Ok(())
}
fn decode_scalar(bytes: &[u8], index: usize, element: &Type) -> Result<Value, Error> {
    let size = scalar_size(element)?;
    let offset = index.checked_mul(size).ok_or_else(|| Error::Runtime("array index too large".into()))?;
    if offset + size > bytes.len() {
        return Err(Error::Runtime(format!("array index {} is out of bounds", index)));
    }

    let cell = &bytes[offset..offset + size];
    match element {
        Type::I8 => Ok(Value::I8(cell[0] as i8)), Type::U8 => Ok(Value::U8(cell[0])), Type::Bool => Ok(Value::Bool(cell[0] != 0)),
        Type::I16 => Ok(Value::I16(i16::from_le_bytes(cell.try_into().unwrap()))), Type::U16 => Ok(Value::U16(u16::from_le_bytes(cell.try_into().unwrap()))),
        Type::F16 => Ok(Value::F16(u16::from_le_bytes(cell.try_into().unwrap()))), Type::I32 => Ok(Value::I32(i32::from_le_bytes(cell.try_into().unwrap()))),
        Type::U32 => Ok(Value::U32(u32::from_le_bytes(cell.try_into().unwrap()))), Type::F32 => Ok(Value::F32(f32::from_bits(u32::from_le_bytes(cell.try_into().unwrap())))),
        Type::I64 => Ok(Value::I64(i64::from_le_bytes(cell.try_into().unwrap()))), Type::U64 => Ok(Value::U64(u64::from_le_bytes(cell.try_into().unwrap()))),
        Type::F64 => Ok(Value::F64(f64::from_bits(u64::from_le_bytes(cell.try_into().unwrap())))),
        _ => Err(Error::Type("not a scalar type".into())),
    }
}

fn write_scalar(bytes: &mut [u8], index: usize, value: &Value, element: &Type) -> Result<(), Error> {
    let size = scalar_size(element)?;
    let offset = index.checked_mul(size).ok_or_else(|| Error::Runtime("array index too large".into()))?;
    if offset + size > bytes.len() {
        return Err(Error::Runtime(format!("array index {} is out of bounds", index)));
    }
    if &value.ty() != element { return Err(Error::Runtime("VM array type invariant broken".into())); }
    let cell = &mut bytes[offset..offset + size];
    match value {
        Value::I8(v) => cell[0] = *v as u8, Value::U8(v) => cell[0] = *v, Value::Bool(v) => cell[0] = u8::from(*v),
        Value::I16(v) => cell.copy_from_slice(&v.to_le_bytes()), Value::U16(v) | Value::F16(v) => cell.copy_from_slice(&v.to_le_bytes()),
        Value::I32(v) => cell.copy_from_slice(&v.to_le_bytes()), Value::U32(v) => cell.copy_from_slice(&v.to_le_bytes()), Value::F32(v) => cell.copy_from_slice(&v.to_bits().to_le_bytes()),
        Value::I64(v) => cell.copy_from_slice(&v.to_le_bytes()), Value::U64(v) => cell.copy_from_slice(&v.to_le_bytes()), Value::F64(v) => cell.copy_from_slice(&v.to_bits().to_le_bytes()),
        _ => return Err(Error::Type("packed arrays can contain only scalar values".into())),
    }
    Ok(())
}
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I8(v) => write!(f, "{v}"), Self::I16(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"), Self::I64(v) => write!(f, "{v}"),
            Self::U8(v) => write!(f, "{v}"), Self::U16(v) => write!(f, "{v}"),
            Self::U32(v) => write!(f, "{v}"), Self::U64(v) => write!(f, "{v}"),
            Self::F16(v) => write!(f, "{}", f16_to_f32(*v)), Self::F32(v) => write!(f, "{v}"),
            Self::F64(v) => write!(f, "{v}"), Self::Bool(v) => write!(f, "{v}"),
            Self::String(reference) => write!(f, "string@{}", reference.0),
            Self::Array(reference, element) => write!(f, "vector<{}>@{}", element, reference.0), Self::Tensor(reference, element, rank) => write!(f, "tensor<{}, {}>@{}", element, rank, reference.0), Self::Table(reference, element) => write!(f, "table<{}>@{}", element, reference.0), Self::TableKey(key) => write!(f, "{}", table_key_display(key)), Self::TableKeys(reference) => write!(f, "table_keys@{}", reference.0), Self::Struct(reference, layout) => write!(f, "{}@{}", layout.name, reference.0), Self::Module(id) => write!(f, "module({id})")
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceLocation { pub offset: usize, pub line: usize, pub column: usize }
impl SourceLocation {
    fn at(chars: &[char], offset: usize) -> Self { let mut line = 1; let mut column = 1; for ch in &chars[..offset.min(chars.len())] { if *ch == '\n' { line += 1; column = 1; } else { column += 1; } } Self { offset, line, column } }
}
#[derive(Debug, Clone, PartialEq)]
pub enum Error { Lex(String), Parse(String), Type(String), Runtime(String), Located { source: Box<Error>, location: SourceLocation } }
impl Error {
    fn at(self, location: SourceLocation) -> Self { match self { Self::Located { .. } => self, source => Self::Located { source: Box::new(source), location } } }
    pub fn location(&self) -> Option<SourceLocation> { match self { Self::Located { location, .. } => Some(*location), _ => None } }
}
impl fmt::Display for Error { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { match self { Self::Lex(message) => write!(f, "lex error: {message}"), Self::Parse(message) => write!(f, "parse error: {message}"), Self::Type(message) => write!(f, "type error: {message}"), Self::Runtime(message) => write!(f, "runtime error: {message}"), Self::Located { source, location } => write!(f, "{source} at line {}, column {}", location.line, location.column), } } }
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinOp { Add, Sub, Mul, Div, Mod, Eq, Neq, Lt, Le, Gt, Ge, And, Or, BitAnd, BitOr, BitXor, Shl, Shr }
#[derive(Clone, Debug, PartialEq)]
pub enum UnOp { Neg, Not }

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Let, Print, Printf, Putc, Input, This, Function, Export, Require, If, Then, Else, ElseIf, While, For, Do, Break, Continue, Struct, Table, End,
    As, Ident(StringId), Integer(i128), Float(f64), StringLit(StringId), Colon, DoubleColon,
    Equal, EqualEqual, Bang, BangEq, Plus, Minus, Star, Slash, Percent,
    Ampersand, Pipe, Caret, Shl, Shr, AndAnd, OrOr,
    Dot, Lt, Le, Gt, Ge, LParen, RParen, LBracket, RBracket, LBrace, RBrace, Comma, Semi, Eof
}

#[derive(Clone, Debug)]
struct SpannedToken { kind: Token, location: SourceLocation }
struct TokenBuffer { tokens: Vec<SpannedToken>, location: SourceLocation }
impl TokenBuffer {
    fn new() -> Self { Self { tokens: Vec::new(), location: SourceLocation { offset: 0, line: 1, column: 1 } } }
    fn begin(&mut self, chars: &[char], offset: usize) { self.location = SourceLocation::at(chars, offset); }
    fn location(&self) -> SourceLocation { self.location }
    fn push(&mut self, kind: Token) { self.tokens.push(SpannedToken { kind, location: self.location }); }
    fn into_tokens(self) -> Vec<SpannedToken> { self.tokens }
}

#[derive(Clone, Debug)]
enum Expr {
    Integer(i128), Float(f64), String(StringId), Input, This, Require(String), Name(StringId),
    Array(Vec<Expr>), Table(Vec<(TableLiteralKey, Expr)>), StructLiteral(String, Vec<(String, Expr)>),
    Cast(Box<Expr>, Type),
    Binary(Box<Expr>, BinOp, Box<Expr>), Unary(UnOp, Box<Expr>),
    Index(Box<Expr>, Vec<Expr>), Field(Box<Expr>, String),
    Call(String, Vec<Expr>), TensorFactory { name: String, element: Type, shape: Box<Expr> },
    Located { node: Box<Expr>, location: SourceLocation },
}

struct LexedTokens { tokens: Vec<SpannedToken>, strings: StringInterner }

fn lex(source: &str) -> Result<LexedTokens, Error> {
    let mut result = TokenBuffer::new(); let mut strings = StringInterner::new(); let chars: Vec<char> = source.chars().collect(); let mut i = 0;
    while i < chars.len() {
        result.begin(&chars, i);
        match chars[i] {
            c if c.is_whitespace() => i += 1,
            '-' if chars.get(i + 1) == Some(&'-') => { while i < chars.len() && chars[i] != '\n' { i += 1; } }
            ':' if chars.get(i + 1) == Some(&':') => { result.push(Token::DoubleColon); i += 2; }, ':' => { result.push(Token::Colon); i += 1; },
            '=' if chars.get(i + 1) == Some(&'=') => { result.push(Token::EqualEqual); i += 2; }, '=' => { result.push(Token::Equal); i += 1; },
            '!' if chars.get(i + 1) == Some(&'=') => { result.push(Token::BangEq); i += 2; },
            '!' => { result.push(Token::Bang); i += 1; },
            '<' if chars.get(i + 1) == Some(&'=') => { result.push(Token::Le); i += 2; },
            '<' if chars.get(i + 1) == Some(&'<') => { result.push(Token::Shl); i += 2; },
            '<' => { result.push(Token::Lt); i += 1; },
            '>' if chars.get(i + 1) == Some(&'=') => { result.push(Token::Ge); i += 2; },
            '>' if chars.get(i + 1) == Some(&'>') => { result.push(Token::Shr); i += 2; },
            '>' => { result.push(Token::Gt); i += 1; },
            '&' if chars.get(i + 1) == Some(&'&') => { result.push(Token::AndAnd); i += 2; },
            '&' => { result.push(Token::Ampersand); i += 1; },
            '|' if chars.get(i + 1) == Some(&'|') => { result.push(Token::OrOr); i += 2; },
            '|' => { result.push(Token::Pipe); i += 1; },
            '^' => { result.push(Token::Caret); i += 1; },
            '+' => { result.push(Token::Plus); i += 1; },
            '-' => { result.push(Token::Minus); i += 1; },
            '*' => { result.push(Token::Star); i += 1; },
            '/' => { result.push(Token::Slash); i += 1; },
            '%' => { result.push(Token::Percent); i += 1; },
            '.' => { result.push(Token::Dot); i += 1; },
            '(' => { result.push(Token::LParen); i += 1; }, ')' => { result.push(Token::RParen); i += 1; },
            '[' => { result.push(Token::LBracket); i += 1; }, ']' => { result.push(Token::RBracket); i += 1; },
            '{' => { result.push(Token::LBrace); i += 1; }, '}' => { result.push(Token::RBrace); i += 1; },
            ',' => { result.push(Token::Comma); i += 1; }, ';' => { result.push(Token::Semi); i += 1; },
            '"' => {
                let mut string_val = String::new();
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        i += 1;
                        match chars[i] {
                            'n' => string_val.push('\n'),
                            't' => string_val.push('\t'),
                            'r' => string_val.push('\r'),
                            '\\' => string_val.push('\\'),
                            '"' => string_val.push('"'),
                            c => { string_val.push('\\'); string_val.push(c); }
                        }
                    } else {
                        string_val.push(chars[i]);
                    }
                    i += 1;
                }
                if i >= chars.len() { return Err(Error::Lex("unterminated string".into()).at(result.location())); }
                i += 1;
                result.push(Token::StringLit(strings.intern(&string_val)));
            },
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i; i += 1; while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') { i += 1; }
                let word: String = chars[start..i].iter().collect();
                result.push(match word.as_str() {
                    "let" => Token::Let, "print" => Token::Print, "printf" => Token::Printf,
                    "putc" => Token::Putc, "input" => Token::Input, "this" => Token::This,
                    "function" => Token::Function, "export" => Token::Export, "require" => Token::Require,
                    "if" => Token::If, "then" => Token::Then, "else" => Token::Else, "elseif" => Token::ElseIf,
                    "while" => Token::While, "for" => Token::For, "do" => Token::Do,
                    "break" => Token::Break, "continue" => Token::Continue, "struct" => Token::Struct,
                    "table" => Token::Table, "end" => Token::End, "as" => Token::As, _ => Token::Ident(strings.intern(&word))
                });
            }
            c if c.is_ascii_digit() => {
                let start = i; let mut dot = false; i += 1;
                while i < chars.len() && (chars[i].is_ascii_digit() || (!dot && chars[i] == '.')) { if chars[i] == '.' { dot = true; } i += 1; }
                let raw: String = chars[start..i].iter().collect();
                result.push(if dot { Token::Float(raw.parse().map_err(|_| Error::Lex(raw.clone()))?) } else { Token::Integer(raw.parse().map_err(|_| Error::Lex(raw.clone()))?) });
            }
            other => return Err(Error::Lex(format!("unexpected character '{other}'")).at(result.location())),
        }
    }
    result.begin(&chars, i); result.push(Token::Eof); Ok(LexedTokens { tokens: result.into_tokens(), strings })
}

#[derive(Clone, Debug)]
enum TableLiteralKey { Index(Expr), Name(String) }
#[derive(Clone, Debug)]
struct StructMethod { name: String, args: Vec<(String, Type)>, body: Option<Vec<Statement>> }
#[derive(Clone, Debug)]
enum MethodReceiver { Name(String), This }
#[derive(Clone, Debug)]
enum Statement {
    Struct { name: String, fields: Vec<(String, Type)>, methods: Vec<StructMethod> },
    MethodDefinition { struct_name: String, method: String, args: Vec<(String, Type)>, body: Vec<Statement> },
    ExportLet { name: StringId, ty: Type, expr: Expr },
    ExportStruct { name: String, fields: Vec<(String, Type)>, methods: Vec<StructMethod> },
    ExportFunction { name: String, body: Vec<Statement> },
    CallMethod { receiver: MethodReceiver, method: String },
    Let { name: StringId, ty: Type, expr: Expr },
    Assign { name: String, expr: Expr },
    SetIndex { name: String, indices: Vec<Expr>, expr: Expr },
    SetField { name: String, field: String, expr: Expr },
    SetFieldIndex { name: String, field: String, index: Expr, expr: Expr },
    Print(Expr), Printf { format: Expr, args: Vec<Expr> }, Putc(Expr),
    If { condition: Expr, then_body: Vec<Statement>, else_body: Vec<Statement> },
    While { condition: Expr, body: Vec<Statement> },
    For { name: String, start: Expr, end: Expr, body: Vec<Statement> },
    Break,
    Continue,
    Located { node: Box<Statement>, location: SourceLocation },
}

struct Parser { tokens: Vec<SpannedToken>, strings: StringInterner, at: usize, last_location: SourceLocation }
impl Parser {
    fn new(lexed: LexedTokens) -> Self { Self { tokens: lexed.tokens, strings: lexed.strings, at: 0, last_location: SourceLocation { offset: 0, line: 1, column: 1 } } }
    fn string(&self, id: StringId) -> String { self.strings.resolve(id).to_owned() }
    fn peek(&self) -> &Token { &self.tokens[self.at].kind }
    fn location(&self) -> SourceLocation { self.tokens.get(self.at).map(|token| token.location).unwrap_or(self.last_location) }
    fn next(&mut self) -> Token { let token = self.tokens[self.at].clone(); self.at += 1; self.last_location = token.location; token.kind }
    fn need(&mut self, wanted: Token) -> Result<(), Error> { let got = self.next(); if got == wanted { Ok(()) } else { Err(Error::Parse(format!("expected {wanted:?}, got {got:?}"))) } }
    fn program(&mut self) -> Result<Vec<Statement>, Error> { self.block().map_err(|error| error.at(self.location())) }
    fn into_program(mut self) -> Result<(Vec<Statement>, StringInterner), Error> {
        let program = self.program()?;
        Ok((program, self.strings))
    }
    fn block(&mut self) -> Result<Vec<Statement>, Error> {
        let mut statements = Vec::new();
        while !matches!(self.peek(), Token::Eof | Token::Else | Token::ElseIf | Token::End) {
            statements.push(self.statement()?);
            if *self.peek() == Token::Semi { self.next(); }
        }
        Ok(statements)
    }
    fn parse_else_if_chain(&mut self) -> Result<Vec<Statement>, Error> {
        if *self.peek() == Token::ElseIf {
            let location = self.location();
            self.next(); // поглощаем ElseIf
            let condition = self.expr()?;
            self.need(Token::Then)?;
            let then_body = self.block()?;
            let else_body = self.parse_else_if_chain()?;
            let nested_if = Statement::Located {
                node: Box::new(Statement::If { condition, then_body, else_body }),
                location
            };
            Ok(vec![nested_if])
        } else if *self.peek() == Token::Else {
            self.next();
            let else_body = self.block()?;
            self.need(Token::End)?;
            Ok(else_body)
        } else {
            self.need(Token::End)?;
            Ok(Vec::new())
        }
    }
    fn struct_declaration(&mut self) -> Result<(String, Vec<(String, Type)>, Vec<StructMethod>), Error> {
        let name = match self.next() { Token::Ident(name) => self.string(name), token => return Err(Error::Parse(format!("expected struct name, got {token:?}"))) };
        self.need(Token::LBrace)?;
        let mut fields = Vec::new(); let mut methods = Vec::new();
        while *self.peek() != Token::RBrace { match self.next() {
            Token::Ident(field) => { self.need(Token::Colon)?; let ty = self.ty()?; fields.push((self.string(field), ty)); if matches!(self.peek(), Token::Semi | Token::Comma) { self.next(); } },
            Token::Function => {
                let method = match self.next() { Token::Ident(method) => self.string(method), token => return Err(Error::Parse(format!("expected method name, got {token:?}"))) };
                self.need(Token::LParen)?;
                let mut args = Vec::new();
                if *self.peek() != Token::RParen {
                    loop {
                        let arg_name = match self.next() { Token::Ident(n) => self.string(n), t => return Err(Error::Parse(format!("expected argument name, got {t:?}"))) };
                        self.need(Token::Colon)?;
                        let arg_ty = self.ty()?;
                        args.push((arg_name, arg_ty));
                        if *self.peek() != Token::Comma { break; }
                        self.next();
                    }
                }
                self.need(Token::RParen)?;
                let body = if *self.peek() == Token::End {
                    self.next();
                    None
                } else if *self.peek() == Token::RBrace {
                    None
                } else {
                    let body = self.block()?;
                    self.need(Token::End)?;
                    Some(body)
                };
                methods.push(StructMethod { name: method, args, body });
            },
            token => return Err(Error::Parse(format!("expected field or method, got {token:?}"))),
        } }
        self.need(Token::RBrace)?; Ok((name, fields, methods))
    }
    fn let_declaration(&mut self) -> Result<(StringId, Type, Expr), Error> {
        let name = match self.next() { Token::Ident(n) => n, x => return Err(Error::Parse(format!("expected name, got {x:?}"))) };
        self.need(Token::Colon)?; let ty = self.ty()?; self.need(Token::Equal)?; let expr = self.expr()?; Ok((name, ty, expr))
    }
    fn statement(&mut self) -> Result<Statement, Error> { let location = self.location(); let node = self.statement_inner().map_err(|error| error.at(location))?; Ok(Statement::Located { node: Box::new(node), location }) }
    fn statement_inner(&mut self) -> Result<Statement, Error> { match self.next() {
        Token::Struct => { let (name, fields, methods) = self.struct_declaration()?; Ok(Statement::Struct { name, fields, methods }) },
        Token::Export => match self.next() {
            Token::Let => { let (name, ty, expr) = self.let_declaration()?; Ok(Statement::ExportLet { name, ty, expr }) },
            Token::Struct => { let (name, fields, methods) = self.struct_declaration()?; Ok(Statement::ExportStruct { name, fields, methods }) },
            Token::Function => { let name = match self.next() { Token::Ident(name) => self.string(name), token => return Err(Error::Parse(format!("expected exported function name, got {token:?}"))) }; self.need(Token::LParen)?; self.need(Token::RParen)?; let body = self.block()?; self.need(Token::End)?; Ok(Statement::ExportFunction { name, body }) },
            token => Err(Error::Parse(format!("expected let, struct, or function after export, got {token:?}"))),
        },
        Token::Function => {
            let struct_name = match self.next() { Token::Ident(name) => self.string(name), token => return Err(Error::Parse(format!("expected struct name, got {token:?}"))) };
            self.need(Token::DoubleColon)?;
            let method = match self.next() { Token::Ident(method) => self.string(method), token => return Err(Error::Parse(format!("expected method name, got {token:?}"))) };
            self.need(Token::LParen)?;
            let mut args = Vec::new();
            if *self.peek() != Token::RParen {
                loop {
                    let arg_name = match self.next() { Token::Ident(name) => self.string(name), token => return Err(Error::Parse(format!("expected argument name, got {token:?}"))) };
                    self.need(Token::Colon)?;
                    let arg_ty = self.ty()?;
                    args.push((arg_name, arg_ty));
                    if *self.peek() != Token::Comma { break; }
                    self.next();
                }
            }
            self.need(Token::RParen)?;
            let body = self.block()?;
            self.need(Token::End)?;
            Ok(Statement::MethodDefinition { struct_name, method, args, body })
        },
        Token::Let => { let (name, ty, expr) = self.let_declaration()?; Ok(Statement::Let { name, ty, expr }) },
        Token::Print => { let parenthesized = *self.peek() == Token::LParen; if parenthesized { self.next(); } let expr = self.expr()?; if parenthesized { self.need(Token::RParen)?; } Ok(Statement::Print(expr)) },
        Token::Printf => {
            let parenthesized = *self.peek() == Token::LParen;
            if parenthesized { self.next(); }
            let format = self.expr()?;
            let mut args = Vec::new();
            while *self.peek() == Token::Comma {
                self.next();
                args.push(self.expr()?);
            }
            if parenthesized { self.need(Token::RParen)?; }
            Ok(Statement::Printf { format, args })
        },
        Token::Putc => { let parenthesized = *self.peek() == Token::LParen; if parenthesized { self.next(); } let expr = self.expr()?; if parenthesized { self.need(Token::RParen)?; } Ok(Statement::Putc(expr)) },
        Token::If => { let condition = self.expr()?; self.need(Token::Then)?; let then_body = self.block()?; let else_body = self.parse_else_if_chain()?; Ok(Statement::If { condition, then_body, else_body }) },
        Token::While => { let condition = self.expr()?; self.need(Token::Do)?; let body = self.block()?; self.need(Token::End)?; Ok(Statement::While { condition, body }) },
        Token::For => {
            let name = match self.next() { Token::Ident(name) => self.string(name), token => return Err(Error::Parse(format!("expected loop variable, got {token:?}"))) };
            self.need(Token::Equal)?;
            let start = self.expr()?;
            self.need(Token::Comma)?;
            let end = self.expr()?;
            self.need(Token::Do)?;
            let body = self.block()?;
            self.need(Token::End)?;
            Ok(Statement::For { name, start, end, body })
        },
        Token::Break => Ok(Statement::Break),
        Token::Continue => Ok(Statement::Continue),
        Token::This => { self.need(Token::Dot)?; let method = match self.next() { Token::Ident(method) => self.string(method), token => return Err(Error::Parse(format!("expected method name, got {token:?}"))) }; self.need(Token::LParen)?; self.need(Token::RParen)?; Ok(Statement::CallMethod { receiver: MethodReceiver::This, method }) },
        Token::Ident(name) => { let name = self.string(name); match self.next() {
            Token::Equal => Ok(Statement::Assign { name, expr: self.expr()? }),
            Token::LBracket => { let indices = self.indices()?; self.need(Token::Equal)?; Ok(Statement::SetIndex { name, indices, expr: self.expr()? }) },
            Token::Dot => {
                let field = match self.next() { Token::Ident(field) => self.string(field), token => return Err(Error::Parse(format!("expected field or method name, got {token:?}"))) };
                match self.next() {
                    Token::Equal => Ok(Statement::SetField { name, field, expr: self.expr()? }),
                    Token::LBracket => { let index = self.expr()?; self.need(Token::RBracket)?; self.need(Token::Equal)?; Ok(Statement::SetFieldIndex { name, field, index, expr: self.expr()? }) },
                    Token::LParen => { self.need(Token::RParen)?; Ok(Statement::CallMethod { receiver: MethodReceiver::Name(name), method: field }) },
                    token => Err(Error::Parse(format!("expected '=', '[', or '(', got {token:?}"))),
                }
            },
            token => Err(Error::Parse(format!("expected '=', '[' or '.', got {token:?}")))
        }},
        x => Err(Error::Parse(format!("expected statement, got {x:?}"))), }
    }
    fn ty(&mut self) -> Result<Type, Error> { match self.next() {
        Token::Ident(n) => { let n = self.string(n); match n.as_str() {
            "i8" => Ok(Type::I8), "i16" => Ok(Type::I16), "i32" => Ok(Type::I32), "i64" => Ok(Type::I64),
            "u8" => Ok(Type::U8), "u16" => Ok(Type::U16), "u32" => Ok(Type::U32), "u64" => Ok(Type::U64),
            "f16" => Ok(Type::F16), "f32" => Ok(Type::F32), "f64" => Ok(Type::F64), "bool" => Ok(Type::Bool),
            "table_key" => Ok(Type::TableKey), "table_keys" => Ok(Type::TableKeys),
            "vector" => { self.need(Token::Lt)?; let inner = self.ty()?; self.need(Token::Gt)?; Ok(Type::Array(Box::new(inner))) },
            "tensor" => {
                self.need(Token::Lt)?; let inner = self.ty()?; self.need(Token::Comma)?;
                let rank = match self.next() { Token::Integer(rank) if rank > 0 => usize::try_from(rank).map_err(|_| Error::Parse("tensor rank is too large".into()))?, token => return Err(Error::Parse(format!("tensor rank must be a positive integer, got {token:?}"))) };
                self.need(Token::Gt)?; scalar_size(&inner)?; Ok(Type::Tensor(Box::new(inner), rank))
            },
            "string" => Ok(Type::String),
            "module" => Ok(Type::Module(String::new())),
            _ if *self.peek() == Token::Dot => {
                self.next();
                let member = match self.next() { Token::Ident(member) => self.string(member), token => return Err(Error::Parse(format!("expected exported struct name, got {token:?}"))) };
                Ok(Type::Struct(format!("{}.{}", n, member)))
            },
            _ => Ok(Type::Struct(n)),
        }},
        Token::Table => { self.need(Token::Lt)?; let inner = self.ty()?; self.need(Token::Gt)?; Ok(Type::Table(Box::new(inner))) },
        Token::Struct => match self.next() { Token::Ident(name) => Ok(Type::Struct(self.string(name))), token => Err(Error::Parse(format!("expected struct name, got {token:?}"))) },
        x => Err(Error::Parse(format!("expected type, got {x:?}"))) }
    }
    fn expr(&mut self) -> Result<Expr, Error> { let location = self.location(); let node = self.expr_inner().map_err(|error| error.at(location))?; Ok(Expr::Located { node: Box::new(node), location }) }
    fn expr_inner(&mut self) -> Result<Expr, Error> { self.logical_or() }
    fn indices(&mut self) -> Result<Vec<Expr>, Error> {
        let mut indices = Vec::new();
        loop {
            indices.push(self.expr()?);
            if *self.peek() != Token::Comma { break; }
            self.next();
        }
        self.need(Token::RBracket)?;
        Ok(indices)
    }
    fn logical_or(&mut self) -> Result<Expr, Error> {
        let mut e = self.logical_and()?;
        while *self.peek() == Token::OrOr { self.next(); e = Expr::Binary(Box::new(e), BinOp::Or, Box::new(self.logical_and()?)); }
        Ok(e)
    }
    fn logical_and(&mut self) -> Result<Expr, Error> {
        let mut e = self.equality()?;
        while *self.peek() == Token::AndAnd { self.next(); e = Expr::Binary(Box::new(e), BinOp::And, Box::new(self.equality()?)); }
        Ok(e)
    }
    fn equality(&mut self) -> Result<Expr, Error> {
        let mut e = self.relational()?;
        while matches!(self.peek(), Token::EqualEqual | Token::BangEq) {
            let op = match self.next() { Token::EqualEqual => BinOp::Eq, Token::BangEq => BinOp::Neq, _ => unreachable!() };
            e = Expr::Binary(Box::new(e), op, Box::new(self.relational()?));
        }
        Ok(e)
    }
    fn relational(&mut self) -> Result<Expr, Error> {
        let mut e = self.bitwise_or()?;
        while matches!(self.peek(), Token::Lt | Token::Le | Token::Gt | Token::Ge) {
            let op = match self.next() { Token::Lt => BinOp::Lt, Token::Le => BinOp::Le, Token::Gt => BinOp::Gt, Token::Ge => BinOp::Ge, _ => unreachable!() };
            e = Expr::Binary(Box::new(e), op, Box::new(self.bitwise_or()?));
        }
        Ok(e)
    }
    fn bitwise_or(&mut self) -> Result<Expr, Error> {
        let mut e = self.bitwise_xor()?;
        while *self.peek() == Token::Pipe { self.next(); e = Expr::Binary(Box::new(e), BinOp::BitOr, Box::new(self.bitwise_xor()?)); }
        Ok(e)
    }
    fn bitwise_xor(&mut self) -> Result<Expr, Error> {
        let mut e = self.bitwise_and()?;
        while *self.peek() == Token::Caret { self.next(); e = Expr::Binary(Box::new(e), BinOp::BitXor, Box::new(self.bitwise_and()?)); }
        Ok(e)
    }
    fn bitwise_and(&mut self) -> Result<Expr, Error> {
        let mut e = self.shift()?;
        while *self.peek() == Token::Ampersand { self.next(); e = Expr::Binary(Box::new(e), BinOp::BitAnd, Box::new(self.shift()?)); }
        Ok(e)
    }
    fn shift(&mut self) -> Result<Expr, Error> {
        let mut e = self.additive()?;
        while matches!(self.peek(), Token::Shl | Token::Shr) {
            let op = match self.next() { Token::Shl => BinOp::Shl, Token::Shr => BinOp::Shr, _ => unreachable!() };
            e = Expr::Binary(Box::new(e), op, Box::new(self.additive()?));
        }
        Ok(e)
    }
    fn additive(&mut self) -> Result<Expr, Error> {
        let mut e = self.multiplicative()?;
        while matches!(self.peek(), Token::Plus | Token::Minus) {
            let op = match self.next() { Token::Plus => BinOp::Add, Token::Minus => BinOp::Sub, _ => unreachable!() };
            e = Expr::Binary(Box::new(e), op, Box::new(self.multiplicative()?));
        }
        Ok(e)
    }
    fn multiplicative(&mut self) -> Result<Expr, Error> {
        let mut e = self.unary()?;
        while matches!(self.peek(), Token::Star | Token::Slash | Token::Percent) {
            let op = match self.next() { Token::Star => BinOp::Mul, Token::Slash => BinOp::Div, Token::Percent => BinOp::Mod, _ => unreachable!() };
            e = Expr::Binary(Box::new(e), op, Box::new(self.unary()?));
        }
        Ok(e)
    }
    fn cast(&mut self) -> Result<Expr, Error> {
        let mut e = self.primary()?;
        while *self.peek() == Token::As {
            self.next();
            let target_ty = self.ty()?;
            e = Expr::Cast(Box::new(e), target_ty);
        }
        Ok(e)
    }
    fn unary(&mut self) -> Result<Expr, Error> {
        if *self.peek() == Token::Minus {
            self.next();
            Ok(Expr::Unary(UnOp::Neg, Box::new(self.unary()?)))
        } else if *self.peek() == Token::Bang {
            self.next();
            Ok(Expr::Unary(UnOp::Not, Box::new(self.unary()?)))
        } else {
            self.cast()
        }
    }
    fn primary(&mut self) -> Result<Expr, Error> {
        let mut e = match self.next() {
            Token::Integer(n) => Expr::Integer(n),
            Token::Float(v) => Expr::Float(v),
            Token::StringLit(s) => Expr::String(s),
            Token::Input => Expr::Input,
            Token::This => Expr::This,
            Token::Require => {
                self.need(Token::LParen)?;
                let path = match self.next() { Token::StringLit(path) => path, token => return Err(Error::Parse(format!("require expects a string literal, got {token:?}"))) };
                self.need(Token::RParen)?;
                Expr::Require(self.string(path))
            },
            Token::Table => {
                self.need(Token::LBrace)?;
                let mut entries = Vec::new();
                while *self.peek() != Token::RBrace {
                    let key = if *self.peek() == Token::LBracket {
                        self.next(); let index = self.expr()?; self.need(Token::RBracket)?; TableLiteralKey::Index(index)
                    } else {
                        match self.next() { Token::Ident(name) => TableLiteralKey::Name(self.string(name)), token => return Err(Error::Parse(format!("expected table key, got {token:?}"))), }
                    };
                    self.need(Token::Equal)?; entries.push((key, self.expr()?));
                    if matches!(self.peek(), Token::Comma | Token::Semi) { self.next(); }
                }
                self.need(Token::RBrace)?; Expr::Table(entries)
            },
            Token::Ident(n) => {
                let n_text = self.string(n);
                let mut struct_name = n_text.clone();
                if *self.peek() == Token::Dot
                    && matches!(self.tokens.get(self.at + 1), Some(SpannedToken { kind: Token::Ident(_), .. }))
                    && matches!(self.tokens.get(self.at + 2), Some(SpannedToken { kind: Token::LBrace, .. }))
                {
                    self.next();
                    let member = match self.next() { Token::Ident(member) => self.string(member), _ => unreachable!() };
                    struct_name = format!("{n_text}.{member}");
                }
                if *self.peek() == Token::Lt && matches!(n_text.as_str(), "zeros" | "random") {
                    self.next();
                    let element = self.ty()?;
                    scalar_size(&element)?;
                    self.need(Token::Gt)?;
                    self.need(Token::LParen)?;
                    let shape = self.expr()?;
                    self.need(Token::RParen)?;
                    Expr::TensorFactory { name: n_text, element, shape: Box::new(shape) }
                } else if *self.peek() == Token::LBrace {
                    self.next(); let mut fields = Vec::new();
                    while *self.peek() != Token::RBrace {
                        let field = match self.next() { Token::Ident(field) => self.string(field), token => return Err(Error::Parse(format!("expected field name, got {token:?}"))) };
                        self.need(Token::Equal)?; fields.push((field, self.expr()?));
                        if *self.peek() == Token::Comma { self.next(); }
                    }
                    self.need(Token::RBrace)?; Expr::StructLiteral(struct_name, fields)
                } else if *self.peek() == Token::LParen {
                    self.next();
                    let mut args = Vec::new();
                    if *self.peek() != Token::RParen {
                        loop {
                            args.push(self.expr()?);
                            if *self.peek() != Token::Comma { break; }
                            self.next();
                        }
                    }
                    self.need(Token::RParen)?;
                    Expr::Call(n_text, args)
                } else { Expr::Name(n) }
            },
            Token::LParen => { let grouped = self.expr()?; self.need(Token::RParen)?; grouped },
            Token::LBracket => {
                let mut items = Vec::new();
                if *self.peek() != Token::RBracket {
                    loop {
                        items.push(self.expr()?);
                        if *self.peek() != Token::Comma { break; }
                        self.next();
                    }
                }
                self.need(Token::RBracket)?; Expr::Array(items)
            },
            x => return Err(Error::Parse(format!("expected expression, got {x:?}")))
        };
        loop {
            if *self.peek() == Token::LBracket {
                self.next(); e = Expr::Index(Box::new(e), self.indices()?);
            } else if *self.peek() == Token::Dot {
                self.next(); let field = match self.next() { Token::Ident(field) => self.string(field), token => return Err(Error::Parse(format!("expected field name, got {token:?}"))) }; e = Expr::Field(Box::new(e), field);
            } else { break; }
        }
        Ok(e)
    }
}
#[derive(Clone, Debug)]
enum TableEntry { Index, Name(Rc<str>) }

#[derive(Clone, Debug)]
enum ModuleExport { Value { slot: usize, ty: Type }, Function { entry: usize }, Struct(StructLayout) }

/// A numeric type is selected while compiling, not by the VM hot loop.
#[derive(Clone, Debug)]
enum BinaryOp {
    I8(BinOp), I16(BinOp), I32(BinOp), I64(BinOp),
    U8(BinOp), U16(BinOp), U32(BinOp), U64(BinOp),
    F16(BinOp), F32(BinOp), F64(BinOp),
    Equal, NotEqual,
}

#[derive(Clone, Debug)]
struct ModuleArtifact { id: String, code: Rc<FlatBytecode>, exports: HashMap<String, ModuleExport> }

#[derive(Clone, Debug)]
struct HostSignature { arguments: Vec<Type>, result: Type }

#[derive(Clone, Debug)]
/// Compiler-only intermediate instruction.  It is lowered before a program
/// reaches the VM; the VM executes `FlatBytecode`, never this enum.
enum IrOp {
    AddI32, AddF32, AddF64,
    Push(Value), MakeString(Rc<str>), Input(Rc<Type>), Require(Rc<ModuleArtifact>), Load(usize), LoadCurrentReceiver,
    LoadCurrentField(Rc<StructField>), Store(usize), StoreIndex(usize, Rc<Type>), StoreTableIndex(usize, Rc<Type>),
    StoreTensorIndex(usize, Rc<Type>, usize), StoreTensorIndexF32(usize, usize),
    StoreField(usize, Rc<StructField>), StoreFieldIndex(usize, Rc<StructField>, Rc<Type>), StoreTableField(usize, Rc<str>, Rc<Type>),
    StoreCurrentField(Rc<StructField>), MakeArray(usize, Rc<Type>), MakeTable(Rc<[TableEntry]>, Rc<Type>), MakeStruct(Rc<StructLayout>),
    MakeTensor(TensorInit, Rc<Type>, usize), Index, TensorIndex(Rc<Type>, usize), TensorIndexF32(usize), TableIndex, TableKeys, TableKeysIndex, TableRemove, Field(Rc<StructField>), TableField(Rc<str>), ModuleField(Rc<str>),
    Binary(BinaryOp), Unary(UnOp, Rc<Type>), Len, ConcatString,
    Cast(Rc<Type>),
    Builtin1(BuiltinFn, Rc<Type>), Builtin2(BuiltinFn, Rc<Type>),
    CallExternal(Rc<str>, usize),
    JumpIfFalse(usize), Jump(usize), JumpIfFalseKeep(usize), JumpIfTrueKeep(usize),
    CallMethod(usize, usize), CallCurrentMethod(usize), CallModule(usize, Rc<str>),
    Return, Print, Printf(usize), Putc
}

// Kept as a source-compatible spelling for the compiler's private IR.  No
// `Op` values are stored in executable bytecode or observed by the VM.
type Op = IrOp;

/// Compact executable program: opcode and operands are consecutive 32-bit
/// words.  Heap-owning data lives once in the constant pool, rather than in
/// every entry of the instruction stream.
#[derive(Clone, Debug, Default)]
struct FlatBytecode { words: Vec<u32>, constants: Vec<Constant> }

#[derive(Clone, Debug)]
enum Constant {
    Value(Value), String(Rc<str>), Type(Rc<Type>), Module(Rc<ModuleArtifact>),
    Field(Rc<StructField>), Entries(Rc<[TableEntry]>), Layout(Rc<StructLayout>),
    Binary(BinaryOp), Unary(UnOp), Builtin(BuiltinFn), TensorInit(TensorInit),
}

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
enum Opcode {
    AddI32, AddF32, AddF64, Push, MakeString, Input, Require, Load, LoadCurrentReceiver,
    LoadCurrentField, Store, StoreIndex, StoreTableIndex, StoreTensorIndex, StoreTensorIndexF32,
    StoreField, StoreFieldIndex, StoreTableField, StoreCurrentField, MakeArray, MakeTable,
    MakeStruct, MakeTensor, Index, TensorIndex, TensorIndexF32, TableIndex, TableKeys,
    TableKeysIndex, TableRemove, Field, TableField, ModuleField, Binary, Unary, Len,
    ConcatString, Builtin1, Builtin2, CallExternal, JumpIfFalse, Jump, JumpIfFalseKeep,
    JumpIfTrueKeep, CallMethod, CallCurrentMethod, CallModule, Return, Print, Printf, Putc,
    // Appended to preserve the numeric representation of existing opcodes.
    Cast,
}

impl Opcode {
    fn from_word(word: u32) -> Result<Self, Error> {
        // Opcodes are emitted only by this compiler.  Keep malformed external
        // bytecode diagnosable rather than treating it as undefined behaviour.
        if word > Self::Cast as u32 { return Err(Error::Runtime("invalid bytecode opcode".into())); }
        Ok(unsafe { std::mem::transmute(word) })
    }
}

enum DecodedOp<'a> {
    AddI32, AddF32, AddF64, Push(&'a Value), MakeString(&'a str), Input(&'a Type), Require(&'a ModuleArtifact), Load(usize), LoadCurrentReceiver,
    LoadCurrentField(&'a StructField), Store(usize), StoreIndex(usize, &'a Type), StoreTableIndex(usize, &'a Type),
    StoreTensorIndex(usize, &'a Type, usize), StoreTensorIndexF32(usize, usize),
    StoreField(usize, &'a StructField), StoreFieldIndex(usize, &'a StructField, &'a Type), StoreTableField(usize, &'a str, &'a Type),
    StoreCurrentField(&'a StructField), MakeArray(usize, &'a Type), MakeTable(&'a [TableEntry], &'a Type), MakeStruct(&'a StructLayout),
    MakeTensor(TensorInit, &'a Type, usize), Index, TensorIndex(&'a Type, usize), TensorIndexF32(usize), TableIndex, TableKeys, TableKeysIndex, TableRemove, Field(&'a StructField), TableField(&'a str), ModuleField(&'a str),
    Binary(&'a BinaryOp), Unary(&'a UnOp, &'a Type), Len, ConcatString, Cast(&'a Type),
    Builtin1(BuiltinFn, &'a Type), Builtin2(BuiltinFn, &'a Type), CallExternal(&'a str, usize),
    JumpIfFalse(usize), Jump(usize), JumpIfFalseKeep(usize), JumpIfTrueKeep(usize),
    CallMethod(usize, usize), CallCurrentMethod(usize), CallModule(usize, &'a str),
    Return, Print, Printf(usize), Putc,
}

impl FlatBytecode {
    fn constant(&mut self, constant: Constant) -> u32 { let index = self.constants.len() as u32; self.constants.push(constant); index }
    fn word(&mut self, value: usize) { self.words.push(value as u32); }
    fn op(&mut self, opcode: Opcode) { self.words.push(opcode as u32); }
    fn lower(ir: Vec<IrOp>) -> Self {
        let mut out = Self::default();
        for op in ir { match op {
            IrOp::AddI32 => out.op(Opcode::AddI32), IrOp::AddF32 => out.op(Opcode::AddF32), IrOp::AddF64 => out.op(Opcode::AddF64),
            IrOp::Push(v) => { out.op(Opcode::Push); let x=out.constant(Constant::Value(v)); out.word(x as usize); }
            IrOp::MakeString(v) => { out.op(Opcode::MakeString); let x=out.constant(Constant::String(v)); out.word(x as usize); }
            IrOp::Input(v) => { out.op(Opcode::Input); let x=out.constant(Constant::Type(v)); out.word(x as usize); }
            IrOp::Require(v) => { out.op(Opcode::Require); let x=out.constant(Constant::Module(v)); out.word(x as usize); }
            IrOp::Load(v) => { out.op(Opcode::Load); out.word(v); }, IrOp::LoadCurrentReceiver => out.op(Opcode::LoadCurrentReceiver),
            IrOp::LoadCurrentField(v) => { out.op(Opcode::LoadCurrentField); let x=out.constant(Constant::Field(v)); out.word(x as usize); }
            IrOp::Store(v) => { out.op(Opcode::Store); out.word(v); }
            IrOp::StoreIndex(a,b) => { out.op(Opcode::StoreIndex); out.word(a); let x=out.constant(Constant::Type(b)); out.word(x as usize); }
            IrOp::StoreTableIndex(a,b) => { out.op(Opcode::StoreTableIndex); out.word(a); let x=out.constant(Constant::Type(b)); out.word(x as usize); }
            IrOp::StoreTensorIndex(a,b,c) => { out.op(Opcode::StoreTensorIndex); out.word(a); let x=out.constant(Constant::Type(b)); out.word(x as usize); out.word(c); }
            IrOp::StoreTensorIndexF32(a,b) => { out.op(Opcode::StoreTensorIndexF32); out.word(a); out.word(b); }
            IrOp::StoreField(a,b) => { out.op(Opcode::StoreField); out.word(a); let x=out.constant(Constant::Field(b)); out.word(x as usize); }
            IrOp::StoreFieldIndex(a,b,c) => { out.op(Opcode::StoreFieldIndex); out.word(a); let x=out.constant(Constant::Field(b)); out.word(x as usize); let x=out.constant(Constant::Type(c)); out.word(x as usize); }
            IrOp::StoreTableField(a,b,c) => { out.op(Opcode::StoreTableField); out.word(a); let x=out.constant(Constant::String(b)); out.word(x as usize); let x=out.constant(Constant::Type(c)); out.word(x as usize); }
            IrOp::StoreCurrentField(v) => { out.op(Opcode::StoreCurrentField); let x=out.constant(Constant::Field(v)); out.word(x as usize); }
            IrOp::MakeArray(a,b) => { out.op(Opcode::MakeArray); out.word(a); let x=out.constant(Constant::Type(b)); out.word(x as usize); }
            IrOp::MakeTable(a,b) => { out.op(Opcode::MakeTable); let x=out.constant(Constant::Entries(a)); out.word(x as usize); let x=out.constant(Constant::Type(b)); out.word(x as usize); }
            IrOp::MakeStruct(v) => { out.op(Opcode::MakeStruct); let x=out.constant(Constant::Layout(v)); out.word(x as usize); }
            IrOp::MakeTensor(a,b,c) => { out.op(Opcode::MakeTensor); let x=out.constant(Constant::TensorInit(a)); out.word(x as usize); let x=out.constant(Constant::Type(b)); out.word(x as usize); out.word(c); }
            IrOp::Index => out.op(Opcode::Index), IrOp::TensorIndex(a,b) => { out.op(Opcode::TensorIndex); let x=out.constant(Constant::Type(a)); out.word(x as usize); out.word(b); }, IrOp::TensorIndexF32(v) => { out.op(Opcode::TensorIndexF32); out.word(v); },
            IrOp::TableIndex => out.op(Opcode::TableIndex), IrOp::TableKeys => out.op(Opcode::TableKeys), IrOp::TableKeysIndex => out.op(Opcode::TableKeysIndex), IrOp::TableRemove => out.op(Opcode::TableRemove),
            IrOp::Field(v) => { out.op(Opcode::Field); let x=out.constant(Constant::Field(v)); out.word(x as usize); }, IrOp::TableField(v) => { out.op(Opcode::TableField); let x=out.constant(Constant::String(v)); out.word(x as usize); }, IrOp::ModuleField(v) => { out.op(Opcode::ModuleField); let x=out.constant(Constant::String(v)); out.word(x as usize); },
            IrOp::Binary(v) => { out.op(Opcode::Binary); let x=out.constant(Constant::Binary(v)); out.word(x as usize); }, IrOp::Unary(a,b) => { out.op(Opcode::Unary); let x=out.constant(Constant::Unary(a)); out.word(x as usize); let x=out.constant(Constant::Type(b)); out.word(x as usize); }, IrOp::Len => out.op(Opcode::Len), IrOp::ConcatString => out.op(Opcode::ConcatString), IrOp::Cast(v) => { out.op(Opcode::Cast); let x=out.constant(Constant::Type(v)); out.word(x as usize); },
            IrOp::Builtin1(a,b) => { out.op(Opcode::Builtin1); let x=out.constant(Constant::Builtin(a)); out.word(x as usize); let x=out.constant(Constant::Type(b)); out.word(x as usize); }, IrOp::Builtin2(a,b) => { out.op(Opcode::Builtin2); let x=out.constant(Constant::Builtin(a)); out.word(x as usize); let x=out.constant(Constant::Type(b)); out.word(x as usize); },
            IrOp::CallExternal(a,b) => { out.op(Opcode::CallExternal); let x=out.constant(Constant::String(a)); out.word(x as usize); out.word(b); },
            IrOp::JumpIfFalse(v) => { out.op(Opcode::JumpIfFalse); out.word(v * 4); }, IrOp::Jump(v) => { out.op(Opcode::Jump); out.word(v * 4); }, IrOp::JumpIfFalseKeep(v) => { out.op(Opcode::JumpIfFalseKeep); out.word(v * 4); }, IrOp::JumpIfTrueKeep(v) => { out.op(Opcode::JumpIfTrueKeep); out.word(v * 4); },
            IrOp::CallMethod(a,b) => { out.op(Opcode::CallMethod); out.word(a); out.word(b * 4); }, IrOp::CallCurrentMethod(v) => { out.op(Opcode::CallCurrentMethod); out.word(v * 4); }, IrOp::CallModule(a,b) => { out.op(Opcode::CallModule); out.word(a); let x=out.constant(Constant::String(b)); out.word(x as usize); },
            IrOp::Return => out.op(Opcode::Return), IrOp::Print => out.op(Opcode::Print), IrOp::Printf(v) => { out.op(Opcode::Printf); out.word(v); }, IrOp::Putc => out.op(Opcode::Putc),
        } while out.words.len() % 4 != 0 { out.words.push(0); } } out
    }
    fn decode(&self, mut pc: usize) -> Result<(DecodedOp<'_>, usize), Error> {
        let instruction_start = pc;
        let opcode=Opcode::from_word(*self.words.get(pc).ok_or_else(|| Error::Runtime("truncated bytecode".into()))?)?; pc+=1;
        let word=|pc: &mut usize| -> Result<usize, Error> { let v=*self.words.get(*pc).ok_or_else(|| Error::Runtime("truncated bytecode operand".into()))? as usize; *pc+=1; Ok(v) };
        let constant=|index: usize| -> Result<&Constant, Error> { self.constants.get(index).ok_or_else(|| Error::Runtime("invalid bytecode constant".into())) };
        macro_rules! c {
            (Constant::$kind:ident($name:ident)) => {
                let $name = {
                    let i = word(&mut pc)?;
                    match constant(i)? {
                        Constant::$kind(value) => value,
                        _ => return Err(Error::Runtime("invalid bytecode constant kind".into())),
                    }
                };
            };
        }
        let decoded=match opcode {
            Opcode::AddI32=>DecodedOp::AddI32, Opcode::AddF32=>DecodedOp::AddF32, Opcode::AddF64=>DecodedOp::AddF64,
            Opcode::Push=>{ c!(Constant::Value(v)); DecodedOp::Push(v) }, Opcode::MakeString=>{ c!(Constant::String(v)); DecodedOp::MakeString(v) }, Opcode::Input=>{ c!(Constant::Type(v)); DecodedOp::Input(v) }, Opcode::Require=>{ c!(Constant::Module(v)); DecodedOp::Require(v) }, Opcode::Load=>DecodedOp::Load(word(&mut pc)?), Opcode::LoadCurrentReceiver=>DecodedOp::LoadCurrentReceiver,
            Opcode::LoadCurrentField=>{ c!(Constant::Field(v)); DecodedOp::LoadCurrentField(v) }, Opcode::Store=>DecodedOp::Store(word(&mut pc)?), Opcode::StoreIndex=>{ let a=word(&mut pc)?; c!(Constant::Type(v)); DecodedOp::StoreIndex(a,v) }, Opcode::StoreTableIndex=>{ let a=word(&mut pc)?; c!(Constant::Type(v)); DecodedOp::StoreTableIndex(a,v) }, Opcode::StoreTensorIndex=>{ let a=word(&mut pc)?; c!(Constant::Type(v)); let b=word(&mut pc)?; DecodedOp::StoreTensorIndex(a,v,b) }, Opcode::StoreTensorIndexF32=>DecodedOp::StoreTensorIndexF32(word(&mut pc)?,word(&mut pc)?),
            Opcode::StoreField=>{let a=word(&mut pc)?;c!(Constant::Field(v));DecodedOp::StoreField(a,v)}, Opcode::StoreFieldIndex=>{let a=word(&mut pc)?;c!(Constant::Field(b));c!(Constant::Type(c));DecodedOp::StoreFieldIndex(a,b,c)}, Opcode::StoreTableField=>{let a=word(&mut pc)?;c!(Constant::String(b));c!(Constant::Type(c));DecodedOp::StoreTableField(a,b,c)}, Opcode::StoreCurrentField=>{c!(Constant::Field(v));DecodedOp::StoreCurrentField(v)},
            Opcode::MakeArray=>{let a=word(&mut pc)?;c!(Constant::Type(b));DecodedOp::MakeArray(a,b)}, Opcode::MakeTable=>{c!(Constant::Entries(a));c!(Constant::Type(b));DecodedOp::MakeTable(a,b)}, Opcode::MakeStruct=>{c!(Constant::Layout(v));DecodedOp::MakeStruct(v)}, Opcode::MakeTensor=>{c!(Constant::TensorInit(a));c!(Constant::Type(b));let c=word(&mut pc)?;DecodedOp::MakeTensor(*a,b,c)},
            Opcode::Index=>DecodedOp::Index, Opcode::TensorIndex=>{c!(Constant::Type(a));let b=word(&mut pc)?;DecodedOp::TensorIndex(a,b)}, Opcode::TensorIndexF32=>DecodedOp::TensorIndexF32(word(&mut pc)?), Opcode::TableIndex=>DecodedOp::TableIndex, Opcode::TableKeys=>DecodedOp::TableKeys, Opcode::TableKeysIndex=>DecodedOp::TableKeysIndex, Opcode::TableRemove=>DecodedOp::TableRemove, Opcode::Field=>{c!(Constant::Field(v));DecodedOp::Field(v)}, Opcode::TableField=>{c!(Constant::String(v));DecodedOp::TableField(v)}, Opcode::ModuleField=>{c!(Constant::String(v));DecodedOp::ModuleField(v)},
            Opcode::Binary=>{c!(Constant::Binary(v));DecodedOp::Binary(v)}, Opcode::Unary=>{c!(Constant::Unary(a));c!(Constant::Type(b));DecodedOp::Unary(a,b)}, Opcode::Len=>DecodedOp::Len, Opcode::ConcatString=>DecodedOp::ConcatString, Opcode::Cast=>{c!(Constant::Type(v));DecodedOp::Cast(v)}, Opcode::Builtin1=>{c!(Constant::Builtin(a));c!(Constant::Type(b));DecodedOp::Builtin1(*a,b)}, Opcode::Builtin2=>{c!(Constant::Builtin(a));c!(Constant::Type(b));DecodedOp::Builtin2(*a,b)}, Opcode::CallExternal=>{c!(Constant::String(a));let b=word(&mut pc)?;DecodedOp::CallExternal(a,b)},
            Opcode::JumpIfFalse=>DecodedOp::JumpIfFalse(word(&mut pc)?), Opcode::Jump=>DecodedOp::Jump(word(&mut pc)?), Opcode::JumpIfFalseKeep=>DecodedOp::JumpIfFalseKeep(word(&mut pc)?), Opcode::JumpIfTrueKeep=>DecodedOp::JumpIfTrueKeep(word(&mut pc)?), Opcode::CallMethod=>DecodedOp::CallMethod(word(&mut pc)?,word(&mut pc)?), Opcode::CallCurrentMethod=>DecodedOp::CallCurrentMethod(word(&mut pc)?), Opcode::CallModule=>{let a=word(&mut pc)?;c!(Constant::String(b));DecodedOp::CallModule(a,b)}, Opcode::Return=>DecodedOp::Return, Opcode::Print=>DecodedOp::Print, Opcode::Printf=>DecodedOp::Printf(word(&mut pc)?), Opcode::Putc=>DecodedOp::Putc,
        }; Ok((decoded,instruction_start + 4))
    }

    fn decode_ir(&self, pc: usize) -> Result<(IrOp, usize), Error> {
        let (op, next_pc) = self.decode(pc)?;
        let op = match op {
            DecodedOp::AddI32 => IrOp::AddI32, DecodedOp::AddF32 => IrOp::AddF32, DecodedOp::AddF64 => IrOp::AddF64,
            DecodedOp::Push(v) => IrOp::Push(v.clone()), DecodedOp::MakeString(v) => IrOp::MakeString(Rc::from(v)), DecodedOp::Input(v) => IrOp::Input(Rc::new(v.clone())), DecodedOp::Require(v) => IrOp::Require(Rc::new(v.clone())), DecodedOp::Load(v) => IrOp::Load(v), DecodedOp::LoadCurrentReceiver => IrOp::LoadCurrentReceiver,
            DecodedOp::LoadCurrentField(v) => IrOp::LoadCurrentField(Rc::new(v.clone())), DecodedOp::Store(v) => IrOp::Store(v), DecodedOp::StoreIndex(a,b) => IrOp::StoreIndex(a,Rc::new(b.clone())), DecodedOp::StoreTableIndex(a,b) => IrOp::StoreTableIndex(a,Rc::new(b.clone())), DecodedOp::StoreTensorIndex(a,b,c) => IrOp::StoreTensorIndex(a,Rc::new(b.clone()),c), DecodedOp::StoreTensorIndexF32(a,b) => IrOp::StoreTensorIndexF32(a,b),
            DecodedOp::StoreField(a,b) => IrOp::StoreField(a,Rc::new(b.clone())), DecodedOp::StoreFieldIndex(a,b,c) => IrOp::StoreFieldIndex(a,Rc::new(b.clone()),Rc::new(c.clone())), DecodedOp::StoreTableField(a,b,c) => IrOp::StoreTableField(a,Rc::from(b),Rc::new(c.clone())), DecodedOp::StoreCurrentField(v) => IrOp::StoreCurrentField(Rc::new(v.clone())),
            DecodedOp::MakeArray(a,b) => IrOp::MakeArray(a,Rc::new(b.clone())), DecodedOp::MakeTable(a,b) => IrOp::MakeTable(Rc::from(a),Rc::new(b.clone())), DecodedOp::MakeStruct(v) => IrOp::MakeStruct(Rc::new(v.clone())), DecodedOp::MakeTensor(a,b,c) => IrOp::MakeTensor(a,Rc::new(b.clone()),c),
            DecodedOp::Index => IrOp::Index, DecodedOp::TensorIndex(a,b) => IrOp::TensorIndex(Rc::new(a.clone()),b), DecodedOp::TensorIndexF32(v) => IrOp::TensorIndexF32(v), DecodedOp::TableIndex => IrOp::TableIndex, DecodedOp::TableKeys => IrOp::TableKeys, DecodedOp::TableKeysIndex => IrOp::TableKeysIndex, DecodedOp::TableRemove => IrOp::TableRemove, DecodedOp::Field(v) => IrOp::Field(Rc::new(v.clone())), DecodedOp::TableField(v) => IrOp::TableField(Rc::from(v)), DecodedOp::ModuleField(v) => IrOp::ModuleField(Rc::from(v)),
            DecodedOp::Binary(v) => IrOp::Binary(v.clone()), DecodedOp::Unary(a,b) => IrOp::Unary(a.clone(),Rc::new(b.clone())), DecodedOp::Len => IrOp::Len, DecodedOp::ConcatString => IrOp::ConcatString, DecodedOp::Cast(v) => IrOp::Cast(Rc::new(v.clone())), DecodedOp::Builtin1(a,b) => IrOp::Builtin1(a,Rc::new(b.clone())), DecodedOp::Builtin2(a,b) => IrOp::Builtin2(a,Rc::new(b.clone())), DecodedOp::CallExternal(a,b) => IrOp::CallExternal(Rc::from(a),b),
            DecodedOp::JumpIfFalse(v) => IrOp::JumpIfFalse(v), DecodedOp::Jump(v) => IrOp::Jump(v), DecodedOp::JumpIfFalseKeep(v) => IrOp::JumpIfFalseKeep(v), DecodedOp::JumpIfTrueKeep(v) => IrOp::JumpIfTrueKeep(v), DecodedOp::CallMethod(a,b) => IrOp::CallMethod(a,b), DecodedOp::CallCurrentMethod(v) => IrOp::CallCurrentMethod(v), DecodedOp::CallModule(a,b) => IrOp::CallModule(a,Rc::from(b)), DecodedOp::Return => IrOp::Return, DecodedOp::Print => IrOp::Print, DecodedOp::Printf(v) => IrOp::Printf(v), DecodedOp::Putc => IrOp::Putc,
        }; Ok((op, next_pc))
    }
}

/// Built-ins are resolved during compilation. Keeping their identity in the
/// opcode avoids retaining and repeatedly comparing function-name strings in
/// the VM's hot loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuiltinFn {
    Sqrt, Sin, Cos, Tan, Asin, Acos, Atan, Floor, Ceil, Round, Abs,
    Pow, Min, Max, Atan2,
}

impl BuiltinFn {
    fn unary(name: &str) -> Option<Self> {
        Some(match name {
            "sqrt" => Self::Sqrt, "sin" => Self::Sin, "cos" => Self::Cos,
            "tan" => Self::Tan, "asin" => Self::Asin, "acos" => Self::Acos,
            "atan" => Self::Atan, "floor" => Self::Floor, "ceil" => Self::Ceil,
            "round" => Self::Round, "abs" => Self::Abs,
            _ => return None,
        })
    }

    fn binary(name: &str) -> Option<Self> {
        Some(match name {
            "pow" => Self::Pow, "min" => Self::Min, "max" => Self::Max,
            "atan2" => Self::Atan2,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum TensorInit { Zeros, Random }

struct LoopContext { break_jumps: Vec<usize>, continue_jumps: Vec<usize>, continue_target: usize }

struct Compiler {
    names: HashMap<String, (usize, Type)>, structs: HashMap<String, StructLayout>,
    methods: HashMap<(String, String), Option<usize>>, pending_method_calls: Vec<(usize, String, String)>,
    current_method_fields: Option<HashMap<String, StructField>>, current_method_struct: Option<String>,
    module_root: Option<PathBuf>, module_artifacts: HashMap<String, ModuleArtifact>,
    compiling_modules: Rc<RefCell<HashSet<String>>>,
    exports: HashMap<String, ModuleExport>, extern_functions: HashMap<String, HostSignature>, code: Vec<Op>,
    interned_names: HashMap<String, Rc<str>>,
    strings: StringInterner,
    next_slot: usize, loops: Vec<LoopContext>
}

impl Default for Compiler { fn default() -> Self { Self { names: HashMap::new(), structs: HashMap::new(), methods: HashMap::new(), pending_method_calls: Vec::new(), current_method_fields: None, current_method_struct: None, module_root: None, module_artifacts: HashMap::new(), compiling_modules: Rc::new(RefCell::new(HashSet::new())), exports: HashMap::new(), extern_functions: HashMap::new(), code: Vec::new(), interned_names: HashMap::new(), strings: StringInterner::new(), next_slot: 0, loops: Vec::new() } } }

impl Compiler {
    fn with_module_root(module_root: PathBuf) -> Self { Self { module_root: Some(module_root), ..Self::default() } }
    fn with_extern_functions(extern_functions: HashMap<String, HostSignature>) -> Self { Self { extern_functions, ..Self::default() } }
    fn with_strings(mut self, strings: StringInterner) -> Self { self.strings = strings; self }
    fn string(&self, id: StringId) -> &str { self.strings.resolve(id) }

    fn intern_name(&mut self, name: &str) -> Rc<str> {
        if let Some(interned) = self.interned_names.get(name) { return interned.clone(); }
        let interned: Rc<str> = Rc::from(name);
        self.interned_names.insert(name.to_owned(), interned.clone());
        interned
    }

    /// Compile a lexical block and then discard bindings and slots introduced
    /// inside it. Slots are reused by later blocks.
    fn scoped_block(&mut self, body: Vec<Statement>) -> Result<(), Error> {
        let saved_names = self.names.clone();
        let saved_next_slot = self.next_slot;
        let result = body.into_iter().try_for_each(|statement| self.statement(statement));
        self.names = saved_names;
        self.next_slot = saved_next_slot;
        result
    }

    fn binary_opcode(op: BinOp, ty: &Type) -> Result<BinaryOp, Error> {
        match op {
            BinOp::Eq => Ok(BinaryOp::Equal),
            BinOp::Neq => Ok(BinaryOp::NotEqual),
            _ => match ty {
                Type::I8 => Ok(BinaryOp::I8(op)), Type::I16 => Ok(BinaryOp::I16(op)),
                Type::I32 => Ok(BinaryOp::I32(op)), Type::I64 => Ok(BinaryOp::I64(op)),
                Type::U8 => Ok(BinaryOp::U8(op)), Type::U16 => Ok(BinaryOp::U16(op)),
                Type::U32 => Ok(BinaryOp::U32(op)), Type::U64 => Ok(BinaryOp::U64(op)),
                Type::F16 => Ok(BinaryOp::F16(op)), Type::F32 => Ok(BinaryOp::F32(op)),
                Type::F64 => Ok(BinaryOp::F64(op)),
                _ => Err(Error::Type(format!("unsupported binary operand type {ty}"))),
            },
        }
    }

    fn compile(mut self, program: Vec<Statement>) -> Result<FlatBytecode, Error> { self.compile_program(program)?; Ok(FlatBytecode::lower(self.code)) }

    fn compile_module(mut self, id: String, program: Vec<Statement>) -> Result<ModuleArtifact, Error> {
        self.compile_program(program)?;
        // Entries are initially compiler-IR indices; each executable record is
        // four u32 words in the final stream.
        for export in self.exports.values_mut() {
            if let ModuleExport::Function { entry } = export { *entry *= 4; }
        }
        Ok(ModuleArtifact { id, code: Rc::new(FlatBytecode::lower(self.code)), exports: self.exports })
    }

    fn compile_program(&mut self, program: Vec<Statement>) -> Result<(), Error> {
        for s in program { self.statement(s)?; }
        let pending_calls = std::mem::take(&mut self.pending_method_calls);
        for (position, struct_name, method) in pending_calls {
            let target = self.methods.get(&(struct_name.clone(), method.clone())).copied().flatten().ok_or_else(|| Error::Type(format!("struct '{struct_name}' has no defined method '{method}'")))?;
            match self.code.get_mut(position) {
                Some(Op::CallMethod(_, call_target)) | Some(Op::CallCurrentMethod(call_target)) => *call_target = target,
                _ => return Err(Error::Runtime("invalid pending method call".into())),
            }
        }
        Ok(())
    }

    fn load_module(&mut self, requested: &str) -> Result<ModuleArtifact, Error> {
        let root = self.module_root.as_ref().ok_or_else(|| Error::Type("require is available only when executing an .l0 file".into()))?;
        let relative = Path::new(requested);
        if relative.is_absolute() { return Err(Error::Type("module path must be relative".into())); }
        let candidate = if relative.extension().is_some() { root.join(relative) } else { root.join(relative).with_extension("l0") };
        let canonical = fs::canonicalize(&candidate).map_err(|error| Error::Runtime(format!("cannot load module '{requested}': {error}")))?;
        if !canonical.starts_with(root) { return Err(Error::Type(format!("module '{requested}' escapes the module root"))); }
        let id = canonical.to_string_lossy().into_owned();
        if let Some(module) = self.module_artifacts.get(&id) { return Ok(module.clone()); }
        if !self.compiling_modules.borrow_mut().insert(id.clone()) {
            return Err(Error::Type(format!("cyclic module import involving '{requested}'")));
        }
        let module = (|| {
            let source = fs::read_to_string(&canonical).map_err(|error| Error::Runtime(format!("cannot read module '{requested}': {error}")))?;
            let (program, strings) = Parser::new(lex(&source)?).into_program()?;
            let mut module_compiler = Compiler::with_module_root(root.to_path_buf()).with_strings(strings);
            module_compiler.compiling_modules = self.compiling_modules.clone();
            module_compiler.extern_functions = self.extern_functions.clone();
            module_compiler.compile_module(id.clone(), program)
        })();
        self.compiling_modules.borrow_mut().remove(&id);
        let module = module?;
        self.module_artifacts.insert(id, module.clone());
        Ok(module)
    }

    fn import_exported_structs(&mut self, binding: &str, module_id: &str) -> Result<(), Error> {
        let module = self.module_artifacts.get(module_id).ok_or_else(|| Error::Runtime("missing compiled module".into()))?.clone();
        for (name, export) in module.exports { if let ModuleExport::Struct(mut layout) = export { let alias = format!("{binding}.{name}"); if self.structs.contains_key(&alias) { return Err(Error::Type(format!("imported struct '{alias}' conflicts with an existing struct"))); } layout.name = alias.clone(); self.structs.insert(alias, layout); } }
        Ok(())
    }
    fn compile_module_function(&mut self, name: String, body: Vec<Statement>) -> Result<(), Error> {
        if self.exports.contains_key(&name) { return Err(Error::Type(format!("module already exports '{name}'"))); }
        let skip_body = self.code.len(); self.code.push(Op::Jump(usize::MAX)); let entry = self.code.len();
        self.scoped_block(body)?;
        self.code.push(Op::Return); let after_body = self.code.len(); self.code[skip_body] = Op::Jump(after_body);
        self.exports.insert(name, ModuleExport::Function { entry }); Ok(())
    }
    fn compile_method_body(&mut self, struct_name: &str, method_name: &str, args: Vec<(String, Type)>, body: Vec<Statement>) -> Result<(), Error> {
        let key = (struct_name.to_owned(), method_name.to_owned());
        match self.methods.get(&key) {
            Some(None) => {},
            Some(Some(_)) => return Err(Error::Type(format!("struct '{struct_name}' already has a body for method '{method_name}'"))),
            None => return Err(Error::Type(format!("struct '{struct_name}' has no declared method '{method_name}'"))),
        }
        let layout = self.structs.get(struct_name).cloned().ok_or_else(|| Error::Type(format!("unknown struct '{struct_name}'")))?;
        let skip_body = self.code.len();
        self.code.push(Op::Jump(usize::MAX));
        let entry = self.code.len();
        self.methods.insert(key, Some(entry));
        let method_fields = layout.fields.iter().cloned().map(|field| (field.name.clone(), field)).collect();
        let previous_fields = self.current_method_fields.replace(method_fields);
        let previous_struct = self.current_method_struct.replace(struct_name.to_owned());
        let saved_names = self.names.clone();
        let saved_next_slot = self.next_slot;
        // Read the arguments and bind them to local variable slots
        for (arg_name, arg_ty) in args.into_iter().rev() {
            let slot = self.next_slot;
            self.next_slot += 1;
            self.names.insert(arg_name, (slot, arg_ty));
            self.code.push(Op::Store(slot));
        }
        let body_result = body.into_iter().try_for_each(|statement| self.statement(statement));
        self.current_method_fields = previous_fields;
        self.current_method_struct = previous_struct;
        self.names = saved_names;
        self.next_slot = saved_next_slot;
        body_result?;
        self.code.push(Op::Return);
        let after_body = self.code.len();
        self.code[skip_body] = Op::Jump(after_body);
        Ok(())
    }
    fn statement(&mut self, stmt: Statement) -> Result<(), Error> { match stmt {
        Statement::Located { node, location } => return self.statement(*node).map_err(|error| error.at(location)),
        Statement::Struct { name, fields, methods } => {
            if self.structs.contains_key(&name) {
                return Err(Error::Type(format!("struct '{name}' is already defined")));
            }
            let mut layout_fields = Vec::with_capacity(fields.len());
            for (index, (field_name, ty)) in fields.into_iter().enumerate() {
                if layout_fields.iter().any(|field: &StructField| field.name == field_name) {
                    return Err(Error::Type(format!("struct '{name}' defines field '{field_name}' more than once")));
                }
                layout_fields.push(StructField { name: field_name, ty, index });
            }
            self.structs.insert(name.clone(), StructLayout { name: name.clone(), fields: layout_fields });
            for method in &methods {
                let key = (name.clone(), method.name.clone());
                if self.methods.insert(key, None).is_some() {
                    return Err(Error::Type(format!("struct '{name}' defines method '{}' more than once", method.name)));
                }
            }
            for method in methods {
                if let Some(body) = method.body {
                    self.compile_method_body(&name, &method.name, method.args, body)?;
                }
            }
            Ok(())
        },
        Statement::MethodDefinition { struct_name, method, args, body } => {
            self.compile_method_body(&struct_name, &method, args, body)
        },
        Statement::ExportLet { name, ty, expr } => { let name = self.string(name).to_owned(); let id = self.strings.intern(&name); self.statement(Statement::Let { name: id, ty, expr })?; let (slot, ty) = self.names.get(&name).cloned().ok_or_else(|| Error::Runtime("missing exported local".into()))?; if self.exports.insert(name.clone(), ModuleExport::Value { slot, ty }).is_some() { return Err(Error::Type(format!("module already exports '{name}'"))); } Ok(()) },
        Statement::ExportStruct { name, fields, methods } => { self.statement(Statement::Struct { name: name.clone(), fields, methods })?; let layout = self.structs.get(&name).cloned().ok_or_else(|| Error::Runtime("missing exported struct".into()))?; if self.exports.insert(name.clone(), ModuleExport::Struct(layout)).is_some() { return Err(Error::Type(format!("module already exports '{name}'"))); } Ok(()) },
        Statement::ExportFunction { name, body } => self.compile_module_function(name, body),
        Statement::CallMethod { receiver, method } => {
            let (struct_name, receiver_slot) = match receiver {
                MethodReceiver::Name(name) => {
                    let (slot, ty) = self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?;
                    match ty {
                        Type::Struct(struct_name) => (struct_name, Some(slot)),
                        Type::Module(module_id) => {
                            let module = self.module_artifacts.get(&module_id).ok_or_else(|| Error::Runtime("missing compiled module".into()))?;
                            if !matches!(module.exports.get(&method), Some(ModuleExport::Function { .. })) { return Err(Error::Type(format!("module has no exported function '{method}'"))); }
                            self.code.push(Op::CallModule(slot, Rc::from(method))); return Ok(());
                        },
                        _ => return Err(Error::Type(format!("'{name}' is not a struct or module"))),
                    }
                },
                MethodReceiver::This => (self.current_method_struct.clone().ok_or_else(|| Error::Type("this is available only inside a struct method".into()))?, None),
            };
            let key = (struct_name.clone(), method.clone());
            let target_opt = self.methods.get(&key).copied().flatten();
            let target = target_opt.unwrap_or(usize::MAX);
            let position = self.code.len();
            self.code.push(match receiver_slot { Some(slot) => Op::CallMethod(slot, target), None => Op::CallCurrentMethod(target) });
            if target == usize::MAX { self.pending_method_calls.push((position, struct_name, method)); }
            Ok(())
        },
        Statement::Let { name, ty, expr } => {
            let name = self.string(name).to_owned();
            let found = self.expr(expr, Some(&ty))?;
            if !types_compatible(&ty, &found) { return Err(Error::Type(format!("'{name}' declared {ty}, but expression has type {found}"))); }

            let slot = if let Some((existing_slot, existing_ty)) = self.names.get(&name) {
                if existing_ty != &ty { return Err(Error::Type(format!("cannot redefine '{name}' with a different type"))); }
                *existing_slot
            } else {
                let s = self.next_slot;
                self.next_slot += 1;
                self.names.insert(name.clone(), (s, found.clone()));
                s
            };
            if let Type::Module(module_id) = &found { self.import_exported_structs(&name, module_id)?; }
            self.code.push(Op::Store(slot));
            Ok(())
        },
        Statement::Assign { name, expr } => { if let Some(field) = self.current_method_fields.as_ref().and_then(|fields| fields.get(&name)).cloned() { let found = self.expr(expr, Some(&field.ty))?; if found != field.ty { return Err(Error::Type(format!("field '{name}' is {}, but expression has type {found}", field.ty))); } self.code.push(Op::StoreCurrentField(Rc::new(field))); Ok(()) } else { let (slot, ty) = self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?; let found = self.expr(expr, Some(&ty))?; if found != ty { return Err(Error::Type(format!("'{name}' is {ty}, but expression has type {found}"))); } self.code.push(Op::Store(slot)); Ok(()) } },
        Statement::SetIndex { name, indices, expr } => {
            let (slot, container_ty) = self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?;
            match container_ty {
                Type::Array(inner) => {
                    if indices.len() != 1 { return Err(Error::Type("vector indexing requires exactly one index".into())); }
                    scalar_size(&inner)?; self.compile_tensor_indices(indices)?;
                    let element = *inner; let found = self.expr(expr, Some(&element))?;
                    if found != element { return Err(Error::Type(format!("item is {found}; expected {element}"))); }
                    self.code.push(Op::StoreIndex(slot, Rc::new(element))); Ok(())
                },
                Type::Table(inner) => {
                    if indices.len() != 1 { return Err(Error::Type("table indexing requires exactly one index".into())); }
                    self.compile_table_key(indices.into_iter().next().expect("checked table index count"))?;
                    let element = *inner; let found = self.expr(expr, Some(&element))?;
                    if found != element { return Err(Error::Type(format!("item is {found}; expected {element}"))); }
                    self.code.push(Op::StoreTableIndex(slot, Rc::new(element))); Ok(())
                },
                Type::Tensor(inner, rank) => {
                    if indices.len() != rank { return Err(Error::Type(format!("tensor rank {rank} requires {rank} index(es)"))); }
                    self.compile_tensor_indices(indices)?;
                    let element = *inner; let found = self.expr(expr, Some(&element))?;
                    if found != element { return Err(Error::Type(format!("tensor item is {found}; expected {element}"))); }
                    if element == Type::F32 { self.code.push(Op::StoreTensorIndexF32(slot, rank)); } else { self.code.push(Op::StoreTensorIndex(slot, Rc::new(element), rank)); } Ok(())
                },
                _ => Err(Error::Type(format!("'{name}' is not indexable"))),
            }
        },
        Statement::SetField { name, field, expr } => { let (slot, ty) = self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?; match ty { Type::Struct(struct_name) => { let layout = self.structs.get(&struct_name).ok_or_else(|| Error::Type(format!("unknown struct '{struct_name}'")))?; let field = layout.fields.iter().find(|candidate| candidate.name == field).cloned().ok_or_else(|| Error::Type(format!("struct '{struct_name}' has no field '{field}'")))?; let found = self.expr(expr, Some(&field.ty))?; if found != field.ty { return Err(Error::Type("struct field type mismatch".into())); } self.code.push(Op::StoreField(slot, Rc::new(field))); Ok(()) }, Type::Table(element) => { let element = *element; let found = self.expr(expr, Some(&element))?; if found != element { return Err(Error::Type("table value type mismatch".into())); } let field = self.intern_name(&field); self.code.push(Op::StoreTableField(slot, field, Rc::new(element))); Ok(()) }, _ => Err(Error::Type(format!("'{name}' has no named keys"))), } },
        Statement::SetFieldIndex { name, field, index, expr } => { let (slot, Type::Struct(struct_name)) = self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))? else { return Err(Error::Type(format!("'{name}' is not a struct"))); }; let layout = self.structs.get(&struct_name).ok_or_else(|| Error::Type(format!("unknown struct '{struct_name}'")))?; let field = layout.fields.iter().find(|candidate| candidate.name == field).cloned().ok_or_else(|| Error::Type(format!("struct '{struct_name}' has no field '{field}'")))?; let Type::Array(element) = field.ty.clone() else { return Err(Error::Type(format!("field '{}' is not a vector", field.name))); }; let element = *element; scalar_size(&element)?; let index_ty = self.expr(index, None)?; if !matches!(index_ty, Type::I8|Type::I16|Type::I32|Type::I64|Type::U8|Type::U16|Type::U32|Type::U64) { return Err(Error::Type("index must be an integer".into())); } let found = self.expr(expr, Some(&element))?; if found != element { return Err(Error::Type(format!("item is {found}; expected {element}"))); } self.code.push(Op::StoreFieldIndex(slot, Rc::new(field), Rc::new(element))); Ok(()) },
        Statement::Print(expr) => { self.expr(expr, None)?; self.code.push(Op::Print); Ok(()) },
        Statement::Printf { format, args } => {
            let num_args = args.len();
            for arg in args { self.expr(arg, None)?; }
            let fmt_ty = self.expr(format, Some(&Type::String))?;
            if fmt_ty != Type::String { return Err(Error::Type("printf format must be a string".into())); }
            self.code.push(Op::Printf(num_args));
            Ok(())
        },
        Statement::Putc(expr) => { self.expr(expr, None)?; self.code.push(Op::Putc); Ok(()) },
        Statement::If { condition, then_body, else_body } => { let ty = self.expr(condition, None)?; if ty != Type::Bool { return Err(Error::Type(format!("if condition must be bool, got {ty}"))); } let false_jump = self.code.len(); self.code.push(Op::JumpIfFalse(usize::MAX)); self.scoped_block(then_body)?; if else_body.is_empty() { let end = self.code.len(); self.code[false_jump] = Op::JumpIfFalse(end); } else { let end_jump = self.code.len(); self.code.push(Op::Jump(usize::MAX)); let else_start = self.code.len(); self.code[false_jump] = Op::JumpIfFalse(else_start); self.scoped_block(else_body)?; let end = self.code.len(); self.code[end_jump] = Op::Jump(end); } Ok(()) },
        Statement::While { condition, body } => {
            let loop_start = self.code.len();
            let ty = self.expr(condition, None)?;
            if ty != Type::Bool { return Err(Error::Type(format!("while condition must be bool, got {ty}"))); }
            let exit_jump = self.code.len();
            self.code.push(Op::JumpIfFalse(usize::MAX));
            self.loops.push(LoopContext { break_jumps: Vec::new(), continue_jumps: Vec::new(), continue_target: loop_start });
            self.scoped_block(body)?;
            self.code.push(Op::Jump(loop_start));
            let end = self.code.len();
            self.code[exit_jump] = Op::JumpIfFalse(end);
            let context = self.loops.pop().expect("loop context");
            for jump in context.break_jumps { self.code[jump] = Op::Jump(end); }
            for jump in context.continue_jumps { self.code[jump] = Op::Jump(context.continue_target); }
            Ok(())
        },
        Statement::For { name, start, end, body } => {
            if self.names.contains_key(&name) { return Err(Error::Type(format!("loop variable '{name}' is already defined"))); }
            let start_ty = self.expr(start, Some(&Type::I32))?;
            let end_ty = self.expr(end, Some(&Type::I32))?;
            if start_ty != Type::I32 || end_ty != Type::I32 { return Err(Error::Type("for bounds must be i32".into())); }
            let index_slot = self.next_slot;
            self.next_slot += 1;
            let end_slot = self.next_slot;
            self.next_slot += 1;
            // The expressions were emitted in source order, so store the end first.
            self.code.push(Op::Store(end_slot));
            self.code.push(Op::Store(index_slot));
            self.names.insert(name.clone(), (index_slot, Type::I32));
            let loop_start = self.code.len();
            self.code.push(Op::Load(index_slot));
            self.code.push(Op::Load(end_slot));
            self.code.push(Op::Binary(BinaryOp::I32(BinOp::Le)));
            let exit_jump = self.code.len();
            self.code.push(Op::JumpIfFalse(usize::MAX));
            self.loops.push(LoopContext { break_jumps: Vec::new(), continue_jumps: Vec::new(), continue_target: usize::MAX });
            self.scoped_block(body)?;
            let increment_start = self.code.len();
            self.code.push(Op::Load(index_slot));
            self.code.push(Op::Push(Value::I32(1)));
            self.code.push(Op::Binary(BinaryOp::I32(BinOp::Add)));
            self.code.push(Op::Store(index_slot));
            self.code.push(Op::Jump(loop_start));
            let loop_end = self.code.len();
            self.code[exit_jump] = Op::JumpIfFalse(loop_end);
            let mut context = self.loops.pop().expect("loop context");
            context.continue_target = increment_start;
            for jump in context.break_jumps { self.code[jump] = Op::Jump(loop_end); }
            for jump in context.continue_jumps { self.code[jump] = Op::Jump(context.continue_target); }
            self.names.remove(&name);
            self.next_slot = index_slot;
            Ok(())
        },
        Statement::Break => {
            let context = self.loops.last_mut().ok_or_else(|| Error::Type("break is available only inside a loop".into()))?;
            let jump = self.code.len();
            self.code.push(Op::Jump(usize::MAX));
            context.break_jumps.push(jump);
            Ok(())
        },
        Statement::Continue => {
            let context = self.loops.last_mut().ok_or_else(|| Error::Type("continue is available only inside a loop".into()))?;
            let jump = self.code.len();
            self.code.push(Op::Jump(usize::MAX));
            context.continue_jumps.push(jump);
            Ok(())
        },
    }
    }

    fn expr(&mut self, expr: Expr, expected: Option<&Type>) -> Result<Type, Error> { match expr {
        Expr::Located { node, location } => return self.expr(*node, expected).map_err(|error| error.at(location)),
        Expr::Integer(n) => { let ty = expected.unwrap_or(&Type::I32); let val = int_value(n, ty)?; self.code.push(Op::Push(val)); Ok(ty.clone()) },
        Expr::Float(n) => {
            let ty = expected.unwrap_or(&Type::F64);
            if !matches!(ty, Type::F16 | Type::F32 | Type::F64) {
                return Err(Error::Type(format!("float literal cannot initialize {ty}")));
            }
            let val = float_value(n, ty);
            self.code.push(Op::Push(val));
            Ok(ty.clone())
        },
        Expr::String(s) => { self.code.push(Op::MakeString(Rc::from(self.string(s)))); Ok(Type::String) },
        Expr::Input => { let ty = expected.filter(|t| is_numeric(t) || **t == Type::String).cloned().ok_or_else(|| Error::Type("input needs an expected scalar or string type, e.g. let value: i32 = input".into()))?; self.code.push(Op::Input(Rc::new(ty.clone()))); Ok(ty) },
        Expr::This => { let struct_name = self.current_method_struct.clone().ok_or_else(|| Error::Type("this is available only inside a struct method".into()))?; self.code.push(Op::LoadCurrentReceiver); Ok(Type::Struct(struct_name)) },
        Expr::Require(path) => { let module = self.load_module(&path)?; let id = module.id.clone(); self.code.push(Op::Require(Rc::new(module))); Ok(Type::Module(id)) },
        Expr::Name(name) => { let name = self.string(name); if let Some(field) = self.current_method_fields.as_ref().and_then(|fields| fields.get(name)).cloned() { let ty = field.ty.clone(); self.code.push(Op::LoadCurrentField(Rc::new(field))); Ok(ty) } else { let (slot, ty) = self.names.get(name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?; self.code.push(Op::Load(slot)); Ok(ty) } },
        Expr::Array(mut items) => {
            let inferred = expected.is_none();
            let element = match expected {
                Some(Type::Array(t)) => (**t).clone(),
                Some(other) => return Err(Error::Type(format!("expected {other}, but array literal creates a vector"))),
                None => {
                    let first = items.first().ok_or_else(|| Error::Type("cannot infer the element type of an empty array".into()))?.clone();
                    self.expr(first, None)?
                },
            };
            scalar_size(&element)?;
            let count = items.len();
            if inferred { items.remove(0); }
            for item in items {
                let found = self.expr(item, Some(&element))?;
                if found != element { return Err(Error::Type(format!("array item is {found}; expected {element}"))); }
            }
            self.code.push(Op::MakeArray(count, Rc::new(element.clone())));
            Ok(Type::Array(Box::new(element)))
        },
        Expr::Table(items) => { let element = match expected { Some(Type::Table(t)) => (**t).clone(), _ => return Err(Error::Type("table needs an explicit value type, e.g. table<i32>".into())) }; let mut entries = Vec::with_capacity(items.len()); for (key, value) in items { match key { TableLiteralKey::Name(name) => { let name = self.intern_name(&name); entries.push(TableEntry::Name(name)); }, TableLiteralKey::Index(index) => { self.compile_table_key(index)?; entries.push(TableEntry::Index); } } let found = self.expr(value, Some(&element))?; if found != element { return Err(Error::Type(format!("table value is {found}; expected {element}"))); } } self.code.push(Op::MakeTable(entries.into(), Rc::new(element.clone()))); Ok(Type::Table(Box::new(element))) },
        Expr::TensorFactory { name, element, shape } => {
            let Type::Tensor(expected_element, rank) = expected.cloned().ok_or_else(|| Error::Type(format!("{name}<T> needs an expected tensor type, e.g. let x: tensor<f32, 2> = {name}<f32>([2, 3])")))? else { return Err(Error::Type(format!("{name}<{}> creates a tensor, but the expected type is not tensor", element))); };
            if expected_element.as_ref() != &element { return Err(Error::Type(format!("{name} element type is {element}; expected {expected_element}"))); }
            let shape_ty = self.expr(*shape, Some(&Type::Array(Box::new(Type::U64))))?;
            if shape_ty != Type::Array(Box::new(Type::U64)) { return Err(Error::Type("tensor shape must be vector<u64>".into())); }
            let init = if name == "zeros" { TensorInit::Zeros } else { TensorInit::Random };
            if matches!(init, TensorInit::Random) && !matches!(element, Type::F16 | Type::F32 | Type::F64) { return Err(Error::Type("random<T> supports f16, f32, and f64 tensors only".into())); }
            self.code.push(Op::MakeTensor(init, Rc::new(element.clone()), rank));
            Ok(Type::Tensor(Box::new(element), rank))
        },
        Expr::StructLiteral(name, fields) => { let layout = self.structs.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown struct '{name}'")))?; if let Some(expected) = expected { if expected != &Type::Struct(name.clone()) { return Err(Error::Type(format!("expected {expected}, got {name}"))); } } if fields.len() != layout.fields.len() { return Err(Error::Type(format!("struct '{name}' needs {} field(s)", layout.fields.len()))); } let mut provided = HashMap::new(); for (field_name, field_expr) in fields { if provided.insert(field_name.clone(), field_expr).is_some() { return Err(Error::Type(format!("struct '{name}' initializes field '{field_name}' more than once"))); } } for field in &layout.fields { let expr = provided.remove(&field.name).ok_or_else(|| Error::Type(format!("struct '{name}' is missing field '{}'", field.name)))?; let found = self.expr(expr, Some(&field.ty))?; if found != field.ty { return Err(Error::Type(format!("field '{}' is {found}; expected {}", field.name, field.ty))); } } if let Some((unknown, _)) = provided.into_iter().next() { return Err(Error::Type(format!("struct '{name}' has no field '{unknown}'"))); } self.code.push(Op::MakeStruct(Rc::new(layout))); Ok(Type::Struct(name)) },
        Expr::Cast(inner, target_ty) => {
            let source_ty = self.expr(*inner, None)?;
            if !is_numeric(&source_ty) || !is_numeric(&target_ty) {
                return Err(Error::Type(format!("cannot cast {} to {}", source_ty, target_ty)));
            }
            self.code.push(Op::Cast(Rc::new(target_ty.clone())));
            Ok(target_ty)
        },
        Expr::Call(name, mut args) => {
            match name.as_str() {
                "len" => {
                    if args.len() != 1 { return Err(Error::Type("len expects 1 argument".into())); }
                    let ty = self.expr(args.remove(0), None)?;
                    if !matches!(ty, Type::Array(_) | Type::String | Type::Table(_) | Type::TableKeys | Type::Tensor(_, _)) { return Err(Error::Type(format!("len requires a vector, string, table, table_keys, or tensor, got {ty}"))); }
                    self.code.push(Op::Len);
                    Ok(Type::I32)
                },
                "keys" => {
                    if args.len() != 1 { return Err(Error::Type("keys expects 1 argument".into())); }
                    let ty = self.expr(args.remove(0), None)?;
                    if !matches!(ty, Type::Table(_)) { return Err(Error::Type(format!("keys requires a table, got {ty}"))); }
                    self.code.push(Op::TableKeys);
                    Ok(Type::TableKeys)
                },
                "remove" => {
                    if args.len() != 2 { return Err(Error::Type("remove expects 2 arguments".into())); }
                    let table_ty = self.expr(args.remove(0), None)?;
                    if !matches!(table_ty, Type::Table(_)) { return Err(Error::Type(format!("remove requires a table as its first argument, got {table_ty}"))); }
                    self.compile_table_key(args.remove(0))?;
                    self.code.push(Op::TableRemove);
                    Ok(Type::Bool)
                },
                "sqrt" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "floor" | "ceil" | "round" => {
                    if args.len() != 1 { return Err(Error::Type(format!("{} expects 1 argument", name))); }
                    let arg = args.remove(0);
                    let ty = self.expr(arg, expected)?;
                    if !matches!(ty, Type::F16 | Type::F32 | Type::F64) { return Err(Error::Type(format!("{} requires a float argument, got {}", name, ty))); }
                    self.code.push(Op::Builtin1(BuiltinFn::unary(&name).expect("known unary built-in"), Rc::new(ty.clone())));
                    Ok(ty)
                },
                "abs" => {
                    if args.len() != 1 { return Err(Error::Type("abs expects 1 argument".into())); }
                    let arg = args.remove(0);
                    let ty = self.expr(arg, expected)?;
                    if !is_numeric(&ty) { return Err(Error::Type("abs requires a numeric argument".into())); }
                    self.code.push(Op::Builtin1(BuiltinFn::unary(&name).expect("known unary built-in"), Rc::new(ty.clone())));
                    Ok(ty)
                },
                "pow" | "min" | "max" | "atan2" => {
                    if args.len() != 2 { return Err(Error::Type(format!("{} expects 2 arguments", name))); }
                    let arg2 = args.remove(1);
                    let arg1 = args.remove(0);
                    let t1 = self.expr(arg1, expected)?;
                    let t2 = self.expr(arg2, Some(&t1))?;
                    if !types_compatible(&t1, &t2) { return Err(Error::Type(format!("{} arguments must have same type", name))); }
                    if name == "atan2" && !matches!(t1, Type::F16 | Type::F32 | Type::F64) { return Err(Error::Type("atan2 requires float arguments".into())); }
                    self.code.push(Op::Builtin2(BuiltinFn::binary(&name).expect("known binary built-in"), Rc::new(t1.clone())));
                    Ok(t1)
                },
                _ => {
                    let signature = self.extern_functions.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown function '{name}'")))?;
                    if args.len() != signature.arguments.len() { return Err(Error::Type(format!("{name} expects {} argument(s)", signature.arguments.len()))); }
                    for (argument, expected_type) in args.into_iter().zip(signature.arguments.iter()) {
                        let found = self.expr(argument, Some(expected_type))?;
                        if found != *expected_type { return Err(Error::Type(format!("{name} argument is {found}; expected {expected_type}"))); }
                    }
                    self.code.push(Op::CallExternal(Rc::from(name), signature.arguments.len()));
                    Ok(signature.result)
                }
            }
        },
        Expr::Unary(op, inner) => {
            let ty = self.expr(*inner, expected)?;
            match op {
                UnOp::Neg => if !is_numeric(&ty) { return Err(Error::Type("'-' requires a numeric type".into())); },
                UnOp::Not => if ty != Type::Bool { return Err(Error::Type("'!' requires a bool".into())); },
            }
            self.code.push(Op::Unary(op, Rc::new(ty.clone())));
            Ok(ty)
        },
        Expr::Binary(left, op, right) => {
            if matches!(op, BinOp::And | BinOp::Or) {
                let lt = self.expr(*left, Some(&Type::Bool))?;
                if lt != Type::Bool { return Err(Error::Type("logical operators require bool".into())); }

                let jump_idx = self.code.len();
                if op == BinOp::And { // <-- Removed *
                    self.code.push(Op::JumpIfFalseKeep(usize::MAX));
                } else {
                    self.code.push(Op::JumpIfTrueKeep(usize::MAX));
                }

                let rt = self.expr(*right, Some(&Type::Bool))?;
                if rt != Type::Bool { return Err(Error::Type("logical operators require bool".into())); }

                let end_idx = self.code.len();
                if op == BinOp::And { // <-- Removed *
                    self.code[jump_idx] = Op::JumpIfFalseKeep(end_idx);
                } else {
                    self.code[jump_idx] = Op::JumpIfTrueKeep(end_idx);
                }
                return Ok(Type::Bool);
            }
            let left_expected = match op { BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => None, _ => expected };
            let lt = self.expr(*left, left_expected)?;
            let rt = self.expr(*right, Some(&lt))?;
            if !types_compatible(&lt, &rt) { return Err(Error::Type(format!("operator {:?} needs matching types, got {} and {}", op, lt, rt))); }
            match op {
                BinOp::Add if lt == Type::String => {
                    self.code.push(Op::ConcatString); Ok(Type::String)
                },
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                    if !is_numeric(&lt) { return Err(Error::Type("arithmetic requires numeric types".into())); }
                    match (op, &lt) {
                        (BinOp::Add, Type::I32) => self.code.push(Op::AddI32),
                        (BinOp::Add, Type::F32) => self.code.push(Op::AddF32),
                        (BinOp::Add, Type::F64) => self.code.push(Op::AddF64),
                        _ => self.code.push(Op::Binary(Self::binary_opcode(op, &lt)?)),
                    }
                    Ok(lt)
                },
                BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                    if !is_integer(&lt) { return Err(Error::Type("bitwise operations require integer types".into())); }
                    self.code.push(Op::Binary(Self::binary_opcode(op, &lt)?)); Ok(lt)
                },
                BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    if !is_numeric(&lt) { return Err(Error::Type("comparisons require numeric types".into())); }
                    self.code.push(Op::Binary(Self::binary_opcode(op, &lt)?)); Ok(Type::Bool)
                },
                BinOp::Eq | BinOp::Neq => {
                    self.code.push(Op::Binary(Self::binary_opcode(op, &lt)?)); Ok(Type::Bool)
                },
                BinOp::And | BinOp::Or => unreachable!(),
            }
        },
        Expr::Index(container, indices) => {
            let ct = self.expr(*container, None)?;
            match ct {
                Type::Array(element) => { if indices.len() != 1 { return Err(Error::Type("vector indexing requires exactly one index".into())); } self.compile_tensor_indices(indices)?; self.code.push(Op::Index); Ok(*element) },
                Type::Table(element) => { if indices.len() != 1 { return Err(Error::Type("table indexing requires exactly one index".into())); } self.compile_table_key(indices.into_iter().next().expect("checked table index count"))?; self.code.push(Op::TableIndex); Ok(*element) },
                Type::TableKeys => { if indices.len() != 1 { return Err(Error::Type("table_keys indexing requires exactly one index".into())); } self.compile_tensor_indices(indices)?; self.code.push(Op::TableKeysIndex); Ok(Type::TableKey) },
                Type::Tensor(element, rank) => { if indices.len() != rank { return Err(Error::Type(format!("tensor rank {rank} requires {rank} index(es)"))); } self.compile_tensor_indices(indices)?; if *element == Type::F32 { self.code.push(Op::TensorIndexF32(rank)); } else { self.code.push(Op::TensorIndex(Rc::new((*element).clone()), rank)); } Ok(*element) },
                _ => Err(Error::Type(format!("cannot index {ct}"))),
            }
        },
        Expr::Field(container, field_name) => { let container_ty = self.expr(*container, None)?; match container_ty { Type::Struct(struct_name) => { let layout = self.structs.get(&struct_name).ok_or_else(|| Error::Type(format!("unknown struct '{struct_name}'")))?; let field = layout.fields.iter().find(|field| field.name == field_name).cloned().ok_or_else(|| Error::Type(format!("struct '{struct_name}' has no field '{field_name}'")))?; self.code.push(Op::Field(Rc::new(field.clone()))); Ok(field.ty) }, Type::Table(element) => { let element = *element; let field_name = self.intern_name(&field_name); self.code.push(Op::TableField(field_name)); Ok(element) }, Type::Module(module_id) => { let ty = { let module = self.module_artifacts.get(&module_id).ok_or_else(|| Error::Runtime("missing compiled module".into()))?; let ModuleExport::Value { ty, .. } = module.exports.get(&field_name).ok_or_else(|| Error::Type(format!("module has no exported value '{field_name}'")))? else { return Err(Error::Type(format!("'{field_name}' is not an exported module value"))); }; ty.clone() }; self.code.push(Op::ModuleField(Rc::from(field_name))); Ok(ty) }, _ => Err(Error::Type("field access requires a struct, table, or module".into())), } },
    } }

    fn compile_tensor_indices(&mut self, indices: Vec<Expr>) -> Result<(), Error> {
        for index in indices {
            let index_ty = self.expr(index, None)?;
            if !is_integer(&index_ty) { return Err(Error::Type("tensor index must be an integer".into())); }
        }
        Ok(())
    }

    fn compile_table_key(&mut self, key: Expr) -> Result<(), Error> {
        let key_ty = self.expr(key, None)?;
        if is_integer(&key_ty) || matches!(key_ty, Type::String | Type::TableKey) { Ok(()) }
        else { Err(Error::Type("table key must be an integer or string".into())) }
    }
}

fn types_compatible(expected: &Type, found: &Type) -> bool { expected == found || matches!((expected, found), (Type::Module(expected), Type::Module(_)) if expected.is_empty()) }
fn is_numeric(t: &Type) -> bool { !matches!(t, Type::Bool|Type::String|Type::Array(_)|Type::Tensor(_, _)|Type::Table(_)|Type::TableKey|Type::TableKeys|Type::Struct(_)|Type::Module(_)) }
fn is_integer(t: &Type) -> bool { matches!(t, Type::I8|Type::I16|Type::I32|Type::I64|Type::U8|Type::U16|Type::U32|Type::U64) }

fn int_value(n: i128, ty: &Type) -> Result<Value, Error> {
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
fn float_value(n: f64, ty: &Type) -> Value { match ty { Type::F16=>Value::F16(f32_to_f16(n as f32)),Type::F32=>Value::F32(n as f32),_=>Value::F64(n) } }

pub type L0RustFunction = fn(&[Value], &RefCell<Heap>) -> Result<Value, Error>;

#[derive(Clone)]
enum ExternalFunction {
    Rust(L0RustFunction),
    C(L0CFunction),
}

#[derive(Clone)]
struct RegisteredExternal { signature: HostSignature, function: ExternalFunction }

struct FfiCall { arguments: Vec<Value>, results: Vec<Value> }

struct ModuleInstance { artifact: ModuleArtifact, vm: Vm }
pub struct Vm {
    stack: Vec<Value>,
    stack_ptr: usize,
    locals: Vec<Value>,
    output: Vec<String>,
    interactive: bool,
    input: VecDeque<String>,
    modules: HashMap<String, ModuleInstance>,
    extern_functions: HashMap<String, RegisteredExternal>,
    heap: Rc<RefCell<Heap>>,
    gc_owner: bool,
    callback_state: Option<*mut L0State>,
    random_state: u64,
}

impl Default for Vm {
    fn default() -> Self {
        Self { stack: vec![Value::Bool(false); 4096], stack_ptr: 0, locals: Vec::with_capacity(64), output: Vec::new(), interactive: false, input: VecDeque::new(), modules: HashMap::new(), extern_functions: HashMap::new(), heap: Rc::new(RefCell::new(Heap::default())), gc_owner: true, callback_state: None, random_state: 0x5EED_CAFE_D15C_A11E }
    }
}

impl Vm {
    fn with_shared_heap(heap: Rc<RefCell<Heap>>, extern_functions: HashMap<String, RegisteredExternal>, callback_state: Option<*mut L0State>) -> Self { Self { heap, extern_functions, gc_owner: false, callback_state, ..Self::default() } }

    // A VM is single-threaded and a heap is never accessed re-entrantly by two
    // VMs.  Module VMs share the allocation domain, hence the Rc<RefCell<_>> at
    // the embedding boundary; the interpreter itself uses these short-lived
    // references to avoid dynamic borrow checks for every memory opcode.
    #[inline(always)]
    fn heap_ref(&self) -> &Heap { unsafe { &*self.heap.as_ptr() } }

    #[inline(always)]
    fn heap_mut(&mut self) -> &mut Heap { unsafe { &mut *self.heap.as_ptr() } }

    pub fn register_rust_function(&mut self, name: impl Into<String>, arguments: Vec<Type>, result: Type, function: L0RustFunction) -> Result<(), Error> {
        self.register_external(name.into(), HostSignature { arguments, result }, ExternalFunction::Rust(function))
    }

    /// Register a C ABI callback that takes `i32` arguments and returns one `i32`.
    /// The callback reads arguments through `l0_to_i32` and pushes its result with
    /// `l0_push_i32`; zero means success and any non-zero status aborts the call.
    pub fn register_c_i32_function(&mut self, name: impl Into<String>, argument_count: usize, function: L0CFunction) -> Result<(), Error> {
        self.register_external(name.into(), HostSignature { arguments: vec![Type::I32; argument_count], result: Type::I32 }, ExternalFunction::C(function))
    }

    fn register_external(&mut self, name: String, signature: HostSignature, function: ExternalFunction) -> Result<(), Error> {
        if name.is_empty() { return Err(Error::Type("external function name cannot be empty".into())); }
        if self.extern_functions.insert(name.clone(), RegisteredExternal { signature, function }).is_some() { return Err(Error::Type(format!("external function '{name}' is already registered"))); }
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

    /// Queue one line for the next `input` expression. This makes input-driven
    /// programs deterministic in tests and embedding applications.
    pub fn push_input(&mut self, data: String) {
        self.input.push_back(data);
    }

    /// Execute a source file while retaining this VM's registered host
    /// functions. Relative `require` paths are resolved from the source file.
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

    /// Run a full collection. The return value is the number of reclaimed
    /// objects; it is public so the runtime can be integration-tested.
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
            Value::Tensor(reference, element, rank) => match self.heap_ref().get(*reference)? {
                HeapObject::Tensor { element: stored_element, shape, .. } if stored_element == element.as_ref() && shape.len() == *rank => Ok(format!("tensor<{}, {}>[{}]", element, rank, shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", "))),
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
            if value.ty() != *expected { return Err(Error::Runtime(format!("external function '{name}' received an invalid argument type"))); }
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
        if result.ty() != registered.signature.result { return Err(Error::Runtime(format!("external function '{name}' returned {}; expected {}", result.ty(), registered.signature.result))); }
        self.stack_ptr = base;
        self.push(result);
        Ok(())
    }

    fn run(&mut self, code: &FlatBytecode) -> Result<&[String], Error> { self.run_from(code, 0, false) }

    fn run_from(&mut self, code: &FlatBytecode, mut pc: usize, terminal_return: bool) -> Result<&[String], Error> {
        let mut call_stack: Vec<(usize, Option<usize>)> = Vec::new();
        let mut current_receiver: Option<usize> = None;

        while pc < code.words.len() {
            let (instruction, next_pc) = code.decode_ir(pc)?;
            match &instruction {
            Op::AddI32 => {
                // The compiler emits two operands for every arithmetic opcode.
                // Keep the diagnostic in debug builds, but do not branch twice
                // per instruction in the production arithmetic hot path.
                let right = self.pop_compiled();
                let left = self.pop_compiled();
                let (Value::I32(left), Value::I32(right)) = (left, right) else {
                    return Err(Error::Runtime("VM i32 addition invariant broken".into()));
                };
                let result = left.checked_add(right)
                    .ok_or_else(|| Error::Runtime("addition overflow".into()))?;
                self.push(Value::I32(result));
            },
            Op::AddF32 => {
                let right = self.pop_compiled();
                let left = self.pop_compiled();
                let (Value::F32(left), Value::F32(right)) = (left, right) else {
                    return Err(Error::Runtime("VM f32 addition invariant broken".into()));
                };
                self.push(Value::F32(left + right));
            },
            Op::AddF64 => {
                let right = self.pop_compiled();
                let left = self.pop_compiled();
                let (Value::F64(left), Value::F64(right)) = (left, right) else {
                    return Err(Error::Runtime("VM f64 addition invariant broken".into()));
                };
                self.push(Value::F64(left + right));
            },
            Op::Push(v) => self.push(v.clone()),
            Op::MakeString(s) => { let reference = self.allocate(HeapObject::String(s.to_string())); self.push(Value::String(reference)); self.collect_if_needed(); },
            Op::Input(ty) => { let val = self.read_input(ty)?; self.push(val); self.collect_if_needed(); },
            Op::Require(module) => self.load_module((**module).clone())?,
            Op::Load(slot) => self.push(self.locals.get(*slot).cloned().ok_or_else(|| Error::Runtime("invalid local slot".into()))?),
            Op::LoadCurrentReceiver => { let receiver = current_receiver.ok_or_else(|| Error::Runtime("this is available only inside a method".into()))?; self.push(self.locals.get(receiver).cloned().ok_or_else(|| Error::Runtime("invalid method receiver".into()))?); },
            Op::LoadCurrentField(field) => { let receiver = current_receiver.ok_or_else(|| Error::Runtime("field access outside method".into()))?; let Value::Struct(reference, _) = self.locals.get(receiver).ok_or_else(|| Error::Runtime("invalid method receiver".into()))? else { return Err(Error::Runtime("VM method receiver invariant broken".into())); }; let value = match self.heap_ref().get(*reference)? { HeapObject::Struct { values, .. } => values.get(field.index).cloned().ok_or_else(|| Error::Runtime("invalid struct field index".into()))?, _ => return Err(Error::Runtime("struct heap invariant broken".into())) }; self.push(value); },
            Op::Store(slot) => {
                let v = self.pop()?;
                if *slot >= self.locals.len() { self.locals.resize(*slot + 1, Value::Bool(false)); }
                self.locals[*slot] = v;
            },
            Op::StoreIndex(slot, element) => {
                let value = self.pop()?;
                if &value.ty() != element.as_ref() { return Err(Error::Runtime("VM type invariant broken".into())); }
                let index = integer_to_usize(&self.pop()?)?;
                let reference = match self.locals.get(*slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? {
                    Value::Array(reference, _) => *reference,
                    _ => return Err(Error::Runtime("VM array slot invariant broken".into())),
                };
                match self.heap_mut().get_mut(reference)? {
                    HeapObject::Array { bytes, element: stored_element } if stored_element == element.as_ref() => {
                        write_scalar(bytes, index, &value, element.as_ref())?;
                    },
                    _ => return Err(Error::Runtime("array heap invariant broken".into())),
                }
            },
            Op::StoreTableIndex(slot, element) => { let value = self.pop()?; if &value.ty() != element.as_ref() { return Err(Error::Runtime("VM table type invariant broken".into())); } let key_value = self.pop()?; let key = table_key_from_value(self.heap_ref(), &key_value)?; let reference = match self.locals.get(*slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? { Value::Table(reference, _) => *reference, _ => return Err(Error::Runtime("VM table slot invariant broken".into())), }; match self.heap_mut().get_mut(reference)? { HeapObject::Table { entries, element: stored_element } if stored_element == element.as_ref() => { entries.insert(key, value); }, _ => return Err(Error::Runtime("table heap invariant broken".into())), } },
            Op::StoreTensorIndex(slot, element, rank) => {
                let value = self.pop()?;
                if &value.ty() != element.as_ref() { return Err(Error::Runtime("VM tensor type invariant broken".into())); }
                let indices = self.pop_tensor_indices(*rank)?;
                let Value::Tensor(reference, stored_element, stored_rank) = self.locals.get(*slot).cloned().ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM tensor slot invariant broken".into())); };
                if stored_element.as_ref() != element.as_ref() || stored_rank != *rank { return Err(Error::Runtime("VM tensor slot type invariant broken".into())); }
                match self.heap_mut().get_mut(reference)? {
                    HeapObject::Tensor { bytes, element: stored_element, shape } if stored_element == element.as_ref() && shape.len() == *rank => {
                        let offset = Self::tensor_offset(shape, &indices)?;
                        write_scalar(bytes, offset, &value, element.as_ref())?;
                    },
                    _ => return Err(Error::Runtime("tensor heap invariant broken".into())),
                }
            },
            Op::StoreTensorIndexF32(slot, rank) => {
                let value = self.pop()?;
                if value.ty() != Type::F32 { return Err(Error::Runtime("VM tensor type invariant broken".into())); }
                let indices = self.pop_tensor_indices(*rank)?;
                let Value::Tensor(reference, _, _) = self.locals.get(*slot).cloned().ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM tensor slot invariant broken".into())); };
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
            Op::StoreField(slot, field) => { let new_value = self.pop()?; if &new_value.ty() != &field.ty { return Err(Error::Runtime("VM type invariant broken".into())); } let reference = match self.locals.get(*slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? { Value::Struct(reference, _) => *reference, _ => return Err(Error::Runtime("VM struct slot invariant broken".into())), }; match self.heap_mut().get_mut(reference)? { HeapObject::Struct { values, .. } => { *values.get_mut(field.index).ok_or_else(|| Error::Runtime("invalid struct field index".into()))? = new_value; }, _ => return Err(Error::Runtime("struct heap invariant broken".into())), } },
            Op::StoreFieldIndex(slot, field, element) => { let value = self.pop()?; if &value.ty() != element.as_ref() { return Err(Error::Runtime("VM vector type invariant broken".into())); } let index = integer_to_usize(&self.pop()?)?; let Value::Struct(struct_reference, _) = self.locals.get(*slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM struct slot invariant broken".into())); }; let array_reference = match self.heap_ref().get(*struct_reference)? { HeapObject::Struct { values, .. } => match values.get(field.index) { Some(Value::Array(reference, _)) => *reference, _ => return Err(Error::Runtime("VM struct vector field invariant broken".into())), }, _ => return Err(Error::Runtime("struct heap invariant broken".into())), }; match self.heap_mut().get_mut(array_reference)? { HeapObject::Array { bytes, element: stored_element } if stored_element == element.as_ref() => write_scalar(bytes, index, &value, element.as_ref())?, _ => return Err(Error::Runtime("array heap invariant broken".into())), } },
            Op::StoreTableField(slot, name, element) => { let value = self.pop()?; if &value.ty() != element.as_ref() { return Err(Error::Runtime("VM table type invariant broken".into())); } let reference = match self.locals.get(*slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? { Value::Table(reference, _) => *reference, _ => return Err(Error::Runtime("VM table slot invariant broken".into())), }; match self.heap_mut().get_mut(reference)? { HeapObject::Table { entries, element: stored_element } if stored_element == element.as_ref() => { entries.insert(TableKey::Name(name.clone()), value); }, _ => return Err(Error::Runtime("table heap invariant broken".into())), } },
            Op::StoreCurrentField(field) => { let new_value = self.pop()?; if &new_value.ty() != &field.ty { return Err(Error::Runtime("VM type invariant broken".into())); } let receiver = current_receiver.ok_or_else(|| Error::Runtime("field assignment outside method".into()))?; let reference = match self.locals.get(receiver).ok_or_else(|| Error::Runtime("invalid method receiver".into()))? { Value::Struct(reference, _) => *reference, _ => return Err(Error::Runtime("VM method receiver invariant broken".into())), }; match self.heap_mut().get_mut(reference)? { HeapObject::Struct { values, .. } => { *values.get_mut(field.index).ok_or_else(|| Error::Runtime("invalid struct field index".into()))? = new_value; }, _ => return Err(Error::Runtime("struct heap invariant broken".into())), } },
            Op::MakeArray(len, ty) => { if self.stack_ptr < *len { return Err(Error::Runtime("stack underflow".into())); } let at = self.stack_ptr - len; let values = self.stack[at..self.stack_ptr].to_vec(); self.stack_ptr = at; let bytes = Value::pack_array(values, ty.as_ref())?; let reference = self.allocate(HeapObject::Array { bytes, element: ty.as_ref().clone() }); self.push(Value::Array(reference, Box::new(ty.as_ref().clone()))); self.collect_if_needed(); },
            Op::MakeTensor(init, element, rank) => {
                let shape_value = self.pop()?;
                let shape = self.tensor_shape_from_value(&shape_value)?;
                if shape.len() != *rank { return Err(Error::Runtime(format!("tensor rank {rank} requires {rank} dimensions, got {}", shape.len()))); }
                let bytes = self.make_tensor_bytes(*init, element.as_ref(), &shape)?;
                let reference = self.allocate(HeapObject::Tensor { bytes, element: element.as_ref().clone(), shape });
                self.push(Value::Tensor(reference, Box::new(element.as_ref().clone()), *rank));
                self.collect_if_needed();
            },
            Op::MakeTable(entries, element) => { let value_count = entries.iter().map(|entry| match entry { TableEntry::Index => 2usize, TableEntry::Name(_) => 1usize }).sum(); if self.stack_ptr < value_count { return Err(Error::Runtime("stack underflow".into())); } let at = self.stack_ptr - value_count; let values = self.stack[at..self.stack_ptr].to_vec(); self.stack_ptr = at; let mut cursor = 0; let mut table = HashMap::with_capacity(entries.len()); for entry in entries.iter() { let key = match entry { TableEntry::Index => { let key = table_key_from_value(self.heap_ref(), values.get(cursor).ok_or_else(|| Error::Runtime("stack underflow".into()))?)?; cursor += 1; key }, TableEntry::Name(name) => TableKey::Name(name.clone()), }; let value = values.get(cursor).cloned().ok_or_else(|| Error::Runtime("stack underflow".into()))?; cursor += 1; if &value.ty() != element.as_ref() { return Err(Error::Runtime("VM table type invariant broken".into())); } if table.insert(key.clone(), value).is_some() { return Err(Error::Runtime(format!("table key {} is declared more than once", table_key_display(&key)))); } } let reference = self.allocate(HeapObject::Table { entries: table, element: element.as_ref().clone() }); self.push(Value::Table(reference, Box::new(element.as_ref().clone()))); self.collect_if_needed(); },
            Op::MakeStruct(layout) => { if self.stack_ptr < layout.fields.len() { return Err(Error::Runtime("stack underflow".into())); } let at = self.stack_ptr - layout.fields.len(); let values = self.stack[at..self.stack_ptr].to_vec(); self.stack_ptr = at; for (field, value) in layout.fields.iter().zip(values.iter()) { if value.ty() != field.ty { return Err(Error::Runtime("VM struct type invariant broken".into())); } } let reference = self.allocate(HeapObject::Struct { values, layout: layout.clone() }); self.push(Value::Struct(reference, layout.clone())); self.collect_if_needed(); },
            Op::Index => { let index = integer_to_usize(&self.pop()?)?; let object = self.pop()?; let Value::Array(reference, element) = object else { return Err(Error::Runtime("VM array invariant broken".into())); }; let value = match self.heap_ref().get(reference)? { HeapObject::Array { bytes, element: stored_element } if stored_element == element.as_ref() => decode_scalar(bytes, index, &element)?, _ => return Err(Error::Runtime("array heap invariant broken".into())) }; self.push(value); },
            Op::TensorIndex(element, rank) => {
                let indices = self.pop_tensor_indices(*rank)?;
                let object = self.pop()?;
                let Value::Tensor(reference, stored_element, stored_rank) = object else { return Err(Error::Runtime("VM tensor invariant broken".into())); };
                if stored_element.as_ref() != element.as_ref() || stored_rank != *rank { return Err(Error::Runtime("VM tensor type invariant broken".into())); }
                let value = match self.heap_ref().get(reference)? {
                    HeapObject::Tensor { bytes, element: stored_element, shape } if stored_element == element.as_ref() && shape.len() == *rank => decode_scalar(bytes, Self::tensor_offset(shape, &indices)?, element.as_ref())?,
                    _ => return Err(Error::Runtime("tensor heap invariant broken".into())),
                };
                self.push(value);
            },
            Op::TensorIndexF32(rank) => {
                let indices = self.pop_tensor_indices(*rank)?;
                let object = self.pop()?;
                let Value::Tensor(reference, _, _) = object else { return Err(Error::Runtime("VM tensor invariant broken".into())); };
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
            Op::TableIndex => { let key_value = self.pop()?; let key = table_key_from_value(self.heap_ref(), &key_value)?; let object = self.pop()?; let Value::Table(reference, _) = object else { return Err(Error::Runtime("VM table invariant broken".into())); }; let value = match self.heap_ref().get(reference)? { HeapObject::Table { entries, .. } => entries.get(&key).cloned().ok_or_else(|| Error::Runtime(format!("table has no key {}", table_key_display(&key))))?, _ => return Err(Error::Runtime("table heap invariant broken".into())) }; self.push(value); },
            Op::TableKeys => { let object = self.pop()?; let Value::Table(reference, _) = object else { return Err(Error::Runtime("VM table invariant broken".into())); }; let mut keys = match self.heap_ref().get(reference)? { HeapObject::Table { entries, .. } => entries.keys().cloned().collect::<Vec<_>>(), _ => return Err(Error::Runtime("table heap invariant broken".into())) }; keys.sort_by(|left, right| match (left, right) { (TableKey::Index(a), TableKey::Index(b)) => a.cmp(b), (TableKey::Name(a), TableKey::Name(b)) => a.cmp(b), (TableKey::Index(_), TableKey::Name(_)) => std::cmp::Ordering::Less, (TableKey::Name(_), TableKey::Index(_)) => std::cmp::Ordering::Greater, }); let reference = self.allocate(HeapObject::TableKeys(keys)); self.push(Value::TableKeys(reference)); self.collect_if_needed(); },
            Op::TableKeysIndex => { let index = integer_to_usize(&self.pop()?)?; let object = self.pop()?; let Value::TableKeys(reference) = object else { return Err(Error::Runtime("VM table_keys invariant broken".into())); }; let key = match self.heap_ref().get(reference)? { HeapObject::TableKeys(keys) => keys.get(index).cloned().ok_or_else(|| Error::Runtime(format!("table key index {index} is out of bounds")))?, _ => return Err(Error::Runtime("table_keys heap invariant broken".into())) }; self.push(Value::TableKey(Rc::new(key))); },
            Op::TableRemove => { let key_value = self.pop()?; let key = table_key_from_value(self.heap_ref(), &key_value)?; let object = self.pop()?; let Value::Table(reference, _) = object else { return Err(Error::Runtime("VM table invariant broken".into())); }; let removed = match self.heap_mut().get_mut(reference)? { HeapObject::Table { entries, .. } => entries.remove(&key).is_some(), _ => return Err(Error::Runtime("table heap invariant broken".into())) }; self.push(Value::Bool(removed)); },
            Op::Field(field) => { let object = self.pop()?; let Value::Struct(reference, _) = object else { return Err(Error::Runtime("VM struct invariant broken".into())); }; let value = match self.heap_ref().get(reference)? { HeapObject::Struct { values, .. } => values.get(field.index).cloned().ok_or_else(|| Error::Runtime("invalid struct field index".into()))?, _ => return Err(Error::Runtime("struct heap invariant broken".into())) }; self.push(value); },
            Op::TableField(name) => { let object = self.pop()?; let Value::Table(reference, _) = object else { return Err(Error::Runtime("VM table invariant broken".into())); }; let value = match self.heap_ref().get(reference)? { HeapObject::Table { entries, .. } => entries.get(&TableKey::Name(name.clone())).cloned().ok_or_else(|| Error::Runtime(format!("table has no key {name}")))?, _ => return Err(Error::Runtime("table heap invariant broken".into())) }; self.push(value); },
            Op::ModuleField(name) => { let Value::Module(id) = self.pop()? else { return Err(Error::Runtime("VM module invariant broken".into())); }; let value = { let instance = self.modules.get(&id).ok_or_else(|| Error::Runtime("loaded module is missing".into()))?; let ModuleExport::Value { slot, .. } = instance.artifact.exports.get(name.as_ref()).ok_or_else(|| Error::Runtime(format!("module has no exported value '{name}'")))? else { return Err(Error::Runtime(format!("'{name}' is not an exported module value"))); }; instance.vm.locals.get(*slot).cloned().ok_or_else(|| Error::Runtime("invalid module export slot".into()))? }; self.push(value); },
            Op::Binary(op) => {
                let right = self.pop_compiled();
                let left = self.pop_compiled();
                let result = {
                    let heap = self.heap_ref();
                    evaluate_binary(&heap, left, right, op)?
                };
                self.push(result);
            },
            Op::Unary(op, ty) => { let val = self.pop()?; self.push(evaluate_unary(val, op, ty.as_ref())?); },
            Op::Cast(target_ty) => {
                let val = self.pop()?;
                self.push(cast_numeric(val, target_ty.as_ref())?);
            },
            Op::Len => {
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
                    Value::Tensor(reference, _, _) => match self.heap_ref().get(reference)? {
                        HeapObject::Tensor { shape, .. } => shape.iter().try_fold(1usize, |total, dimension| total.checked_mul(*dimension)).ok_or_else(|| Error::Runtime("tensor is too large".into()))?,
                        _ => return Err(Error::Runtime("tensor heap invariant broken".into())),
                    },
                    _ => return Err(Error::Runtime("VM len invariant broken".into())),
                };
                let length = i32::try_from(length).map_err(|_| Error::Runtime("vector length exceeds i32".into()))?;
                self.push(Value::I32(length));
            },
            Op::ConcatString => {
                let right = self.pop()?;
                let left = self.pop()?;
                let Value::String(left_ref) = left else { return Err(Error::Runtime("VM string invariant broken".into())); };
                let Value::String(right_ref) = right else { return Err(Error::Runtime("VM string invariant broken".into())); };

                // Borrow only long enough to copy both source slices directly
                // into one exactly-sized allocation.
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
            Op::Builtin1(name, _ty) => { let arg = self.pop()?; self.push(evaluate_builtin1(*name, arg)?); },
            Op::Builtin2(name, ty) => { let arg2 = self.pop_compiled(); let arg1 = self.pop_compiled(); self.push(evaluate_builtin2(*name, arg1, arg2, ty.as_ref())?); },
            Op::CallExternal(name, argument_count) => self.call_external(name.as_ref(), *argument_count)?,
            Op::JumpIfFalse(target) => { match self.pop()? { Value::Bool(false) => { pc = *target; continue; }, Value::Bool(true) => {}, _ => return Err(Error::Runtime("VM condition invariant broken".into())), } },
            Op::Jump(target) => { pc = *target; continue; },
            Op::JumpIfFalseKeep(target) => {
                if self.stack_ptr == 0 { return Err(Error::Runtime("stack underflow".into())); }
                let val = &self.stack[self.stack_ptr - 1];
                if let Value::Bool(false) = val {
                    pc = *target; continue;
                } else if let Value::Bool(true) = val {
                    self.pop()?;
                } else {
                    return Err(Error::Runtime("VM condition invariant broken".into()));
                }
            },
            Op::JumpIfTrueKeep(target) => {
                if self.stack_ptr == 0 { return Err(Error::Runtime("stack underflow".into())); }
                let val = &self.stack[self.stack_ptr - 1];
                if let Value::Bool(true) = val {
                    pc = *target; continue;
                } else if let Value::Bool(false) = val {
                    self.pop()?;
                } else {
                    return Err(Error::Runtime("VM condition invariant broken".into()));
                }
            },
            Op::CallMethod(receiver, target) => { call_stack.push((next_pc, current_receiver)); current_receiver = Some(*receiver); pc = *target; continue; },
            Op::CallCurrentMethod(target) => { let receiver = current_receiver.ok_or_else(|| Error::Runtime("this is available only inside a method".into()))?; call_stack.push((next_pc, current_receiver)); current_receiver = Some(receiver); pc = *target; continue; },
            Op::CallModule(slot, name) => { self.call_module_function(*slot, name.as_ref())?; },
            Op::Return => { if let Some((return_pc, previous_receiver)) = call_stack.pop() { current_receiver = previous_receiver; pc = return_pc; continue; } if terminal_return { return Ok(&self.output); } return Err(Error::Runtime("return outside method".into())); },
            Op::Print => { let value = self.pop()?; let text = self.format_value(&value)?; self.emit(text); },
            Op::Printf(num_args) => {
                let format_val = self.pop()?;
                let mut arg_vals = Vec::with_capacity(*num_args);
                for _ in 0..*num_args { arg_vals.push(self.pop()?); }
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
            Op::Putc => {
                let value = self.pop()?;
                let codepoint = u32::try_from(integer_to_usize(&value)?)
                    .map_err(|_| Error::Runtime("putc code point exceeds u32".into()))?;
                let c = char::from_u32(codepoint)
                    .ok_or_else(|| Error::Runtime(format!("putc requires a valid Unicode scalar value, got {codepoint}")))?;
                if self.interactive { print!("{c}"); let _ = std::io::stdout().flush(); }
            },
        } pc = next_pc; } Ok(&self.output)
    }

    fn load_module(&mut self, artifact: ModuleArtifact) -> Result<(), Error> {
        let id = artifact.id.clone();
        if !self.modules.contains_key(&id) {
            let mut vm = Vm::with_shared_heap(self.heap.clone(), self.extern_functions.clone(), self.callback_state);
            vm.run(&artifact.code)?;
            let output = std::mem::take(&mut vm.output);
            self.modules.insert(id.clone(), ModuleInstance { artifact, vm });
            for line in output { self.emit(line); }
        }
        self.push(Value::Module(id)); Ok(())
    }

    fn call_module_function(&mut self, slot: usize, name: &str) -> Result<(), Error> {
        let Value::Module(id) = self.locals.get(slot).cloned().ok_or_else(|| Error::Runtime("invalid module slot".into()))? else { return Err(Error::Runtime("VM module slot invariant broken".into())); };
        let output = {
            let instance = self.modules.get_mut(&id).ok_or_else(|| Error::Runtime("loaded module is missing".into()))?;
            let entry = match instance.artifact.exports.get(name).ok_or_else(|| Error::Runtime(format!("module has no exported function '{name}'")))? {
                ModuleExport::Function { entry } => *entry,
                _ => return Err(Error::Runtime(format!("'{name}' is not an exported module function"))),
            };
            // Module bytecode is immutable and shared. This is an O(1)
            // reference-count increment, not a bytecode-buffer copy.
            let code = Rc::clone(&instance.artifact.code);
            instance.vm.run_from(&code, entry, true)?;
            std::mem::take(&mut instance.vm.output)
        };
        for line in output { self.emit(line); } Ok(())
    }

    #[inline(always)]
    fn push(&mut self, value: Value) {
        if self.stack_ptr >= self.stack.len() { self.stack.resize((self.stack_ptr + 1) * 2, Value::Bool(false)); }
        unsafe { self.push_unchecked(value) }
    }

    #[inline(always)]
    unsafe fn push_unchecked(&mut self, value: Value) {
        *self.stack.get_unchecked_mut(self.stack_ptr) = value;
        self.stack_ptr += 1;
    }

    #[inline(always)]
    unsafe fn pop_unchecked(&mut self) -> Value {
        self.stack_ptr -= 1;
        self.stack.get_unchecked(self.stack_ptr).clone()
    }

    /// Used only by opcodes whose stack effect is fixed and emitted by the
    /// compiler.  The debug assertion keeps compiler/VM mismatches visible
    /// during development; release builds perform no underflow branch.
    #[inline(always)]
    fn pop_compiled(&mut self) -> Value {
        debug_assert!(self.stack_ptr > 0, "compiler emitted a stack underflow");
        unsafe { self.pop_unchecked() }
    }

    fn pop(&mut self) -> Result<Value, Error> {
        if self.stack_ptr == 0 { return Err(Error::Runtime("stack underflow".into())); }
        Ok(unsafe { self.pop_unchecked() })
    }
    fn emit(&mut self, s: String) {
        if self.interactive { println!("{s}"); }
        self.output.push(s);
    }

    fn read_input(&mut self, ty: &Type) -> Result<Value, Error> {
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

fn integer_to_usize(v: &Value) -> Result<usize, Error> { match v { Value::I8(x) if *x>=0 => Ok(*x as usize),Value::I16(x) if *x>=0 => Ok(*x as usize),Value::I32(x) if *x>=0 => Ok(*x as usize),Value::I64(x) if *x>=0 => Ok(*x as usize),Value::U8(x)=>Ok(*x as usize),Value::U16(x)=>Ok(*x as usize),Value::U32(x)=>Ok(*x as usize),Value::U64(x)=>usize::try_from(*x).map_err(|_|Error::Runtime("array index too large".into())),_=>Err(Error::Runtime("array index must be non-negative integer".into())) } }

fn evaluate_builtin1(builtin: BuiltinFn, arg: Value) -> Result<Value, Error> {
    if builtin == BuiltinFn::Abs {
        match arg {
            Value::I8(v) => return v.checked_abs().map(Value::I8).ok_or_else(|| Error::Runtime("abs overflow".into())),
            Value::I16(v) => return v.checked_abs().map(Value::I16).ok_or_else(|| Error::Runtime("abs overflow".into())),
            Value::I32(v) => return v.checked_abs().map(Value::I32).ok_or_else(|| Error::Runtime("abs overflow".into())),
            Value::I64(v) => return v.checked_abs().map(Value::I64).ok_or_else(|| Error::Runtime("abs overflow".into())),
            Value::U8(_) | Value::U16(_) | Value::U32(_) | Value::U64(_) => return Ok(arg),
            Value::F32(v) => return Ok(Value::F32(v.abs())),
            Value::F64(v) => return Ok(Value::F64(v.abs())),
            Value::F16(v) => return Ok(Value::F16(f32_to_f16(f16_to_f32(v).abs()))),
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
        _ => Err(Error::Runtime("built-in requires a float".into()))
    }
}

fn evaluate_builtin2(builtin: BuiltinFn, a: Value, b: Value, ty: &Type) -> Result<Value, Error> {
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
        _ => Err(Error::Runtime("built-in requires matching floats".into()))
    }
}
fn evaluate_binary(heap: &Heap, a: Value, b: Value, opcode: &BinaryOp) -> Result<Value, Error> {
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
        BinaryOp::Equal | BinaryOp::NotEqual => unreachable!(),
    }
    Err(Error::Runtime("VM execution invariant broken: unsupported binary op".into()))
}

fn evaluate_unary(a: Value, op: &UnOp, ty: &Type) -> Result<Value, Error> {
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
                _ => {}
            }
        },
        UnOp::Not => {
            if let Value::Bool(v) = a { return Ok(Value::Bool(!v)); }
        }
    }
    Err(Error::Runtime("VM execution invariant broken: unsupported unary op".into()))
}

fn cast_numeric(val: Value, target_ty: &Type) -> Result<Value, Error> {
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
        _ => return Err(Error::Runtime("invalid cast source".into())),
    })
}

/// Compile and execute a source unit. `print` output is returned line by line.
pub fn execute(source: &str) -> Result<Vec<String>, Error> { let (program, strings) = Parser::new(lex(source)?).into_program()?; let code = Compiler::default().with_strings(strings).compile(program)?; let mut vm = Vm::default(); Ok(vm.run(&code)?.to_vec()) }
/// Compile and execute a source unit with deterministic input lines.  This is
/// intended for embeddings and tests, where reading the host stdin is not
/// appropriate.
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
/// Compile and execute a source unit with output flushed as it is produced.
pub fn execute_interactive(source: &str) -> Result<(), Error> { let (program, strings) = Parser::new(lex(source)?).into_program()?; let code = Compiler::default().with_strings(strings).compile(program)?; let mut vm = Vm { interactive: true, ..Vm::default() }; vm.run(&code)?; Ok(()) }
/// Compile and execute an L0 file, allowing `require` to load relative modules below the directory containing that file.
pub fn execute_file(path: impl AsRef<Path>) -> Result<Vec<String>, Error> { Vm::default().execute_file(path) }
/// File-based interactive execution. Unlike `execute_interactive`, this mode supports `require` and treats the source file's directory as module root.
pub fn execute_interactive_file(path: impl AsRef<Path>) -> Result<(), Error> { let path = fs::canonicalize(path.as_ref()).map_err(|error| Error::Runtime(format!("cannot open source file: {error}")))?; let root = path.parent().ok_or_else(|| Error::Runtime("source file has no parent directory".into()))?.to_path_buf(); let source = fs::read_to_string(&path).map_err(|error| Error::Runtime(format!("cannot read source file: {error}")))?; let (program, strings) = Parser::new(lex(&source)?).into_program()?; let code = Compiler::with_module_root(root).with_strings(strings).compile(program)?; let mut vm = Vm { interactive: true, ..Vm::default() }; vm.run(&code)?; Ok(()) }

/// Opaque C ABI state. Only this crate may access its interior.
#[repr(C)] pub struct L0State {
    vm: Vm,
    ffi_call: Option<FfiCall>,
    is_executing: bool,
    last_error: Option<CString>,
}
pub type L0CFunction = unsafe extern "C" fn(*mut L0State) -> c_int;

/// Stable scalar type IDs accepted by the C FFI registration API.
///
/// C values are decoded from `c_int` before becoming this enum, so invalid C
/// input cannot create an invalid Rust enum discriminant.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum L0TypeId {
    I8 = 0, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, Bool,
}

impl L0TypeId {
    fn from_raw(value: c_int) -> Option<Self> {
        Some(match value {
            0 => Self::I8, 1 => Self::I16, 2 => Self::I32, 3 => Self::I64,
            4 => Self::U8, 5 => Self::U16, 6 => Self::U32, 7 => Self::U64,
            8 => Self::F16, 9 => Self::F32, 10 => Self::F64, 11 => Self::Bool,
            _ => return None,
        })
    }

    fn to_l0_type(self) -> Type {
        match self {
            Self::I8 => Type::I8, Self::I16 => Type::I16, Self::I32 => Type::I32, Self::I64 => Type::I64,
            Self::U8 => Type::U8, Self::U16 => Type::U16, Self::U32 => Type::U32, Self::U64 => Type::U64,
            Self::F16 => Type::F16, Self::F32 => Type::F32, Self::F64 => Type::F64, Self::Bool => Type::Bool,
        }
    }
}

fn c_scalar_type(value: c_int) -> Option<Type> { L0TypeId::from_raw(value).map(L0TypeId::to_l0_type) }
fn ffi_argument(state: &L0State, index: usize) -> Option<&Value> { if let Some(call) = state.ffi_call.as_ref() { call.arguments.get(index) } else { state.vm.stack[..state.vm.stack_ptr].get(index) } }
fn ffi_push(state: &mut L0State, value: Value) { if let Some(call) = state.ffi_call.as_mut() { call.results.push(value); } else { state.vm.push(value); } }
macro_rules! c_scalar_helpers {
    ($push:ident, $read:ident, $variant:ident, $ty:ty) => {
        #[no_mangle] pub unsafe extern "C" fn $push(state: *mut L0State, value: $ty) {
            if let Some(state) = state.as_mut() { ffi_push(state, Value::$variant(value)); }
        }
        #[no_mangle] pub unsafe extern "C" fn $read(state: *mut L0State, index: usize, out: *mut $ty) -> c_int {
            let Some(state) = state.as_ref() else { return 0 };
            let Some(out) = out.as_mut() else { return 0 };
            let Some(Value::$variant(value)) = ffi_argument(state, index) else { return 0 };
            *out = *value;
            1
        }
    };
}

c_scalar_helpers!(l0_push_i8, l0_to_i8, I8, i8);
c_scalar_helpers!(l0_push_i16, l0_to_i16, I16, i16);
c_scalar_helpers!(l0_push_i32, l0_to_i32, I32, i32);
c_scalar_helpers!(l0_push_i64, l0_to_i64, I64, i64);
c_scalar_helpers!(l0_push_u8, l0_to_u8, U8, u8);
c_scalar_helpers!(l0_push_u16, l0_to_u16, U16, u16);
c_scalar_helpers!(l0_push_u32, l0_to_u32, U32, u32);
c_scalar_helpers!(l0_push_u64, l0_to_u64, U64, u64);
c_scalar_helpers!(l0_push_f16, l0_to_f16, F16, u16);
c_scalar_helpers!(l0_push_f32, l0_to_f32, F32, f32);
c_scalar_helpers!(l0_push_f64, l0_to_f64, F64, f64);

#[no_mangle] pub unsafe extern "C" fn l0_push_bool(state: *mut L0State, value: c_int) {
    if let Some(state) = state.as_mut() { ffi_push(state, Value::Bool(value != 0)); }
}
#[no_mangle] pub unsafe extern "C" fn l0_to_bool(state: *mut L0State, index: usize, out: *mut c_int) -> c_int {
    let Some(state) = state.as_ref() else { return 0 };
    let Some(out) = out.as_mut() else { return 0 };
    let Some(Value::Bool(value)) = ffi_argument(state, index) else { return 0 };
    *out = c_int::from(*value);
    1
}

#[no_mangle] pub extern "C" fn l0_abi_version() -> u32 { ABI_VERSION }
#[no_mangle] pub extern "C" fn l0_new_state() -> *mut L0State { Box::into_raw(Box::new(L0State { vm: Vm::default(), ffi_call: None, is_executing: false, last_error: None })) }
/// # Safety
/// `state` must be valid. The returned pointer remains valid until the next
/// operation on this state and must not be freed by the caller.
#[no_mangle] pub unsafe extern "C" fn l0_last_error(state: *const L0State) -> *const std::os::raw::c_char {
    state.as_ref().and_then(|state| state.last_error.as_ref()).map_or(std::ptr::null(), |message| message.as_ptr())
}
/// # Safety
/// `state` must have been returned by `l0_new_state` and not freed already.
#[no_mangle] pub unsafe extern "C" fn l0_free_state(state: *mut L0State) { if !state.is_null() { drop(Box::from_raw(state)); } }
/// # Safety
/// `state` must be valid and `name` must be a NUL-terminated UTF-8 string.
#[no_mangle] pub unsafe extern "C" fn l0_register_i32_function(state: *mut L0State, name: *const std::os::raw::c_char, function: L0CFunction, argument_count: usize) -> c_int {
    let Some(state) = state.as_mut() else { return 0 };
    if name.is_null() { return 0; }
    let Ok(name) = CStr::from_ptr(name).to_str() else { return 0; };
    match state.vm.register_c_i32_function(name, argument_count, function) { Ok(()) => 1, Err(_) => 0 }
}
/// # Safety
/// `state` must be valid, `name` must be NUL-terminated UTF-8, and `arg_types`
/// must address `argument_count` type IDs when that count is nonzero.
#[no_mangle] pub unsafe extern "C" fn l0_register_c_function(state: *mut L0State, name: *const std::os::raw::c_char, function: L0CFunction, arg_types: *const c_int, argument_count: usize, result_type: c_int) -> c_int {
    let Some(state) = state.as_mut() else { return 0 };
    if name.is_null() || (argument_count != 0 && arg_types.is_null()) { return 0; }
    let Ok(name) = CStr::from_ptr(name).to_str() else { return 0; };
    let Some(result) = c_scalar_type(result_type) else { return 0; };
    let raw_arguments = if argument_count == 0 { &[] } else { std::slice::from_raw_parts(arg_types, argument_count) };
    let mut arguments = Vec::with_capacity(argument_count);
    for &raw_type in raw_arguments { let Some(ty) = c_scalar_type(raw_type) else { return 0; }; arguments.push(ty); }
    match state.vm.register_external(name.to_owned(), HostSignature { arguments, result }, ExternalFunction::C(function)) { Ok(()) => 1, Err(_) => 0 }
}
/// # Safety
/// `state` must be valid and `source` must be a NUL-terminated UTF-8 L0 unit.
#[no_mangle] pub unsafe extern "C" fn l0_execute(state: *mut L0State, source: *const std::os::raw::c_char) -> c_int {
    let Some(state_ref) = state.as_mut() else { return 0 };
    if source.is_null() { return 0; }
    let Ok(source) = CStr::from_ptr(source).to_str() else { return 0; };
    if state_ref.is_executing { return 0; }
    state_ref.last_error = None;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state_ref.is_executing = true;
        let previous_state = state_ref.vm.callback_state.replace(state);
        let result = state_ref.vm.execute(source);
        state_ref.vm.callback_state = previous_state;
        state_ref.is_executing = false;
        result
    }));
    match result {
        Ok(Ok(_)) => 1,
        Ok(Err(error)) => { state_ref.last_error = CString::new(error.to_string()).ok(); 0 }
        Err(_) => {
            state_ref.is_executing = false;
            state_ref.vm.callback_state = None;
            state_ref.last_error = CString::new("panic prevented from crossing the C ABI boundary").ok();
            0
        }
    }
}
/// Converts an `f32` to IEEE-754 binary16 bits using the `half` crate's
/// correctly rounded implementation.
pub fn f32_to_f16(value: f32) -> u16 { f16::from_f32(value).to_bits() }

/// Converts IEEE-754 binary16 bits to `f32` using the `half` crate.
pub fn f16_to_f32(bits: u16) -> f32 { f16::from_bits(bits).to_f32() }

#[cfg(test)]
mod immediate_regression_tests {
    use super::*;
    #[test]
    fn float_literals_cannot_claim_non_float_types() {
        assert!(matches!(execute("let value: i32 = 1.0"), Err(Error::Located { source, location: SourceLocation { line: 1, column: 18, .. } }) if matches!(*source, Error::Type(_))));
        assert!(matches!(execute("let value: bool = 1.0"), Err(Error::Located { source, location: SourceLocation { line: 1, column: 19, .. } }) if matches!(*source, Error::Type(_))));
    }
    #[test]
    fn specialized_i32_addition_reports_overflow() {
        assert!(matches!(
            execute("let value: i32 = 2147483647; print(value + 1)"),
            Err(Error::Runtime(message)) if message == "addition overflow"
        ));
    }
    #[test]
    fn unary_not_is_lexed_parsed_and_evaluated() {
        assert_eq!(execute("let no: bool = 1 == 0; print(!no)").unwrap(), vec!["true".to_owned()]);
    }
    #[test]
    fn strings_compare_by_content() {
        assert_eq!(
            execute("let left: string = \"same\"; let right: string = \"same\"; print(left == right)").unwrap(),
            vec!["true".to_owned()]
        );
    }
    #[test]
    fn block_locals_do_not_escape_or_consume_slots_forever() {
        assert!(matches!(
            execute("if 1 == 1 then let inner: i32 = 7 end print(inner)"),
            Err(Error::Located { source, location: SourceLocation { line: 1, column: 45, .. } }) if matches!(&*source, Error::Type(message) if message == "unknown name 'inner'")
        ));
        assert!(matches!(
            execute("for i = 1, 2 do let temporary: i32 = i end print(i)"),
            Err(Error::Located { source, location: SourceLocation { line: 1, column: 50, .. } }) if matches!(&*source, Error::Type(message) if message == "unknown name 'i'")
        ));
    }
    #[test]
    fn len_supports_all_sized_runtime_values() {
        assert_eq!(
            execute("let text: string = \"ёж\"; let map: table<i32> = table { a = 1, b = 2 }; print(len(text)); print(len(map))").unwrap(),
            vec!["2".to_owned(), "2".to_owned()]
        );
    }
    #[test]
    fn packed_scalars_are_little_endian() {
        let mut bytes = Vec::new();
        encode_scalar(&Value::I32(0x0102_0304), &Type::I32, &mut bytes).unwrap();
        assert_eq!(bytes, [4, 3, 2, 1]);
        assert_eq!(decode_scalar(&bytes, 0, &Type::I32).unwrap(), Value::I32(0x0102_0304));
    }
    #[test]
    fn queued_input_is_consumed_before_stdin() {
        let mut vm = Vm::default();
        vm.push_input("42".to_owned());
        assert_eq!(vm.execute("let answer: i32 = input; print(answer)").unwrap(), vec!["42"]);
    }
    #[test]
    fn public_input_helper_supports_strings_and_numbers() {
        assert_eq!(
            execute_with_input(
                "let label: string = input; let answer: i32 = input; print(label); print(answer)",
                ["ready", "42"],
            ).unwrap(),
            vec!["ready", "42"],
        );
    }
    #[test]
    fn tables_accept_string_keys_and_support_removal_and_iteration() {
        let source = r#"
            let map: table<i32> = table { ["alpha"] = 2, [7] = 3 };
            let name: string = "alpha";
            map[name] = 5;
            print(len(map));
            print(remove(map, name));
            print(len(map));
            let table_keys: table_keys = keys(map);
            for i = 0, len(table_keys) - 1 do
                let key: table_key = table_keys[i];
                print(map[key]);
            end
        "#;
        assert_eq!(execute(source).unwrap(), vec!["2", "true", "1", "3"]);
    }
    #[test]
    fn printf_validates_arguments_and_brace_escapes() {
        assert_eq!(execute("printf(\"{{{}}}\", 7)").unwrap(), vec!["{7}"]);
        assert!(matches!(execute("printf(\"{}\", 7, 8)"), Err(Error::Runtime(message)) if message == "printf has more arguments than placeholders"));
        assert!(matches!(execute("printf(\"{\", 7)"), Err(Error::Runtime(message)) if message.contains("invalid '{' escape")));
    }
    #[test]
    fn putc_rejects_invalid_unicode_scalars() {
        assert!(matches!(execute("putc(1114112)"), Err(Error::Runtime(message)) if message.contains("valid Unicode scalar")));
    }
    #[test]
    fn zero_tensor_bytes_need_no_scalar_encoding() {
        let mut vm = Vm::default();
        assert_eq!(vm.make_tensor_bytes(TensorInit::Zeros, &Type::I32, &[2, 3]).unwrap(), vec![0; 24]);
    }

    #[test]
    fn stack_value_stays_within_four_machine_words() {
        // This protects the VM stack from a future metadata-bearing variant
        // accidentally turning every stack slot into a cache-line fragment.
        assert!(std::mem::size_of::<Value>() <= 4 * std::mem::size_of::<usize>());
    }
    #[test]
    fn garbage_collection_tracks_payload_bytes() {
        let mut heap = Heap::default();
        heap.allocate(HeapObject::String("payload".repeat(128)));
        assert!(heap.allocated_bytes > 0);
        assert_eq!(heap.collect(Vec::new()), 1);
        assert_eq!(heap.allocated_bytes, 0);
    }
    #[test]
    fn garbage_collection_traverses_references_without_cloning_values() {
        let mut heap = Heap::default();
        let child = heap.allocate(HeapObject::String("reachable".into()));
        let parent = heap.allocate(HeapObject::Table {
            entries: HashMap::from([(TableKey::Name(Rc::from("child")), Value::String(child))]),
            element: Type::String,
        });

        assert_eq!(heap.collect([parent]), 0);
        assert!(matches!(heap.get(child), Ok(HeapObject::String(text)) if text == "reachable"));
    }
    #[test]
    fn f16_conversion_uses_standard_binary16_encoding() {
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f16_to_f32(0xc000), -2.0);
    }
    #[test]
    fn builtins_are_resolved_during_compilation() {
        assert_eq!(execute("let value: f32 = 4.0; print(sqrt(value)); print(min(value, 3.0))").unwrap(), vec!["2", "3"]);
    }
    #[test]
    fn lexer_and_core_ast_names_share_string_ids() {
        let lexed = lex("let repeated: i32 = 1; print(repeated)").unwrap();
        let ids: Vec<_> = lexed.tokens.iter().filter_map(|token| match token.kind {
            Token::Ident(id) => Some(id), _ => None,
        }).collect();
        assert_eq!(ids[0], ids[2]);
        assert_eq!(lexed.strings.resolve(ids[0]), "repeated");
    }
    #[test]
    fn executable_bytecode_is_a_flat_u32_stream() {
        let (program, strings) = Parser::new(lex("let answer: i32 = 42; print(answer)").unwrap()).into_program().unwrap();
        let code = Compiler::default().with_strings(strings).compile(program).unwrap();
        assert!(code.words.iter().all(|word| *word <= u32::MAX));
        assert_eq!(code.words.len() % 4, 0);
        assert_eq!(Vm::default().run(&code).unwrap(), ["42"]);
    }
    #[test]
    fn concat_allocates_exact_payload_capacity() {
        let mut vm = Vm::default();
        assert_eq!(vm.execute("let left: string = \"left\"; let right: string = \"right\"; let joined: string = left + right; print(joined)").unwrap(), ["leftright"]);
        let Value::String(reference) = vm.locals[2] else { panic!("joined local is not a string") };
        let HeapObject::String(joined) = vm.heap_ref().get(reference).unwrap() else { panic!("joined value is not a heap string") };
        assert_eq!(joined.capacity(), joined.len());
    }
    #[test]
    fn numeric_casts_work_correctly() {
        let source = "let a: i8 = 10; let b: i32 = a as i32; print(b)";
        assert_eq!(execute(source).unwrap(), vec!["10".to_owned()]);
        let source_float = "let f: f32 = 42.5; let i: i32 = f as i32; print(i)";
        assert_eq!(execute(source_float).unwrap(), vec!["42".to_owned()]);
    }
    #[test]
    fn elseif_chain_evaluates_correctly() {
        let source = "
            let x: i32 = 2
            if x == 1 then
                print(10)
            elseif x == 2 then
                print(20)
            elseif x == 3 then
                print(30)
            else
                print(40)
            end
        ";
        assert_eq!(execute(source).unwrap(), vec!["20".to_owned()]);
    }
    #[test]
    fn cast_various_numeric_types_and_chains() {
        let source = r#"
            -- знаковое в беззнаковое с циклическим переносом (-1 -> 255)
            let neg: i32 = -1;
            let u: u8 = neg as u8;
            print(u);
            -- усечение старших битов (300 % 256 = 44)
            let big: i32 = 300;
            let wrapped: u8 = big as u8;
            print(wrapped);
            -- усечение дробной части к нулю (-3.7 -> -3)
            let f_neg: f64 = -3.7;
            let i_trunc: i32 = f_neg as i32;
            print(i_trunc);
            -- конвертация с участием f16 (binary16)
            let orig_f32: f32 = 2.5;
            let half: f16 = orig_f32 as f16;
            let back_f64: f64 = half as f64;
            print(back_f64);
            -- цепочки приведений
            let chained: i32 = 123.75 as f32 as i32;
            print(chained);
            -- приведение результата выражения в скобках
            let expr_cast: f64 = (10 + 5) as f64 / 2.0;
            print(expr_cast);
        "#;
        assert_eq!(
            execute(source).unwrap(),
            vec!["255", "44", "-3", "2.5", "123", "7.5"]
        );
    }
    #[test]
    fn cast_rejects_non_numeric_types() {
        // bool нельзя приводить к числовым типам (используем 1 == 1 вместо true)
        assert!(matches!(
            execute("let b: bool = 1 == 1; let i: i32 = b as i32"),
            Err(Error::Located { source, .. }) if matches!(*source, Error::Type(ref msg) if msg.contains("cannot cast bool to i32"))
        ));
        // string нельзя приводить через cast
        assert!(matches!(
            execute("let s: string = \"hello\"; let i: i32 = s as i32"),
            Err(Error::Located { source, .. }) if matches!(*source, Error::Type(ref msg) if msg.contains("cannot cast string to i32"))
        ));
        // числа нельзя приводить к bool
        assert!(matches!(
            execute("let i: i32 = 42; let b: bool = i as bool"),
            Err(Error::Located { source, .. }) if matches!(*source, Error::Type(ref msg) if msg.contains("cannot cast i32 to bool"))
        ));
    }
    #[test]
    fn garbage_collection_allocates_and_frees_999999_chunks() {
        let mut heap = Heap::default();
        let mut refs = Vec::new();
        for i in 0..999999 {
            let k:  i32 = i + 979999999 % 1333;
            let j:   i64 = (i as i64) * (k as i64);
            let fv: f32 = j  as f32;
            refs.push(heap.allocate(HeapObject::String(format!("mem-Chunk {i}-{k}-{j}-{fv}"))));
        }
        assert_eq!(heap.slots.len(), 999999);
        let roots = vec![refs[0], refs[999998]];
        let reclaimed = heap.collect(roots);
        assert_eq!(reclaimed, 999997);
        assert_eq!(heap.slots.len(), 999999);
        assert!(heap.free_head.is_some());
        let reused_ref = heap.allocate(HeapObject::String("reused".into()));
        assert!(reused_ref.0 > 0 && reused_ref.0 < 999998);
        assert_eq!(heap.slots.len(), 999999);
        let reclaimed_final = heap.collect(Vec::new());
        assert_eq!(reclaimed_final, 3);
        assert_eq!(heap.slots.len(), 0);
        assert_eq!(heap.free_head, None);
        assert_eq!(heap.allocated_bytes, 0);
    }
}
