//! L0: experimental typed Lua-like language with a stack bytecode VM.
//! The public FFI boundary is C ABI, so it is callable both from C and Rust.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::ffi::CStr;
use std::fs;
use std::fmt;
use std::io::{self, Write};
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::rc::Rc;

pub const ABI_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringId(pub u32);

#[derive(Clone, Debug, Default)]
pub struct StringInterner {
    strings: Vec<String>,
    lookup: std::collections::HashMap<String, StringId>,
}

impl StringInterner {
    pub fn new() -> Self { Self::default() }
    pub fn intern(&mut self, s: &str) -> StringId {
        if let Some(&id) = self.lookup.get(s) { return id; }
        let id = StringId(self.strings.len() as u32);
        self.strings.push(s.to_owned());
        self.lookup.insert(s.to_owned(), id);
        id
    }
    pub fn resolve(&self, id: StringId) -> &str { &self.strings[id.0 as usize] }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, Bool, String,
    Array(Box<Type>), Tensor(Box<Type>, usize), Table(Box<Type>), Struct(String), Module(String),
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
    Struct(HeapRef, StructLayout),
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
    Struct { values: Vec<Value>, layout: StructLayout },
}

#[derive(Clone, Debug)]
pub struct HeapSlot { pub marked: bool, pub object: HeapObject }

/// Non-moving mark-and-sweep storage for every reference value in the VM.
#[derive(Debug)]
pub struct Heap {
    pub slots: Vec<Option<HeapSlot>>,
    pub free: Vec<usize>,
    pub allocations_since_collection: usize,
    pub threshold: usize,
}

impl Default for Heap {
    fn default() -> Self { Self { slots: Vec::new(), free: Vec::new(), allocations_since_collection: 0, threshold: 64 } }
}

impl Heap {
    pub fn allocate(&mut self, object: HeapObject) -> HeapRef {
        let index = self.free.pop().unwrap_or_else(|| { self.slots.push(None); self.slots.len() - 1 });
        self.slots[index] = Some(HeapSlot { marked: false, object });
        self.allocations_since_collection += 1;
        HeapRef(index)
    }

    pub fn get(&self, reference: HeapRef) -> Result<&HeapObject, Error> {
        self.slots.get(reference.0).and_then(Option::as_ref).map(|slot| &slot.object).ok_or_else(|| Error::Runtime("dangling heap reference".into()))
    }

    pub fn get_mut(&mut self, reference: HeapRef) -> Result<&mut HeapObject, Error> {
        self.slots.get_mut(reference.0).and_then(Option::as_mut).map(|slot| &mut slot.object).ok_or_else(|| Error::Runtime("dangling heap reference".into()))
    }

    fn mark_value(&mut self, value: &Value) {
        let reference = match value { Value::Array(r, _) | Value::Tensor(r, _, _) | Value::String(r) | Value::Table(r, _) | Value::Struct(r, _) => *r, _ => return };
        let Some(slot) = self.slots.get_mut(reference.0).and_then(Option::as_mut) else { return };
        if slot.marked { return; }
        slot.marked = true;
        let children = match &slot.object {
            HeapObject::Array { .. } | HeapObject::Tensor { .. } | HeapObject::String(_) => Vec::new(),
            HeapObject::Table { entries, .. } => entries.values().cloned().collect(),
            HeapObject::Struct { values, .. } => values.clone(),
        };
        for child in &children { self.mark_value(child); }
    }

    fn collect(&mut self, roots: impl IntoIterator<Item = Value>) -> usize {
        for root in roots { self.mark_value(&root); }
        let mut reclaimed = 0;
        for (index, slot) in self.slots.iter_mut().enumerate() {
            let Some(live) = slot.as_mut() else { continue };
            if live.marked { live.marked = false; } else { *slot = None; self.free.push(index); reclaimed += 1; }
        }
        self.allocations_since_collection = 0;
        self.threshold = (self.slots.len().max(64) * 2).max(64);
        reclaimed
    }

    fn should_collect(&self) -> bool { self.allocations_since_collection >= self.threshold }
    #[cfg(test)] fn allocated_count(&self) -> usize { self.slots.iter().filter(|slot| slot.is_some()).count() }
}

fn table_key_display(key: &TableKey) -> String { match key { TableKey::Index(index) => format!("[{index}]"), TableKey::Name(name) => name.to_string() } }
fn table_key_from_value(value: &Value) -> Result<TableKey, Error> {
    let index = match value {
        Value::I8(v) => *v as i128, Value::I16(v) => *v as i128, Value::I32(v) => *v as i128, Value::I64(v) => *v as i128,
        Value::U8(v) => *v as i128, Value::U16(v) => *v as i128, Value::U32(v) => *v as i128, Value::U64(v) => *v as i128,
        _ => return Err(Error::Runtime("table index must be an integer".into())),
    };
    Ok(TableKey::Index(index))
}

pub fn type_size(ty: &Type) -> Option<usize> {
    match ty {
        Type::I8 | Type::U8 | Type::Bool => Some(1),
        Type::I16 | Type::U16 | Type::F16 => Some(2),
        Type::I32 | Type::U32 | Type::F32 => Some(4),
        Type::I64 | Type::U64 | Type::F64 => Some(8),
        Type::Array(_) | Type::Tensor(_, _) | Type::Table(_) | Type::Struct(_) | Type::String | Type::Module(_) => None,
    }
}

fn scalar_size(ty: &Type) -> Result<usize, Error> { type_size(ty).ok_or_else(|| Error::Type("packed storage supports scalar field and element types only".into())) }

fn encode_scalar(value: &Value, element: &Type, bytes: &mut Vec<u8>) -> Result<(), Error> {
    if &value.ty() != element { return Err(Error::Runtime("VM array type invariant broken".into())); }
    let size = scalar_size(element)?;
    let start = bytes.len();
    bytes.reserve(size);
    unsafe {
        bytes.set_len(start + size);
        let ptr = bytes.as_mut_ptr().add(start);
        match value {
            Value::I8(v) => *ptr = *v as u8,
            Value::U8(v) => *ptr = *v,
            Value::Bool(v) => *ptr = u8::from(*v),
            Value::I16(v) => std::ptr::write_unaligned(ptr as *mut i16, *v),
            Value::U16(v) | Value::F16(v) => std::ptr::write_unaligned(ptr as *mut u16, *v),
            Value::I32(v) => std::ptr::write_unaligned(ptr as *mut i32, *v),
            Value::U32(v) => std::ptr::write_unaligned(ptr as *mut u32, *v),
            Value::F32(v) => std::ptr::write_unaligned(ptr as *mut f32, *v),
            Value::I64(v) => std::ptr::write_unaligned(ptr as *mut i64, *v),
            Value::U64(v) => std::ptr::write_unaligned(ptr as *mut u64, *v),
            Value::F64(v) => std::ptr::write_unaligned(ptr as *mut f64, *v),
            _ => return Err(Error::Type("packed arrays can contain only scalar values".into())),
        }
    }
    Ok(())
}
fn decode_scalar(bytes: &[u8], index: usize, element: &Type) -> Result<Value, Error> {
    let size = scalar_size(element)?;
    let offset = index.checked_mul(size).ok_or_else(|| Error::Runtime("array index too large".into()))?;
    if offset + size > bytes.len() {
        return Err(Error::Runtime(format!("array index {} is out of bounds", index)));
    }

    let ptr = unsafe { bytes.as_ptr().add(offset) };
    unsafe {
        match element {
            Type::I8 => Ok(Value::I8(*ptr as i8)),
            Type::U8 => Ok(Value::U8(*ptr)),
            Type::Bool => Ok(Value::Bool(*ptr != 0)),
            Type::I16 => Ok(Value::I16(std::ptr::read_unaligned(ptr as *const i16))),
            Type::U16 => Ok(Value::U16(std::ptr::read_unaligned(ptr as *const u16))),
            Type::F16 => Ok(Value::F16(std::ptr::read_unaligned(ptr as *const u16))),
            Type::I32 => Ok(Value::I32(std::ptr::read_unaligned(ptr as *const i32))),
            Type::U32 => Ok(Value::U32(std::ptr::read_unaligned(ptr as *const u32))),
            Type::F32 => Ok(Value::F32(std::ptr::read_unaligned(ptr as *const f32))),
            Type::I64 => Ok(Value::I64(std::ptr::read_unaligned(ptr as *const i64))),
            Type::U64 => Ok(Value::U64(std::ptr::read_unaligned(ptr as *const u64))),
            Type::F64 => Ok(Value::F64(std::ptr::read_unaligned(ptr as *const f64))),
            _ => Err(Error::Type("not a scalar type".into())),
        }
    }
}

fn write_scalar(bytes: &mut [u8], index: usize, value: &Value, element: &Type) -> Result<(), Error> {
    let size = scalar_size(element)?;
    let offset = index.checked_mul(size).ok_or_else(|| Error::Runtime("array index too large".into()))?;
    if offset + size > bytes.len() {
        return Err(Error::Runtime(format!("array index {} is out of bounds", index)));
    }

    let ptr = unsafe { bytes.as_mut_ptr().add(offset) };
    unsafe {
        match value {
            Value::I8(v) => *ptr = *v as u8,
            Value::U8(v) => *ptr = *v,
            Value::Bool(v) => *ptr = u8::from(*v),
            Value::I16(v) => std::ptr::write_unaligned(ptr as *mut i16, *v),
            Value::U16(v) => std::ptr::write_unaligned(ptr as *mut u16, *v),
            Value::F16(v) => std::ptr::write_unaligned(ptr as *mut u16, *v),
            Value::I32(v) => std::ptr::write_unaligned(ptr as *mut i32, *v),
            Value::U32(v) => std::ptr::write_unaligned(ptr as *mut u32, *v),
            Value::F32(v) => std::ptr::write_unaligned(ptr as *mut f32, *v),
            Value::I64(v) => std::ptr::write_unaligned(ptr as *mut i64, *v),
            Value::U64(v) => std::ptr::write_unaligned(ptr as *mut u64, *v),
            Value::F64(v) => std::ptr::write_unaligned(ptr as *mut f64, *v),
            _ => return Err(Error::Type("not a scalar type".into())),
        }
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
            Self::Array(reference, element) => write!(f, "vector<{}>@{}", element, reference.0), Self::Tensor(reference, element, rank) => write!(f, "tensor<{}, {}>@{}", element, rank, reference.0), Self::Table(reference, element) => write!(f, "table<{}>@{}", element, reference.0), Self::Struct(reference, layout) => write!(f, "{}@{}", layout.name, reference.0), Self::Module(id) => write!(f, "module({id})")
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub enum Error { Lex(String), Parse(String), Type(String), Runtime(String) }
impl fmt::Display for Error { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self) } }
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinOp { Add, Sub, Mul, Div, Mod, Eq, Neq, Lt, Le, Gt, Ge, And, Or, BitAnd, BitOr, BitXor, Shl, Shr }
#[derive(Clone, Debug, PartialEq)]
pub enum UnOp { Neg, Not }

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Let, Print, Printf, Putc, Input, This, Function, Export, Require, If, Then, Else, While, For, Do, Break, Continue, Struct, Table, End,
    Ident(String), Integer(i128), Float(f64), StringLit(String), Colon, DoubleColon,
    Equal, EqualEqual, BangEq, Plus, Minus, Star, Slash, Percent,
    Ampersand, Pipe, Caret, Shl, Shr, AndAnd, OrOr,
    Dot, Lt, Le, Gt, Ge, LParen, RParen, LBracket, RBracket, LBrace, RBrace, Comma, Semi, Eof
}

#[derive(Clone, Debug)]
enum Expr {
    Integer(i128), Float(f64), String(String), Input, This, Require(String), Name(String),
    Array(Vec<Expr>), Table(Vec<(TableLiteralKey, Expr)>), StructLiteral(String, Vec<(String, Expr)>),
    Binary(Box<Expr>, BinOp, Box<Expr>), Unary(UnOp, Box<Expr>),
    Index(Box<Expr>, Vec<Expr>), Field(Box<Expr>, String),
    Call(String, Vec<Expr>), TensorFactory { name: String, element: Type, shape: Box<Expr> }
}

fn lex(source: &str) -> Result<Vec<Token>, Error> {
    let mut result = Vec::new(); let chars: Vec<char> = source.chars().collect(); let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            c if c.is_whitespace() => i += 1,
            '-' if chars.get(i + 1) == Some(&'-') => { while i < chars.len() && chars[i] != '\n' { i += 1; } }
            '-' if chars.get(i + 1).is_some_and(|c| c.is_ascii_digit()) => {
                let start = i; let mut dot = false; i += 2;
                while i < chars.len() && (chars[i].is_ascii_digit() || (!dot && chars[i] == '.')) { if chars[i] == '.' { dot = true; } i += 1; }
                let raw: String = chars[start..i].iter().collect();
                result.push(if dot { Token::Float(raw.parse().map_err(|_| Error::Lex(raw.clone()))?) }
                            else { Token::Integer(raw.parse().map_err(|_| Error::Lex(raw.clone()))?) });
            }
            ':' if chars.get(i + 1) == Some(&':') => { result.push(Token::DoubleColon); i += 2; }, ':' => { result.push(Token::Colon); i += 1; },
            '=' if chars.get(i + 1) == Some(&'=') => { result.push(Token::EqualEqual); i += 2; }, '=' => { result.push(Token::Equal); i += 1; },
            '!' if chars.get(i + 1) == Some(&'=') => { result.push(Token::BangEq); i += 2; },
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
                if i >= chars.len() { return Err(Error::Lex("unterminated string".into())); }
                i += 1;
                result.push(Token::StringLit(string_val));
            },
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i; i += 1; while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') { i += 1; }
                let word: String = chars[start..i].iter().collect();
                result.push(match word.as_str() {
                    "let" => Token::Let, "print" => Token::Print, "printf" => Token::Printf,
                    "putc" => Token::Putc, "input" => Token::Input, "this" => Token::This,
                    "function" => Token::Function, "export" => Token::Export, "require" => Token::Require,
                    "if" => Token::If, "then" => Token::Then, "else" => Token::Else,
                    "while" => Token::While, "for" => Token::For, "do" => Token::Do,
                    "break" => Token::Break, "continue" => Token::Continue, "struct" => Token::Struct,
                    "table" => Token::Table, "end" => Token::End, _ => Token::Ident(word)
                });
            }
            c if c.is_ascii_digit() => {
                let start = i; let mut dot = false; i += 1;
                while i < chars.len() && (chars[i].is_ascii_digit() || (!dot && chars[i] == '.')) { if chars[i] == '.' { dot = true; } i += 1; }
                let raw: String = chars[start..i].iter().collect();
                result.push(if dot { Token::Float(raw.parse().map_err(|_| Error::Lex(raw.clone()))?) } else { Token::Integer(raw.parse().map_err(|_| Error::Lex(raw.clone()))?) });
            }
            other => return Err(Error::Lex(format!("unexpected character '{other}'"))),
        }
    }
    result.push(Token::Eof); Ok(result)
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
    ExportLet { name: String, ty: Type, expr: Expr },
    ExportStruct { name: String, fields: Vec<(String, Type)>, methods: Vec<StructMethod> },
    ExportFunction { name: String, body: Vec<Statement> },
    CallMethod { receiver: MethodReceiver, method: String },
    Let { name: String, ty: Type, expr: Expr },
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
}

struct Parser { tokens: Vec<Token>, at: usize }
impl Parser {
    fn new(tokens: Vec<Token>) -> Self { Self { tokens, at: 0 } }
    fn peek(&self) -> &Token { &self.tokens[self.at] }
    fn next(&mut self) -> Token { let t = self.tokens[self.at].clone(); self.at += 1; t }
    fn need(&mut self, wanted: Token) -> Result<(), Error> { let got = self.next(); if got == wanted { Ok(()) } else { Err(Error::Parse(format!("expected {wanted:?}, got {got:?}"))) } }
    fn program(&mut self) -> Result<Vec<Statement>, Error> { self.block() }
    fn block(&mut self) -> Result<Vec<Statement>, Error> {
        let mut statements = Vec::new();
        while !matches!(self.peek(), Token::Eof | Token::Else | Token::End) {
            statements.push(self.statement()?);
            if *self.peek() == Token::Semi { self.next(); }
        }
        Ok(statements)
    }
    fn struct_declaration(&mut self) -> Result<(String, Vec<(String, Type)>, Vec<StructMethod>), Error> {
        let name = match self.next() { Token::Ident(name) => name, token => return Err(Error::Parse(format!("expected struct name, got {token:?}"))) };
        self.need(Token::LBrace)?;
        let mut fields = Vec::new(); let mut methods = Vec::new();
        while *self.peek() != Token::RBrace { match self.next() {
            Token::Ident(field) => { self.need(Token::Colon)?; let ty = self.ty()?; fields.push((field, ty)); if matches!(self.peek(), Token::Semi | Token::Comma) { self.next(); } },
            Token::Function => {
                let method = match self.next() { Token::Ident(method) => method, token => return Err(Error::Parse(format!("expected method name, got {token:?}"))) };
                self.need(Token::LParen)?;
                let mut args = Vec::new();
                if *self.peek() != Token::RParen {
                    loop {
                        let arg_name = match self.next() { Token::Ident(n) => n, t => return Err(Error::Parse(format!("expected argument name, got {t:?}"))) };
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
    fn let_declaration(&mut self) -> Result<(String, Type, Expr), Error> {
        let name = match self.next() { Token::Ident(n) => n, x => return Err(Error::Parse(format!("expected name, got {x:?}"))) };
        self.need(Token::Colon)?; let ty = self.ty()?; self.need(Token::Equal)?; let expr = self.expr()?; Ok((name, ty, expr))
    }
    fn statement(&mut self) -> Result<Statement, Error> { match self.next() {
        Token::Struct => { let (name, fields, methods) = self.struct_declaration()?; Ok(Statement::Struct { name, fields, methods }) },
        Token::Export => match self.next() {
            Token::Let => { let (name, ty, expr) = self.let_declaration()?; Ok(Statement::ExportLet { name, ty, expr }) },
            Token::Struct => { let (name, fields, methods) = self.struct_declaration()?; Ok(Statement::ExportStruct { name, fields, methods }) },
            Token::Function => { let name = match self.next() { Token::Ident(name) => name, token => return Err(Error::Parse(format!("expected exported function name, got {token:?}"))) }; self.need(Token::LParen)?; self.need(Token::RParen)?; let body = self.block()?; self.need(Token::End)?; Ok(Statement::ExportFunction { name, body }) },
            token => Err(Error::Parse(format!("expected let, struct, or function after export, got {token:?}"))),
        },
        Token::Function => {
            let struct_name = match self.next() { Token::Ident(name) => name, token => return Err(Error::Parse(format!("expected struct name, got {token:?}"))) };
            self.need(Token::DoubleColon)?;
            let method = match self.next() { Token::Ident(method) => method, token => return Err(Error::Parse(format!("expected method name, got {token:?}"))) };
            self.need(Token::LParen)?;
            let mut args = Vec::new();
            if *self.peek() != Token::RParen {
                loop {
                    let arg_name = match self.next() { Token::Ident(name) => name, token => return Err(Error::Parse(format!("expected argument name, got {token:?}"))) };
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
        Token::If => { let condition = self.expr()?; self.need(Token::Then)?; let then_body = self.block()?; let else_body = if *self.peek() == Token::Else { self.next(); self.block()? } else { Vec::new() }; self.need(Token::End)?; Ok(Statement::If { condition, then_body, else_body }) },
        Token::While => { let condition = self.expr()?; self.need(Token::Do)?; let body = self.block()?; self.need(Token::End)?; Ok(Statement::While { condition, body }) },
        Token::For => {
            let name = match self.next() { Token::Ident(name) => name, token => return Err(Error::Parse(format!("expected loop variable, got {token:?}"))) };
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
        Token::This => { self.need(Token::Dot)?; let method = match self.next() { Token::Ident(method) => method, token => return Err(Error::Parse(format!("expected method name, got {token:?}"))) }; self.need(Token::LParen)?; self.need(Token::RParen)?; Ok(Statement::CallMethod { receiver: MethodReceiver::This, method }) },
        Token::Ident(name) => match self.next() {
            Token::Equal => Ok(Statement::Assign { name, expr: self.expr()? }),
            Token::LBracket => { let indices = self.indices()?; self.need(Token::Equal)?; Ok(Statement::SetIndex { name, indices, expr: self.expr()? }) },
            Token::Dot => {
                let field = match self.next() { Token::Ident(field) => field, token => return Err(Error::Parse(format!("expected field or method name, got {token:?}"))) };
                match self.next() {
                    Token::Equal => Ok(Statement::SetField { name, field, expr: self.expr()? }),
                    Token::LBracket => { let index = self.expr()?; self.need(Token::RBracket)?; self.need(Token::Equal)?; Ok(Statement::SetFieldIndex { name, field, index, expr: self.expr()? }) },
                    Token::LParen => { self.need(Token::RParen)?; Ok(Statement::CallMethod { receiver: MethodReceiver::Name(name), method: field }) },
                    token => Err(Error::Parse(format!("expected '=', '[', or '(', got {token:?}"))),
                }
            },
            token => Err(Error::Parse(format!("expected '=', '[' or '.', got {token:?}")))
        },
        x => Err(Error::Parse(format!("expected statement, got {x:?}"))), }
    }
    fn ty(&mut self) -> Result<Type, Error> { match self.next() {
        Token::Ident(n) => match n.as_str() {
            "i8" => Ok(Type::I8), "i16" => Ok(Type::I16), "i32" => Ok(Type::I32), "i64" => Ok(Type::I64),
            "u8" => Ok(Type::U8), "u16" => Ok(Type::U16), "u32" => Ok(Type::U32), "u64" => Ok(Type::U64),
            "f16" => Ok(Type::F16), "f32" => Ok(Type::F32), "f64" => Ok(Type::F64), "bool" => Ok(Type::Bool),
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
                let member = match self.next() { Token::Ident(member) => member, token => return Err(Error::Parse(format!("expected exported struct name, got {token:?}"))) };
                Ok(Type::Struct(format!("{n}.{member}")))
            },
            _ => Ok(Type::Struct(n)),
        },
        Token::Table => { self.need(Token::Lt)?; let inner = self.ty()?; self.need(Token::Gt)?; Ok(Type::Table(Box::new(inner))) },
        Token::Struct => match self.next() { Token::Ident(name) => Ok(Type::Struct(name)), token => Err(Error::Parse(format!("expected struct name, got {token:?}"))) },
        x => Err(Error::Parse(format!("expected type, got {x:?}"))) }
    }
    fn expr(&mut self) -> Result<Expr, Error> { self.logical_or() }
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
    fn unary(&mut self) -> Result<Expr, Error> {
        if *self.peek() == Token::Minus {
            self.next();
            Ok(Expr::Unary(UnOp::Neg, Box::new(self.unary()?)))
        } else if *self.peek() == Token::BangEq {
            // Обработка логического Not, если он поддерживается
            // Здесь для примера вызывается primary(), но если есть оператор '!', нужно его добавить
            self.primary()
        } else {
            self.primary()
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
                Expr::Require(path)
            },
            Token::Table => {
                self.need(Token::LBrace)?;
                let mut entries = Vec::new();
                while *self.peek() != Token::RBrace {
                    let key = if *self.peek() == Token::LBracket {
                        self.next(); let index = self.expr()?; self.need(Token::RBracket)?; TableLiteralKey::Index(index)
                    } else {
                        match self.next() { Token::Ident(name) => TableLiteralKey::Name(name), token => return Err(Error::Parse(format!("expected table key, got {token:?}"))), }
                    };
                    self.need(Token::Equal)?; entries.push((key, self.expr()?));
                    if matches!(self.peek(), Token::Comma | Token::Semi) { self.next(); }
                }
                self.need(Token::RBrace)?; Expr::Table(entries)
            },
            Token::Ident(n) => {
                let mut struct_name = n.clone();
                if *self.peek() == Token::Dot && matches!(self.tokens.get(self.at + 1), Some(Token::Ident(_))) && matches!(self.tokens.get(self.at + 2), Some(Token::LBrace)) {
                    self.next();
                    let member = match self.next() { Token::Ident(member) => member, _ => unreachable!() };
                    struct_name = format!("{n}.{member}");
                }
                if *self.peek() == Token::Lt && matches!(n.as_str(), "zeros" | "random") {
                    self.next();
                    let element = self.ty()?;
                    scalar_size(&element)?;
                    self.need(Token::Gt)?;
                    self.need(Token::LParen)?;
                    let shape = self.expr()?;
                    self.need(Token::RParen)?;
                    Expr::TensorFactory { name: n, element, shape: Box::new(shape) }
                } else if *self.peek() == Token::LBrace {
                    self.next(); let mut fields = Vec::new();
                    while *self.peek() != Token::RBrace {
                        let field = match self.next() { Token::Ident(field) => field, token => return Err(Error::Parse(format!("expected field name, got {token:?}"))) };
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
                    Expr::Call(n, args)
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
                self.next(); let field = match self.next() { Token::Ident(field) => field, token => return Err(Error::Parse(format!("expected field name, got {token:?}"))) }; e = Expr::Field(Box::new(e), field);
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
struct ModuleArtifact { id: String, code: Vec<Op>, exports: HashMap<String, ModuleExport> }

#[derive(Clone, Debug)]
struct HostSignature { arguments: Vec<Type>, result: Type }

#[derive(Clone, Debug)]
enum Op {
    AddI32, AddF32, AddF64,
    Push(Value), MakeString(String), Input(Type), Require(ModuleArtifact), Load(usize), LoadCurrentReceiver,
    LoadCurrentField(StructField), Store(usize), StoreIndex(usize, Type), StoreTableIndex(usize, Type),
    StoreTensorIndex(usize, Type, usize), StoreTensorIndexF32(usize, usize),
    StoreField(usize, StructField), StoreFieldIndex(usize, StructField, Type), StoreTableField(usize, Rc<str>, Type),
    StoreCurrentField(StructField), MakeArray(usize, Type), MakeTable(Vec<TableEntry>, Type), MakeStruct(StructLayout),
    MakeTensor(TensorInit, Type, usize), Index, TensorIndex(Type, usize), TensorIndexF32(usize), TableIndex, Field(StructField), TableField(Rc<str>), ModuleField(String),
    Binary(BinaryOp), Unary(UnOp, Type), Len, ConcatString,
    Builtin1(String, Type), Builtin2(String, Type),
    CallExternal(String, usize),
    JumpIfFalse(usize), Jump(usize), JumpIfFalseKeep(usize), JumpIfTrueKeep(usize),
    CallMethod(usize, usize), CallCurrentMethod(usize), CallModule(usize, String),
    Return, Print, Printf(usize), Putc
}

#[derive(Clone, Copy, Debug)]
enum TensorInit { Zeros, Random }

struct LoopContext { break_jumps: Vec<usize>, continue_jumps: Vec<usize>, continue_target: usize }

struct Compiler {
    names: HashMap<String, (usize, Type)>, structs: HashMap<String, StructLayout>,
    methods: HashMap<(String, String), Option<usize>>, pending_method_calls: Vec<(usize, String, String)>,
    current_method_fields: Option<HashMap<String, StructField>>, current_method_struct: Option<String>,
    module_root: Option<PathBuf>, module_artifacts: HashMap<String, ModuleArtifact>,
    exports: HashMap<String, ModuleExport>, extern_functions: HashMap<String, HostSignature>, code: Vec<Op>,
    interned_names: HashMap<String, Rc<str>>,
    next_slot: usize, loops: Vec<LoopContext>
}

impl Default for Compiler { fn default() -> Self { Self { names: HashMap::new(), structs: HashMap::new(), methods: HashMap::new(), pending_method_calls: Vec::new(), current_method_fields: None, current_method_struct: None, module_root: None, module_artifacts: HashMap::new(), exports: HashMap::new(), extern_functions: HashMap::new(), code: Vec::new(), interned_names: HashMap::new(), next_slot: 0, loops: Vec::new() } } }

impl Compiler {
    fn with_module_root(module_root: PathBuf) -> Self { Self { module_root: Some(module_root), ..Self::default() } }
    fn with_extern_functions(extern_functions: HashMap<String, HostSignature>) -> Self { Self { extern_functions, ..Self::default() } }

    fn intern_name(&mut self, name: &str) -> Rc<str> {
        if let Some(interned) = self.interned_names.get(name) { return interned.clone(); }
        let interned: Rc<str> = Rc::from(name);
        self.interned_names.insert(name.to_owned(), interned.clone());
        interned
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

    fn compile(mut self, program: Vec<Statement>) -> Result<Vec<Op>, Error> { self.compile_program(program)?; Ok(self.code) }

    fn compile_module(mut self, id: String, program: Vec<Statement>) -> Result<ModuleArtifact, Error> { self.compile_program(program)?; Ok(ModuleArtifact { id, code: self.code, exports: self.exports }) }

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
        let source = fs::read_to_string(&canonical).map_err(|error| Error::Runtime(format!("cannot read module '{requested}': {error}")))?;
        let program = Parser::new(lex(&source)?).program()?;
        let mut module_compiler = Compiler::with_module_root(root.to_path_buf());
        module_compiler.extern_functions = self.extern_functions.clone();
        let module = module_compiler.compile_module(id.clone(), program)?;
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
        for statement in body { self.statement(statement)?; }
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
        // Read the arguments and bind them to local variable slots
        for (arg_name, arg_ty) in args.into_iter().rev() {
            let slot = self.next_slot;
            self.next_slot += 1;
            self.names.insert(arg_name, (slot, arg_ty));
            self.code.push(Op::Store(slot));
        }
        for statement in body { self.statement(statement)?; }
        self.current_method_fields = previous_fields;
        self.current_method_struct = previous_struct;
        self.code.push(Op::Return);
        let after_body = self.code.len();
        self.code[skip_body] = Op::Jump(after_body);
        Ok(())
    }
    fn statement(&mut self, stmt: Statement) -> Result<(), Error> { match stmt {
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
        Statement::ExportLet { name, ty, expr } => { self.statement(Statement::Let { name: name.clone(), ty, expr })?; let (slot, ty) = self.names.get(&name).cloned().ok_or_else(|| Error::Runtime("missing exported local".into()))?; if self.exports.insert(name.clone(), ModuleExport::Value { slot, ty }).is_some() { return Err(Error::Type(format!("module already exports '{name}'"))); } Ok(()) },
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
                            self.code.push(Op::CallModule(slot, method)); return Ok(());
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
        Statement::Assign { name, expr } => { if let Some(field) = self.current_method_fields.as_ref().and_then(|fields| fields.get(&name)).cloned() { let found = self.expr(expr, Some(&field.ty))?; if found != field.ty { return Err(Error::Type(format!("field '{name}' is {}, but expression has type {found}", field.ty))); } self.code.push(Op::StoreCurrentField(field)); Ok(()) } else { let (slot, ty) = self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?; let found = self.expr(expr, Some(&ty))?; if found != ty { return Err(Error::Type(format!("'{name}' is {ty}, but expression has type {found}"))); } self.code.push(Op::Store(slot)); Ok(()) } },
        Statement::SetIndex { name, indices, expr } => {
            let (slot, container_ty) = self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?;
            match container_ty {
                Type::Array(inner) => {
                    if indices.len() != 1 { return Err(Error::Type("vector indexing requires exactly one index".into())); }
                    scalar_size(&inner)?; self.compile_tensor_indices(indices)?;
                    let element = *inner; let found = self.expr(expr, Some(&element))?;
                    if found != element { return Err(Error::Type(format!("item is {found}; expected {element}"))); }
                    self.code.push(Op::StoreIndex(slot, element)); Ok(())
                },
                Type::Table(inner) => {
                    if indices.len() != 1 { return Err(Error::Type("table indexing requires exactly one index".into())); }
                    self.compile_tensor_indices(indices)?;
                    let element = *inner; let found = self.expr(expr, Some(&element))?;
                    if found != element { return Err(Error::Type(format!("item is {found}; expected {element}"))); }
                    self.code.push(Op::StoreTableIndex(slot, element)); Ok(())
                },
                Type::Tensor(inner, rank) => {
                    if indices.len() != rank { return Err(Error::Type(format!("tensor rank {rank} requires {rank} index(es)"))); }
                    self.compile_tensor_indices(indices)?;
                    let element = *inner; let found = self.expr(expr, Some(&element))?;
                    if found != element { return Err(Error::Type(format!("tensor item is {found}; expected {element}"))); }
                    if element == Type::F32 { self.code.push(Op::StoreTensorIndexF32(slot, rank)); } else { self.code.push(Op::StoreTensorIndex(slot, element, rank)); } Ok(())
                },
                _ => Err(Error::Type(format!("'{name}' is not indexable"))),
            }
        },
        Statement::SetField { name, field, expr } => { let (slot, ty) = self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?; match ty { Type::Struct(struct_name) => { let layout = self.structs.get(&struct_name).ok_or_else(|| Error::Type(format!("unknown struct '{struct_name}'")))?; let field = layout.fields.iter().find(|candidate| candidate.name == field).cloned().ok_or_else(|| Error::Type(format!("struct '{struct_name}' has no field '{field}'")))?; let found = self.expr(expr, Some(&field.ty))?; if found != field.ty { return Err(Error::Type("struct field type mismatch".into())); } self.code.push(Op::StoreField(slot, field)); Ok(()) }, Type::Table(element) => { let element = *element; let found = self.expr(expr, Some(&element))?; if found != element { return Err(Error::Type("table value type mismatch".into())); } let field = self.intern_name(&field); self.code.push(Op::StoreTableField(slot, field, element)); Ok(()) }, _ => Err(Error::Type(format!("'{name}' has no named keys"))), } },
        Statement::SetFieldIndex { name, field, index, expr } => { let (slot, Type::Struct(struct_name)) = self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))? else { return Err(Error::Type(format!("'{name}' is not a struct"))); }; let layout = self.structs.get(&struct_name).ok_or_else(|| Error::Type(format!("unknown struct '{struct_name}'")))?; let field = layout.fields.iter().find(|candidate| candidate.name == field).cloned().ok_or_else(|| Error::Type(format!("struct '{struct_name}' has no field '{field}'")))?; let Type::Array(element) = field.ty.clone() else { return Err(Error::Type(format!("field '{}' is not a vector", field.name))); }; let element = *element; scalar_size(&element)?; let index_ty = self.expr(index, None)?; if !matches!(index_ty, Type::I8|Type::I16|Type::I32|Type::I64|Type::U8|Type::U16|Type::U32|Type::U64) { return Err(Error::Type("index must be an integer".into())); } let found = self.expr(expr, Some(&element))?; if found != element { return Err(Error::Type(format!("item is {found}; expected {element}"))); } self.code.push(Op::StoreFieldIndex(slot, field, element)); Ok(()) },
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
        Statement::If { condition, then_body, else_body } => { let ty = self.expr(condition, None)?; if ty != Type::Bool { return Err(Error::Type(format!("if condition must be bool, got {ty}"))); } let false_jump = self.code.len(); self.code.push(Op::JumpIfFalse(usize::MAX)); for statement in then_body { self.statement(statement)?; } if else_body.is_empty() { let end = self.code.len(); self.code[false_jump] = Op::JumpIfFalse(end); } else { let end_jump = self.code.len(); self.code.push(Op::Jump(usize::MAX)); let else_start = self.code.len(); self.code[false_jump] = Op::JumpIfFalse(else_start); for statement in else_body { self.statement(statement)?; } let end = self.code.len(); self.code[end_jump] = Op::Jump(end); } Ok(()) },
        Statement::While { condition, body } => {
            let loop_start = self.code.len();
            let ty = self.expr(condition, None)?;
            if ty != Type::Bool { return Err(Error::Type(format!("while condition must be bool, got {ty}"))); }
            let exit_jump = self.code.len();
            self.code.push(Op::JumpIfFalse(usize::MAX));
            self.loops.push(LoopContext { break_jumps: Vec::new(), continue_jumps: Vec::new(), continue_target: loop_start });
            for statement in body { self.statement(statement)?; }
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
            for statement in body { self.statement(statement)?; }
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
        Expr::Integer(n) => { let ty = expected.unwrap_or(&Type::I32); let val = int_value(n, ty)?; self.code.push(Op::Push(val)); Ok(ty.clone()) },
        Expr::Float(n) => { let ty = expected.unwrap_or(&Type::F64); let val = float_value(n, ty); self.code.push(Op::Push(val)); Ok(ty.clone()) },
        Expr::String(s) => { self.code.push(Op::MakeString(s)); Ok(Type::String) },
        Expr::Input => { let ty = expected.filter(|t| is_numeric(t)).cloned().ok_or_else(|| Error::Type("input needs an expected numeric type, e.g. let value: i32 = input".into()))?; self.code.push(Op::Input(ty.clone())); Ok(ty) },
        Expr::This => { let struct_name = self.current_method_struct.clone().ok_or_else(|| Error::Type("this is available only inside a struct method".into()))?; self.code.push(Op::LoadCurrentReceiver); Ok(Type::Struct(struct_name)) },
        Expr::Require(path) => { let module = self.load_module(&path)?; let id = module.id.clone(); self.code.push(Op::Require(module)); Ok(Type::Module(id)) },
        Expr::Name(name) => { if let Some(field) = self.current_method_fields.as_ref().and_then(|fields| fields.get(&name)).cloned() { let ty = field.ty.clone(); self.code.push(Op::LoadCurrentField(field)); Ok(ty) } else { let (slot, ty) = self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?; self.code.push(Op::Load(slot)); Ok(ty) } },
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
            self.code.push(Op::MakeArray(count, element.clone()));
            Ok(Type::Array(Box::new(element)))
        },
        Expr::Table(items) => { let element = match expected { Some(Type::Table(t)) => (**t).clone(), _ => return Err(Error::Type("table needs an explicit value type, e.g. table<i32>".into())) }; let mut entries = Vec::with_capacity(items.len()); for (key, value) in items { match key { TableLiteralKey::Name(name) => { let name = self.intern_name(&name); entries.push(TableEntry::Name(name)); }, TableLiteralKey::Index(index) => { let key_ty = self.expr(index, None)?; if !matches!(key_ty, Type::I8|Type::I16|Type::I32|Type::I64|Type::U8|Type::U16|Type::U32|Type::U64) { return Err(Error::Type("table index must be an integer".into())); } entries.push(TableEntry::Index); } } let found = self.expr(value, Some(&element))?; if found != element { return Err(Error::Type(format!("table value is {found}; expected {element}"))); } } self.code.push(Op::MakeTable(entries, element.clone())); Ok(Type::Table(Box::new(element))) },
        Expr::TensorFactory { name, element, shape } => {
            let Type::Tensor(expected_element, rank) = expected.cloned().ok_or_else(|| Error::Type(format!("{name}<T> needs an expected tensor type, e.g. let x: tensor<f32, 2> = {name}<f32>([2, 3])")))? else { return Err(Error::Type(format!("{name}<{}> creates a tensor, but the expected type is not tensor", element))); };
            if expected_element.as_ref() != &element { return Err(Error::Type(format!("{name} element type is {element}; expected {expected_element}"))); }
            let shape_ty = self.expr(*shape, Some(&Type::Array(Box::new(Type::U64))))?;
            if shape_ty != Type::Array(Box::new(Type::U64)) { return Err(Error::Type("tensor shape must be vector<u64>".into())); }
            let init = if name == "zeros" { TensorInit::Zeros } else { TensorInit::Random };
            if matches!(init, TensorInit::Random) && !matches!(element, Type::F16 | Type::F32 | Type::F64) { return Err(Error::Type("random<T> supports f16, f32, and f64 tensors only".into())); }
            self.code.push(Op::MakeTensor(init, element.clone(), rank));
            Ok(Type::Tensor(Box::new(element), rank))
        },
        Expr::StructLiteral(name, fields) => { let layout = self.structs.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown struct '{name}'")))?; if let Some(expected) = expected { if expected != &Type::Struct(name.clone()) { return Err(Error::Type(format!("expected {expected}, got {name}"))); } } if fields.len() != layout.fields.len() { return Err(Error::Type(format!("struct '{name}' needs {} field(s)", layout.fields.len()))); } let mut provided = HashMap::new(); for (field_name, field_expr) in fields { if provided.insert(field_name.clone(), field_expr).is_some() { return Err(Error::Type(format!("struct '{name}' initializes field '{field_name}' more than once"))); } } for field in &layout.fields { let expr = provided.remove(&field.name).ok_or_else(|| Error::Type(format!("struct '{name}' is missing field '{}'", field.name)))?; let found = self.expr(expr, Some(&field.ty))?; if found != field.ty { return Err(Error::Type(format!("field '{}' is {found}; expected {}", field.name, field.ty))); } } if let Some((unknown, _)) = provided.into_iter().next() { return Err(Error::Type(format!("struct '{name}' has no field '{unknown}'"))); } self.code.push(Op::MakeStruct(layout)); Ok(Type::Struct(name)) },
        Expr::Call(name, mut args) => {
            match name.as_str() {
                "len" => {
                    if args.len() != 1 { return Err(Error::Type("len expects 1 argument".into())); }
                    let ty = self.expr(args.remove(0), None)?;
                    if !matches!(ty, Type::Array(_)) { return Err(Error::Type(format!("len requires a vector, got {ty}"))); }
                    self.code.push(Op::Len);
                    Ok(Type::I32)
                },
                "sqrt" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "floor" | "ceil" | "round" => {
                    if args.len() != 1 { return Err(Error::Type(format!("{} expects 1 argument", name))); }
                    let arg = args.remove(0);
                    let ty = self.expr(arg, expected)?;
                    if !matches!(ty, Type::F16 | Type::F32 | Type::F64) { return Err(Error::Type(format!("{} requires a float argument, got {}", name, ty))); }
                    self.code.push(Op::Builtin1(name, ty.clone()));
                    Ok(ty)
                },
                "abs" => {
                    if args.len() != 1 { return Err(Error::Type("abs expects 1 argument".into())); }
                    let arg = args.remove(0);
                    let ty = self.expr(arg, expected)?;
                    if !is_numeric(&ty) { return Err(Error::Type("abs requires a numeric argument".into())); }
                    self.code.push(Op::Builtin1(name, ty.clone()));
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
                    self.code.push(Op::Builtin2(name, t1.clone()));
                    Ok(t1)
                },
                _ => {
                    let signature = self.extern_functions.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown function '{name}'")))?;
                    if args.len() != signature.arguments.len() { return Err(Error::Type(format!("{name} expects {} argument(s)", signature.arguments.len()))); }
                    for (argument, expected_type) in args.into_iter().zip(signature.arguments.iter()) {
                        let found = self.expr(argument, Some(expected_type))?;
                        if found != *expected_type { return Err(Error::Type(format!("{name} argument is {found}; expected {expected_type}"))); }
                    }
                    self.code.push(Op::CallExternal(name, signature.arguments.len()));
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
            self.code.push(Op::Unary(op, ty.clone()));
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
                Type::Table(element) => { if indices.len() != 1 { return Err(Error::Type("table indexing requires exactly one index".into())); } self.compile_tensor_indices(indices)?; self.code.push(Op::TableIndex); Ok(*element) },
                Type::Tensor(element, rank) => { if indices.len() != rank { return Err(Error::Type(format!("tensor rank {rank} requires {rank} index(es)"))); } self.compile_tensor_indices(indices)?; if *element == Type::F32 { self.code.push(Op::TensorIndexF32(rank)); } else { self.code.push(Op::TensorIndex((*element).clone(), rank)); } Ok(*element) },
                _ => Err(Error::Type(format!("cannot index {ct}"))),
            }
        },
        Expr::Field(container, field_name) => { let container_ty = self.expr(*container, None)?; match container_ty { Type::Struct(struct_name) => { let layout = self.structs.get(&struct_name).ok_or_else(|| Error::Type(format!("unknown struct '{struct_name}'")))?; let field = layout.fields.iter().find(|field| field.name == field_name).cloned().ok_or_else(|| Error::Type(format!("struct '{struct_name}' has no field '{field_name}'")))?; self.code.push(Op::Field(field.clone())); Ok(field.ty) }, Type::Table(element) => { let element = *element; let field_name = self.intern_name(&field_name); self.code.push(Op::TableField(field_name)); Ok(element) }, Type::Module(module_id) => { let ty = { let module = self.module_artifacts.get(&module_id).ok_or_else(|| Error::Runtime("missing compiled module".into()))?; let ModuleExport::Value { ty, .. } = module.exports.get(&field_name).ok_or_else(|| Error::Type(format!("module has no exported value '{field_name}'")))? else { return Err(Error::Type(format!("'{field_name}' is not an exported module value"))); }; ty.clone() }; self.code.push(Op::ModuleField(field_name)); Ok(ty) }, _ => Err(Error::Type("field access requires a struct, table, or module".into())), } },
    } }

    fn compile_tensor_indices(&mut self, indices: Vec<Expr>) -> Result<(), Error> {
        for index in indices {
            let index_ty = self.expr(index, None)?;
            if !is_integer(&index_ty) { return Err(Error::Type("tensor index must be an integer".into())); }
        }
        Ok(())
    }
}

fn types_compatible(expected: &Type, found: &Type) -> bool { expected == found || matches!((expected, found), (Type::Module(expected), Type::Module(_)) if expected.is_empty()) }
fn is_numeric(t: &Type) -> bool { !matches!(t, Type::Bool|Type::String|Type::Array(_)|Type::Tensor(_, _)|Type::Table(_)|Type::Struct(_)|Type::Module(_)) }
fn is_integer(t: &Type) -> bool { matches!(t, Type::I8|Type::I16|Type::I32|Type::I64|Type::U8|Type::U16|Type::U32|Type::U64) }

fn int_value(n: i128, ty: &Type) -> Result<Value, Error> { macro_rules! v { ($t:ident, $x:ident) => { n.try_into().map(Value::$t).map_err(|_| Error::Type(format!("{n} does not fit in {}", stringify!($x)))) }; } match ty { Type::I8=>v!(I8,i8),Type::I16=>v!(I16,i16),Type::I32=>v!(I32,i32),Type::I64=>v!(I64,i64),Type::U8=>v!(U8,u8),Type::U16=>v!(U16,u16),Type::U32=>v!(U32,u32),Type::U64=>v!(U64,u64), _=>Err(Error::Type(format!("integer literal cannot initialize {ty}"))) } }
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
        let program = Parser::new(lex(source)?).program()?;
        let code = Compiler::with_extern_functions(self.external_signatures()).compile(program)?;
        self.output.clear();
        Ok(self.run(&code)?.to_vec())
    }

    fn roots(&self) -> Vec<Value> {
        let mut roots = Vec::with_capacity(self.stack_ptr + self.locals.len());
        roots.extend(self.stack[..self.stack_ptr].iter().cloned());
        roots.extend(self.locals.iter().cloned());
        for module in self.modules.values() { roots.extend(module.vm.roots()); }
        roots
    }

    /// Run a full collection. The return value is the number of reclaimed
    /// objects; it is public so the runtime can be integration-tested.
    pub fn collect_garbage(&mut self) -> usize {
        self.heap.borrow_mut().collect(self.roots())
    }

    fn collect_if_needed(&mut self) {
        if self.gc_owner && self.heap.borrow().should_collect() { self.collect_garbage(); }
    }

    pub fn allocate(&mut self, object: HeapObject) -> HeapRef { self.heap.borrow_mut().allocate(object) }

    fn tensor_shape_from_value(&self, value: &Value) -> Result<Vec<usize>, Error> {
        let Value::Array(reference, element) = value else { return Err(Error::Runtime("tensor shape must be vector<u64>".into())); };
        if element.as_ref() != &Type::U64 { return Err(Error::Runtime("tensor shape must be vector<u64>".into())); }
        let bytes = match self.heap.borrow().get(*reference)? { HeapObject::Array { bytes, element } if element == &Type::U64 => bytes.clone(), _ => return Err(Error::Runtime("tensor shape heap invariant broken".into())) };
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
        let mut bytes = Vec::with_capacity(count.checked_mul(element_size).ok_or_else(|| Error::Runtime("tensor is too large".into()))?);
        for _ in 0..count {
            let value = match init {
                TensorInit::Zeros => match element {
                    Type::I8 => Value::I8(0), Type::I16 => Value::I16(0), Type::I32 => Value::I32(0), Type::I64 => Value::I64(0),
                    Type::U8 => Value::U8(0), Type::U16 => Value::U16(0), Type::U32 => Value::U32(0), Type::U64 => Value::U64(0),
                    Type::F16 => Value::F16(0), Type::F32 => Value::F32(0.0), Type::F64 => Value::F64(0.0), Type::Bool => Value::Bool(false),
                    _ => return Err(Error::Runtime("tensor element invariant broken".into())),
                },
                TensorInit::Random => match element {
                    Type::F16 => Value::F16(f32_to_f16(self.next_random_unit() as f32)),
                    Type::F32 => Value::F32(self.next_random_unit() as f32),
                    Type::F64 => Value::F64(self.next_random_unit()),
                    _ => return Err(Error::Runtime("random tensor element invariant broken".into())),
                },
            };
            encode_scalar(&value, element, &mut bytes)?;
        }
        Ok(bytes)
    }

    fn format_value(&self, value: &Value) -> Result<String, Error> {
        match value {
            Value::Array(reference, element) => match self.heap.borrow().get(*reference)? {
                HeapObject::Array { bytes, element: stored_element } if stored_element == element.as_ref() => {
                    let size = scalar_size(element)?;
                    let mut values = Vec::with_capacity(bytes.len() / size);
                    for index in 0..bytes.len() / size { values.push(decode_scalar(bytes, index, element)?.to_string()); }
                    Ok(format!("[{}]", values.join(", ")))
                },
                _ => Err(Error::Runtime("array heap invariant broken".into())),
            },
            Value::Tensor(reference, element, rank) => match self.heap.borrow().get(*reference)? {
                HeapObject::Tensor { element: stored_element, shape, .. } if stored_element == element.as_ref() && shape.len() == *rank => Ok(format!("tensor<{}, {}>[{}]", element, rank, shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", "))),
                _ => Err(Error::Runtime("tensor heap invariant broken".into())),
            },
            Value::String(reference) => match self.heap.borrow().get(*reference)? { HeapObject::String(text) => Ok(text.clone()), _ => Err(Error::Runtime("string heap invariant broken".into())) },
            Value::Table(reference, element) => match self.heap.borrow().get(*reference)? { HeapObject::Table { entries, .. } => Ok(format!("table<{}>({})", element, entries.len())), _ => Err(Error::Runtime("table heap invariant broken".into())) },
            Value::Struct(reference, _) => {
                let (values, layout) = match self.heap.borrow().get(*reference)? { HeapObject::Struct { values, layout } => (values.clone(), layout.clone()), _ => return Err(Error::Runtime("struct heap invariant broken".into())) };
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

    fn run(&mut self, code: &[Op]) -> Result<&[String], Error> { self.run_from(code, 0, false) }

    fn run_from(&mut self, code: &[Op], mut pc: usize, terminal_return: bool) -> Result<&[String], Error> {
        let mut call_stack: Vec<(usize, Option<usize>)> = Vec::new();
        let mut current_receiver: Option<usize> = None;

        while pc < code.len() { match &code[pc] {
            Op::AddI32 => unsafe {
                let right = self.pop_unchecked();
                let left = self.pop_unchecked();
                let r = if let Value::I32(v) = right { v } else { std::hint::unreachable_unchecked() };
                let l = if let Value::I32(v) = left { v } else { std::hint::unreachable_unchecked() };
                self.push_unchecked(Value::I32(l + r));
            },
            Op::AddF32 => unsafe {
                let right = self.pop_unchecked();
                let left = self.pop_unchecked();
                let r = if let Value::F32(v) = right { v } else { std::hint::unreachable_unchecked() };
                let l = if let Value::F32(v) = left { v } else { std::hint::unreachable_unchecked() };
                self.push_unchecked(Value::F32(l + r));
            },
            Op::AddF64 => unsafe {
                let right = self.pop_unchecked();
                let left = self.pop_unchecked();
                let r = if let Value::F64(v) = right { v } else { std::hint::unreachable_unchecked() };
                let l = if let Value::F64(v) = left { v } else { std::hint::unreachable_unchecked() };
                self.push_unchecked(Value::F64(l + r));
            },
            Op::Push(v) => self.push(v.clone()),
            Op::MakeString(s) => { let reference = self.allocate(HeapObject::String(s.clone())); self.push(Value::String(reference)); self.collect_if_needed(); },
            Op::Input(ty) => { let val = self.read_input(ty)?; self.push(val); },
            Op::Require(module) => self.load_module(module.clone())?,
            Op::Load(slot) => self.push(self.locals.get(*slot).cloned().ok_or_else(|| Error::Runtime("invalid local slot".into()))?),
            Op::LoadCurrentReceiver => { let receiver = current_receiver.ok_or_else(|| Error::Runtime("this is available only inside a method".into()))?; self.push(self.locals.get(receiver).cloned().ok_or_else(|| Error::Runtime("invalid method receiver".into()))?); },
            Op::LoadCurrentField(field) => { let receiver = current_receiver.ok_or_else(|| Error::Runtime("field access outside method".into()))?; let Value::Struct(reference, _) = self.locals.get(receiver).ok_or_else(|| Error::Runtime("invalid method receiver".into()))? else { return Err(Error::Runtime("VM method receiver invariant broken".into())); }; let value = match self.heap.borrow().get(*reference)? { HeapObject::Struct { values, .. } => values.get(field.index).cloned().ok_or_else(|| Error::Runtime("invalid struct field index".into()))?, _ => return Err(Error::Runtime("struct heap invariant broken".into())) }; self.push(value); },
            Op::Store(slot) => {
                let v = self.pop()?;
                if *slot >= self.locals.len() { self.locals.resize(*slot + 1, Value::Bool(false)); }
                self.locals[*slot] = v;
            },
            Op::StoreIndex(slot, element) => {
                let value = self.pop()?;
                if &value.ty() != element { return Err(Error::Runtime("VM type invariant broken".into())); }
                let index = integer_to_usize(&self.pop()?)?;
                let Value::Array(reference, _) = self.locals.get(*slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM array slot invariant broken".into())); };
                match self.heap.borrow_mut().get_mut(*reference)? {
                    HeapObject::Array { bytes, element: stored_element } if stored_element == element => {
                        write_scalar(bytes, index, &value, element)?;
                    },
                    _ => return Err(Error::Runtime("array heap invariant broken".into())),
                }
            },
            Op::StoreTableIndex(slot, element) => { let value = self.pop()?; if &value.ty() != element { return Err(Error::Runtime("VM table type invariant broken".into())); } let key = table_key_from_value(&self.pop()?)?; let Value::Table(reference, _) = self.locals.get(*slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM table slot invariant broken".into())); }; match self.heap.borrow_mut().get_mut(*reference)? { HeapObject::Table { entries, element: stored_element } if stored_element == element => { entries.insert(key, value); }, _ => return Err(Error::Runtime("table heap invariant broken".into())), } },
            Op::StoreTensorIndex(slot, element, rank) => {
                let value = self.pop()?;
                if &value.ty() != element { return Err(Error::Runtime("VM tensor type invariant broken".into())); }
                let indices = self.pop_tensor_indices(*rank)?;
                let Value::Tensor(reference, stored_element, stored_rank) = self.locals.get(*slot).cloned().ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM tensor slot invariant broken".into())); };
                if stored_element.as_ref() != element || stored_rank != *rank { return Err(Error::Runtime("VM tensor slot type invariant broken".into())); }
                match self.heap.borrow_mut().get_mut(reference)? {
                    HeapObject::Tensor { bytes, element: stored_element, shape } if stored_element == element && shape.len() == *rank => {
                        let offset = Self::tensor_offset(shape, &indices)?;
                        write_scalar(bytes, offset, &value, element)?;
                    },
                    _ => return Err(Error::Runtime("tensor heap invariant broken".into())),
                }
            },
            Op::StoreTensorIndexF32(slot, rank) => {
                let value = self.pop()?;
                if value.ty() != Type::F32 { return Err(Error::Runtime("VM tensor type invariant broken".into())); }
                let indices = self.pop_tensor_indices(*rank)?;
                let Value::Tensor(reference, _, _) = self.locals.get(*slot).cloned().ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM tensor slot invariant broken".into())); };
                match self.heap.borrow_mut().get_mut(reference)? {
                    HeapObject::Tensor { bytes, shape, .. } => {
                        let offset = Self::tensor_offset(shape, &indices)?;
                        let start = offset.checked_mul(4).ok_or_else(|| Error::Runtime("tensor offset is too large".into()))?;
                        if start + 4 > bytes.len() { return Err(Error::Runtime("tensor offset out of bounds".into())); }
                        if let Value::F32(v) = value {
                            unsafe { std::ptr::write_unaligned(bytes.as_mut_ptr().add(start) as *mut f32, v); }
                        }
                    },
                    _ => return Err(Error::Runtime("tensor heap invariant broken".into())),
                }
            },
            Op::StoreField(slot, field) => { let new_value = self.pop()?; if &new_value.ty() != &field.ty { return Err(Error::Runtime("VM type invariant broken".into())); } let Value::Struct(reference, _) = self.locals.get(*slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM struct slot invariant broken".into())); }; match self.heap.borrow_mut().get_mut(*reference)? { HeapObject::Struct { values, .. } => { *values.get_mut(field.index).ok_or_else(|| Error::Runtime("invalid struct field index".into()))? = new_value; }, _ => return Err(Error::Runtime("struct heap invariant broken".into())), } },
            Op::StoreFieldIndex(slot, field, element) => { let value = self.pop()?; if &value.ty() != element { return Err(Error::Runtime("VM vector type invariant broken".into())); } let index = integer_to_usize(&self.pop()?)?; let Value::Struct(struct_reference, _) = self.locals.get(*slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM struct slot invariant broken".into())); }; let array_reference = match self.heap.borrow().get(*struct_reference)? { HeapObject::Struct { values, .. } => match values.get(field.index) { Some(Value::Array(reference, _)) => *reference, _ => return Err(Error::Runtime("VM struct vector field invariant broken".into())), }, _ => return Err(Error::Runtime("struct heap invariant broken".into())), }; let mut encoded = Vec::new(); encode_scalar(&value, element, &mut encoded)?; match self.heap.borrow_mut().get_mut(array_reference)? { HeapObject::Array { bytes, element: stored_element } if stored_element == element => { let start = index.checked_mul(encoded.len()).ok_or_else(|| Error::Runtime("array index too large".into()))?; let end = start.checked_add(encoded.len()).ok_or_else(|| Error::Runtime("array index too large".into()))?; if end > bytes.len() { return Err(Error::Runtime(format!("array index {index} is out of bounds (length {})", bytes.len() / encoded.len()))); } bytes[start..end].copy_from_slice(&encoded); }, _ => return Err(Error::Runtime("array heap invariant broken".into())), } },
            Op::StoreTableField(slot, name, element) => { let value = self.pop()?; if &value.ty() != element { return Err(Error::Runtime("VM table type invariant broken".into())); } let Value::Table(reference, _) = self.locals.get(*slot).ok_or_else(|| Error::Runtime("invalid local slot".into()))? else { return Err(Error::Runtime("VM table slot invariant broken".into())); }; match self.heap.borrow_mut().get_mut(*reference)? { HeapObject::Table { entries, element: stored_element } if stored_element == element => { entries.insert(TableKey::Name(name.clone()), value); }, _ => return Err(Error::Runtime("table heap invariant broken".into())), } },
            Op::StoreCurrentField(field) => { let new_value = self.pop()?; if &new_value.ty() != &field.ty { return Err(Error::Runtime("VM type invariant broken".into())); } let receiver = current_receiver.ok_or_else(|| Error::Runtime("field assignment outside method".into()))?; let Value::Struct(reference, _) = self.locals.get(receiver).ok_or_else(|| Error::Runtime("invalid method receiver".into()))? else { return Err(Error::Runtime("VM method receiver invariant broken".into())); }; match self.heap.borrow_mut().get_mut(*reference)? { HeapObject::Struct { values, .. } => { *values.get_mut(field.index).ok_or_else(|| Error::Runtime("invalid struct field index".into()))? = new_value; }, _ => return Err(Error::Runtime("struct heap invariant broken".into())), } },
            Op::MakeArray(len, ty) => { if self.stack_ptr < *len { return Err(Error::Runtime("stack underflow".into())); } let at = self.stack_ptr - len; let values = self.stack[at..self.stack_ptr].to_vec(); self.stack_ptr = at; let bytes = Value::pack_array(values, ty)?; let reference = self.allocate(HeapObject::Array { bytes, element: ty.clone() }); self.push(Value::Array(reference, Box::new(ty.clone()))); self.collect_if_needed(); },
            Op::MakeTensor(init, element, rank) => {
                let shape_value = self.pop()?;
                let shape = self.tensor_shape_from_value(&shape_value)?;
                if shape.len() != *rank { return Err(Error::Runtime(format!("tensor rank {rank} requires {rank} dimensions, got {}", shape.len()))); }
                let bytes = self.make_tensor_bytes(*init, element, &shape)?;
                let reference = self.allocate(HeapObject::Tensor { bytes, element: element.clone(), shape });
                self.push(Value::Tensor(reference, Box::new(element.clone()), *rank));
                self.collect_if_needed();
            },
            Op::MakeTable(entries, element) => { let value_count = entries.iter().map(|entry| match entry { TableEntry::Index => 2usize, TableEntry::Name(_) => 1usize }).sum(); if self.stack_ptr < value_count { return Err(Error::Runtime("stack underflow".into())); } let at = self.stack_ptr - value_count; let values = self.stack[at..self.stack_ptr].to_vec(); self.stack_ptr = at; let mut cursor = 0; let mut table = HashMap::with_capacity(entries.len()); for entry in entries { let key = match entry { TableEntry::Index => { let key = table_key_from_value(values.get(cursor).ok_or_else(|| Error::Runtime("stack underflow".into()))?)?; cursor += 1; key }, TableEntry::Name(name) => TableKey::Name(name.clone()), }; let value = values.get(cursor).cloned().ok_or_else(|| Error::Runtime("stack underflow".into()))?; cursor += 1; if &value.ty() != element { return Err(Error::Runtime("VM table type invariant broken".into())); } if table.insert(key.clone(), value).is_some() { return Err(Error::Runtime(format!("table key {} is declared more than once", table_key_display(&key)))); } } let reference = self.allocate(HeapObject::Table { entries: table, element: element.clone() }); self.push(Value::Table(reference, Box::new(element.clone()))); self.collect_if_needed(); },
            Op::MakeStruct(layout) => { if self.stack_ptr < layout.fields.len() { return Err(Error::Runtime("stack underflow".into())); } let at = self.stack_ptr - layout.fields.len(); let values = self.stack[at..self.stack_ptr].to_vec(); self.stack_ptr = at; for (field, value) in layout.fields.iter().zip(values.iter()) { if value.ty() != field.ty { return Err(Error::Runtime("VM struct type invariant broken".into())); } } let reference = self.allocate(HeapObject::Struct { values, layout: layout.clone() }); self.push(Value::Struct(reference, layout.clone())); self.collect_if_needed(); },
            Op::Index => { let index = integer_to_usize(&self.pop()?)?; let object = self.pop()?; let Value::Array(reference, element) = object else { return Err(Error::Runtime("VM array invariant broken".into())); }; let value = match self.heap.borrow().get(reference)? { HeapObject::Array { bytes, element: stored_element } if stored_element == element.as_ref() => decode_scalar(bytes, index, &element)?, _ => return Err(Error::Runtime("array heap invariant broken".into())) }; self.push(value); },
            Op::TensorIndex(element, rank) => {
                let indices = self.pop_tensor_indices(*rank)?;
                let object = self.pop()?;
                let Value::Tensor(reference, stored_element, stored_rank) = object else { return Err(Error::Runtime("VM tensor invariant broken".into())); };
                if stored_element.as_ref() != element || stored_rank != *rank { return Err(Error::Runtime("VM tensor type invariant broken".into())); }
                let value = match self.heap.borrow().get(reference)? {
                    HeapObject::Tensor { bytes, element: stored_element, shape } if stored_element == element && shape.len() == *rank => decode_scalar(bytes, Self::tensor_offset(shape, &indices)?, element)?,
                    _ => return Err(Error::Runtime("tensor heap invariant broken".into())),
                };
                self.push(value);
            },
            Op::TensorIndexF32(rank) => {
                let indices = self.pop_tensor_indices(*rank)?;
                let object = self.pop()?;
                let Value::Tensor(reference, _, _) = object else { return Err(Error::Runtime("VM tensor invariant broken".into())); };
                let value = match self.heap.borrow().get(reference)? {
                    HeapObject::Tensor { bytes, shape, .. } => {
                        let offset = Self::tensor_offset(shape, &indices)?;
                        let start = offset.checked_mul(4).ok_or_else(|| Error::Runtime("tensor offset is too large".into()))?;
                        unsafe { Value::F32(std::ptr::read_unaligned(bytes.as_ptr().add(start) as *const f32)) }
                    },
                    _ => return Err(Error::Runtime("tensor heap invariant broken".into())),
                };
                self.push(value);
            },
            Op::TableIndex => { let key = table_key_from_value(&self.pop()?)?; let object = self.pop()?; let Value::Table(reference, _) = object else { return Err(Error::Runtime("VM table invariant broken".into())); }; let value = match self.heap.borrow().get(reference)? { HeapObject::Table { entries, .. } => entries.get(&key).cloned().ok_or_else(|| Error::Runtime(format!("table has no key {}", table_key_display(&key))))?, _ => return Err(Error::Runtime("table heap invariant broken".into())) }; self.push(value); },
            Op::Field(field) => { let object = self.pop()?; let Value::Struct(reference, _) = object else { return Err(Error::Runtime("VM struct invariant broken".into())); }; let value = match self.heap.borrow().get(reference)? { HeapObject::Struct { values, .. } => values.get(field.index).cloned().ok_or_else(|| Error::Runtime("invalid struct field index".into()))?, _ => return Err(Error::Runtime("struct heap invariant broken".into())) }; self.push(value); },
            Op::TableField(name) => { let object = self.pop()?; let Value::Table(reference, _) = object else { return Err(Error::Runtime("VM table invariant broken".into())); }; let value = match self.heap.borrow().get(reference)? { HeapObject::Table { entries, .. } => entries.get(&TableKey::Name(name.clone())).cloned().ok_or_else(|| Error::Runtime(format!("table has no key {name}")))?, _ => return Err(Error::Runtime("table heap invariant broken".into())) }; self.push(value); },
            Op::ModuleField(name) => { let Value::Module(id) = self.pop()? else { return Err(Error::Runtime("VM module invariant broken".into())); }; let value = { let instance = self.modules.get(&id).ok_or_else(|| Error::Runtime("loaded module is missing".into()))?; let ModuleExport::Value { slot, .. } = instance.artifact.exports.get(name).ok_or_else(|| Error::Runtime(format!("module has no exported value '{name}'")))? else { return Err(Error::Runtime(format!("'{name}' is not an exported module value"))); }; instance.vm.locals.get(*slot).cloned().ok_or_else(|| Error::Runtime("invalid module export slot".into()))? }; self.push(value); },
            Op::Binary(op) => { let right = self.pop()?; let left = self.pop()?; self.push(evaluate_binary(left, right, op)?); },
            Op::Unary(op, ty) => { let val = self.pop()?; self.push(evaluate_unary(val, op, ty)?); },
            Op::Len => {
                let value = self.pop()?;
                let Value::Array(reference, element) = value else { return Err(Error::Runtime("VM len invariant broken".into())); };
                let length = match self.heap.borrow().get(reference)? {
                    HeapObject::Array { bytes, element: stored_element } if stored_element == element.as_ref() => bytes.len() / scalar_size(&element)?,
                    _ => return Err(Error::Runtime("array heap invariant broken".into())),
                };
                let length = i32::try_from(length).map_err(|_| Error::Runtime("vector length exceeds i32".into()))?;
                self.push(Value::I32(length));
            },
            Op::ConcatString => {
                let right = self.pop()?;
                let left = self.pop()?;
                let Value::String(left_ref) = left else { return Err(Error::Runtime("VM string invariant broken".into())); };
                let Value::String(right_ref) = right else { return Err(Error::Runtime("VM string invariant broken".into())); };

                // Ограничиваем область видимости .borrow() с помощью блока
                let (left, right) = {
                    let heap = self.heap.borrow();
                    match (heap.get(left_ref)?, heap.get(right_ref)?) {
                        (HeapObject::String(l), HeapObject::String(r)) => (l.clone(), r.clone()),
                        _ => return Err(Error::Runtime("string heap invariant broken".into())),
                    }
                };
                let reference = self.allocate(HeapObject::String(format!("{left}{right}")));
                self.push(Value::String(reference));
                self.collect_if_needed();
            },
            Op::Builtin1(name, _ty) => { let arg = self.pop()?; self.push(evaluate_builtin1(name, arg)?); },
            Op::Builtin2(name, ty) => { let arg2 = self.pop()?; let arg1 = self.pop()?; self.push(evaluate_builtin2(name, arg1, arg2, ty)?); },
            Op::CallExternal(name, argument_count) => self.call_external(name, *argument_count)?,
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
            Op::CallMethod(receiver, target) => { call_stack.push((pc + 1, current_receiver)); current_receiver = Some(*receiver); pc = *target; continue; },
            Op::CallCurrentMethod(target) => { let receiver = current_receiver.ok_or_else(|| Error::Runtime("this is available only inside a method".into()))?; call_stack.push((pc + 1, current_receiver)); current_receiver = Some(receiver); pc = *target; continue; },
            Op::CallModule(slot, name) => { self.call_module_function(*slot, name)?; },
            Op::Return => { if let Some((return_pc, previous_receiver)) = call_stack.pop() { current_receiver = previous_receiver; pc = return_pc; continue; } if terminal_return { return Ok(&self.output); } return Err(Error::Runtime("return outside method".into())); },
            Op::Print => { let value = self.pop()?; let text = self.format_value(&value)?; self.emit(text); },
            Op::Printf(num_args) => {
                let format_val = self.pop()?;
                let mut arg_vals = Vec::with_capacity(*num_args);
                for _ in 0..*num_args { arg_vals.push(self.pop()?); }
                arg_vals.reverse();
                let format_str = if let Value::String(reference) = format_val { match self.heap.borrow().get(reference)? { HeapObject::String(text) => text.clone(), _ => return Err(Error::Runtime("string heap invariant broken".into())) } } else { return Err(Error::Runtime("printf format is not a string".into())); };

                let mut result = String::new();
                let mut arg_idx = 0;
                let mut chars = format_str.chars().peekable();

                while let Some(c) = chars.next() {
                    if c == '{' && chars.peek() == Some(&'}') {
                        chars.next();
                        if arg_idx < arg_vals.len() { result.push_str(&self.format_value(&arg_vals[arg_idx])?); arg_idx += 1; }
                        else { result.push_str("{}"); }
                    } else { result.push(c); }
                }
                if self.interactive { print!("{}", result); let _ = io::stdout().flush(); }
                self.output.push(result);
            },
            Op::Putc => {
                let value = self.pop()?;
                let c = match integer_to_usize(&value) { Ok(v) => v as u8 as char, _ => '?' };
                if self.interactive { print!("{c}"); let _ = std::io::stdout().flush(); }
            },
        } pc += 1; } Ok(&self.output)
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
            let code = instance.artifact.code.clone();
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

    fn pop(&mut self) -> Result<Value, Error> {
        if self.stack_ptr == 0 { return Err(Error::Runtime("stack underflow".into())); }
        self.stack_ptr -= 1;
        Ok(self.stack[self.stack_ptr].clone())
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
        if is_integer(ty) {
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

fn evaluate_builtin1(name: &str, arg: Value) -> Result<Value, Error> {
    if name == "abs" {
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
                let res = match name {
                    "sqrt" => val.sqrt(), "sin" => val.sin(), "cos" => val.cos(),
                    "tan" => val.tan(), "asin" => val.asin(), "acos" => val.acos(),
                    "atan" => val.atan(), "floor" => val.floor(), "ceil" => val.ceil(),
                    "round" => val.round(),
                    _ => return Err(Error::Runtime(format!("unknown function {}", name))),
                };
                if $is_f16 { Ok(Value::F16(f32_to_f16(res as f32))) } else { Ok(Value::$variant(res as _)) }
            }
        }
    }
    match &arg {
        Value::F32(v) => float_math1!(v, f32, F32, false),
        Value::F64(v) => float_math1!(v, f64, F64, false),
        Value::F16(v) => float_math1!(v, f32, F16, true),
        _ => Err(Error::Runtime(format!("{} requires a float", name)))
    }
}

fn evaluate_builtin2(name: &str, a: Value, b: Value, ty: &Type) -> Result<Value, Error> {
    if name == "min" || name == "max" {
        macro_rules! min_max {
            ($x:ident, $l:ident, $r:ident) => {
                if name == "min" { Ok(Value::$x(if $l < $r { *$l } else { *$r })) }
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
                if name == "min" { return Ok(Value::F16(f32_to_f16(if l < r { l } else { r }))); }
                else { return Ok(Value::F16(f32_to_f16(if l > r { l } else { r }))); }
            },
            _ => return Err(Error::Runtime("invalid type for min/max".into()))
        }
    }
    if name == "pow" && is_integer(ty) {
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
                let res = match name {
                    "pow" => l.powf(r), "atan2" => l.atan2(r),
                    _ => return Err(Error::Runtime(format!("unknown function {}", name))),
                };
                if $is_f16 { Ok(Value::F16(f32_to_f16(res as f32))) } else { Ok(Value::$variant(res as _)) }
            }
        }
    }
    match (&a, &b) {
        (Value::F32(l), Value::F32(r)) => float_math2!(l, r, f32, F32, false),
        (Value::F64(l), Value::F64(r)) => float_math2!(l, r, f64, F64, false),
        (Value::F16(l), Value::F16(r)) => float_math2!(l, r, f32, F16, true),
        _ => Err(Error::Runtime(format!("{} requires matching floats", name)))
    }
}
fn evaluate_binary(a: Value, b: Value, opcode: &BinaryOp) -> Result<Value, Error> {
    if matches!(opcode, BinaryOp::Equal) { return Ok(Value::Bool(a == b)); }
    if matches!(opcode, BinaryOp::NotEqual) { return Ok(Value::Bool(a != b)); }
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

/// Compile and execute a source unit. `print` output is returned line by line.
pub fn execute(source: &str) -> Result<Vec<String>, Error> { let program = Parser::new(lex(source)?).program()?; let code = Compiler::default().compile(program)?; let mut vm = Vm::default(); Ok(vm.run(&code)?.to_vec()) }
/// Compile and execute a source unit with output flushed as it is produced.
pub fn execute_interactive(source: &str) -> Result<(), Error> { let program = Parser::new(lex(source)?).program()?; let code = Compiler::default().compile(program)?; let mut vm = Vm { interactive: true, ..Vm::default() }; vm.run(&code)?; Ok(()) }
/// Compile and execute an L0 file, allowing `require` to load relative modules below the directory containing that file.
pub fn execute_file(path: impl AsRef<Path>) -> Result<Vec<String>, Error> { let path = fs::canonicalize(path.as_ref()).map_err(|error| Error::Runtime(format!("cannot open source file: {error}")))?; let root = path.parent().ok_or_else(|| Error::Runtime("source file has no parent directory".into()))?.to_path_buf(); let source = fs::read_to_string(&path).map_err(|error| Error::Runtime(format!("cannot read source file: {error}")))?; let program = Parser::new(lex(&source)?).program()?; let code = Compiler::with_module_root(root).compile(program)?; let mut vm = Vm::default(); Ok(vm.run(&code)?.to_vec()) }
/// File-based interactive execution. Unlike `execute_interactive`, this mode supports `require` and treats the source file's directory as module root.
pub fn execute_interactive_file(path: impl AsRef<Path>) -> Result<(), Error> { let path = fs::canonicalize(path.as_ref()).map_err(|error| Error::Runtime(format!("cannot open source file: {error}")))?; let root = path.parent().ok_or_else(|| Error::Runtime("source file has no parent directory".into()))?.to_path_buf(); let source = fs::read_to_string(&path).map_err(|error| Error::Runtime(format!("cannot read source file: {error}")))?; let program = Parser::new(lex(&source)?).program()?; let code = Compiler::with_module_root(root).compile(program)?; let mut vm = Vm { interactive: true, ..Vm::default() }; vm.run(&code)?; Ok(()) }

/// Opaque C ABI state. Only this crate may access its interior.
#[repr(C)] pub struct L0State { vm: Vm, ffi_call: Option<FfiCall> }
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
#[no_mangle] pub extern "C" fn l0_new_state() -> *mut L0State { Box::into_raw(Box::new(L0State { vm: Vm::default(), ffi_call: None })) }
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
    let previous_state = state_ref.vm.callback_state.replace(state);
    let result = state_ref.vm.execute(source);
    state_ref.vm.callback_state = previous_state;
    if result.is_ok() { 1 } else { 0 }
}
// IEEE binary16 conversion, dependency-free so the language's f16 type is portable.
pub fn f32_to_f16(value: f32) -> u16 { let bits=value.to_bits(); let sign=((bits>>16)&0x8000) as u16; let exp=((bits>>23)&0xff) as i32-127+15; let mant=bits&0x7fffff; if exp<=0 { if exp < -10{return sign}; return sign|(((mant|0x800000)>>(14-exp)) as u16); } if exp>=31 { return sign|0x7c00|if mant==0{0}else{1}; } sign|((exp as u16)<<10)|((mant>>13) as u16) }
pub fn f16_to_f32(bits: u16) -> f32 { let sign=((bits as u32)&0x8000)<<16; let exp=(bits>>10)&0x1f; let mant=(bits&0x03ff) as u32; let out=if exp==0 { if mant==0 {sign} else { let mut m=mant; let mut e=-14i32; while m&0x400==0 {m<<=1;e-=1;} sign|(((e+127) as u32)<<23)|((m&0x3ff)<<13) } } else if exp==31 {sign|0x7f800000|(mant<<13)} else {sign|(((exp as u32+112)<<23))|(mant<<13)}; f32::from_bits(out) }

#[cfg(test)]
mod tests {
    use super::*;

    fn execute_with_input(source: &str, input: &[&str]) -> Result<Vec<String>, Error> {
        let program = Parser::new(lex(source)?).program()?;
        let code = Compiler::default().compile(program)?;
        let mut vm = Vm { input: input.iter().map(|line| (*line).to_owned()).collect(), ..Vm::default() };
        Ok(vm.run(&code)?.to_vec())
    }

    #[test]
    fn typed_stack_vm_and_zero_index() {
        let out = execute("let a: i8 = 7; let b: i8 = 5; let v: vector<i32> = [10, 20, 30]; print(a + b); print(v[0]); print(v[2]);").unwrap();
        assert_eq!(out, ["12", "10", "30"]);
    }

    #[test]
    fn inline_struct_methods_can_read_and_update_their_fields() {
        let source = "struct Board { size: i8; function show_board() print(size); size = size + 1; print(size) end } let board: Board = Board { size = 3 }; board.show_board(); print(board.size);";
        assert_eq!(execute(source).unwrap(), ["3", "4", "4"]);
    }

    #[test]
    fn declared_methods_can_be_defined_outside_the_struct() {
        let source = "let size: i8 = 9; struct Board { size: i8; function show_board() } function Board::show_board() print(size); size = size + 1 end let board: Board = Board { size = 3 }; board.show_board(); print(board.size); print(size);";
        assert_eq!(execute(source).unwrap(), ["3", "4", "9"]);
    }

    #[test]
    fn method_fields_hide_same_named_outer_variables() {
        let source = "let size: i8 = 9; struct Board { size: i8; function show_board() print(size) end } let board: Board = Board { size = 3 }; board.show_board(); print(size);";
        assert_eq!(execute(source).unwrap(), ["3", "9"]);
    }

    #[test]
    fn this_calls_methods_on_the_current_instance() {
        let source = "struct Board { size: i8; function play() this.increment(); this.increment(); print(this.size) end function increment() size = size + 1 end } let board: Board = Board { size = 3 }; board.play(); print(board.size);";
        assert_eq!(execute(source).unwrap(), ["5", "5"]);
    }

    #[test]
    fn this_is_rejected_outside_a_method() {
        assert!(execute("print(this);").is_err());
        assert!(execute("this.run();").is_err());
    }

    #[test]
    fn methods_require_empty_parentheses_and_declarations() {
        assert!(execute("struct Board { size: i8; function show_board(x) end }").is_err());
        assert!(execute("struct Board { size: i8; function show_board() end } let board: Board = Board { size = 1 }; board.show_board(1)").is_err());
        assert!(execute("struct Board { size: i8; } function Board::show_board() end").is_err());
    }

    #[test]
    fn input_uses_the_expected_numeric_type() {
        let source = "let move: i8 = input; let weight: f32 = input; print(move); print(weight);";
        assert_eq!(execute_with_input(source, &["-12", "2.5"]).unwrap(), ["-12", "2.5"]);
    }

    #[test]
    fn input_rejects_invalid_or_out_of_range_numbers() {
        assert!(execute_with_input("let move: i8 = input;", &["128"]).is_err());
        assert!(execute_with_input("let move: i8 = input;", &["not a number"]).is_err());
    }

    #[test]
    fn input_requires_an_expected_numeric_type() {
        assert!(execute("print(input);").is_err());
        assert!(execute("let ready: bool = input;").is_err());
    }

    #[test]
    fn rejects_overflowing_literal() {
        assert!(execute("let x: u8 = 300;").is_err());
    }

    #[test]
    fn f16_roundtrip_is_reasonable() {
        assert!((f16_to_f32(f32_to_f16(1.5)) - 1.5).abs() < 0.001);
    }

    #[test]
    fn parenthesized_expressions_work() {
        assert_eq!(execute("let a: i32 = 2; print((a + 2));").unwrap(), ["4"]);
    }

    #[test]
    fn logical_operators_short_circuit_their_right_hand_side() {
        let source = "let yes: bool = 1 == 1; let no: bool = 1 == 0; print(yes || [1][9] == 1); print(no && [1][9] == 1);";
        assert_eq!(execute(source).unwrap(), ["true", "false"]);
    }

    #[test]
    fn range_for_break_and_continue_work() {
        let source = "let total: i32 = 0; for i = 0, 5 do if i == 2 then continue end total = total + i; if i == 4 then break end end print(total);";
        assert_eq!(execute(source).unwrap(), ["8"]);
    }

    #[test]
    fn break_and_continue_are_rejected_outside_loops() {
        assert!(execute("break;").is_err());
        assert!(execute("continue;").is_err());
    }

    #[test]
    fn len_string_concatenation_and_inferred_arrays_work() {
        let source = "let values: vector<i32> = [1, 2, 3]; let message: string = \"value=\" + \"3\"; print(len(values)); print([4, 5][1]); print(message);";
        assert_eq!(execute(source).unwrap(), ["3", "5", "value=3"]);
        assert!(execute("print([]);").is_err());
    }

    #[test]
    fn packed_vectors_round_trip_every_scalar_type() {
        let cases = vec![
            (Value::I8(-7), Type::I8), (Value::I16(-300), Type::I16),
            (Value::I32(-70_000), Type::I32), (Value::I64(-9_000_000_000), Type::I64),
            (Value::U8(7), Type::U8), (Value::U16(300), Type::U16),
            (Value::U32(70_000), Type::U32), (Value::U64(9_000_000_000), Type::U64),
            (Value::F16(0x3e00), Type::F16), (Value::F32(1.5), Type::F32),
            (Value::F64(-2.25), Type::F64), (Value::Bool(true), Type::Bool),
        ];
        for (value, ty) in cases {
            let bytes = Value::pack_array(vec![value.clone()], &ty).unwrap();
            assert_eq!(decode_scalar(&bytes, 0, &ty).unwrap(), value);
        }
        let bytes = Value::pack_array(vec![Value::I32(0x0102_0304)], &Type::I32).unwrap();
        assert_eq!(bytes, [4, 3, 2, 1]);
    }

    #[test]
    fn vector_syntax_and_mutation_work() {
        let source = "let a: vector<u16> = [1, 258]; a[0] = 513; let b: vector<f32> = [1.5, 2.5]; print(a[0]); print(a[1]); print(b[1]);";
        assert_eq!(execute(source).unwrap(), ["513", "258", "2.5"]);
    }

    #[test]
    fn cpu_tensors_support_typed_factories_and_multidimensional_indexing() {
        let source = "let weights: tensor<f32, 2> = zeros<f32>([2, 3]); weights[0, 2] = 0.99; let alias: tensor<f32, 2> = weights; alias[1, 0] = 2.5; print(weights[0, 2]); print(weights[1, 0]); print(weights);";
        assert_eq!(execute(source).unwrap(), ["0.99", "2.5", "tensor<f32, 2>[2, 3]"]);
    }

    #[test]
    fn cpu_tensor_random_factory_and_shape_validation_work() {
        assert_eq!(execute("let values: tensor<f64, 1> = random<f64>([4]); print(values[0] >= 0.0 && values[0] < 1.0);").unwrap(), ["true"]);
        assert!(execute("let values: tensor<f32, 2> = zeros<f32>([4]);").is_err());
        assert!(execute("let values: tensor<i32, 1> = random<i32>([4]);").is_err());
        assert!(execute("let values: tensor<f32, 2> = zeros<f32>([2, 2]); print(values[0]);").is_err());
    }

    #[test]
    fn tables_support_numeric_and_named_keys_with_one_value_type() {
        let source = "let board: table<i8> = table { [0] = 1, [4] = 2, winner = 0 }; board[8] = 1; board.winner = 1; print(board[0]); print(board[4]); print(board[8]); print(board.winner);";
        assert_eq!(execute(source).unwrap(), ["1", "2", "1", "1"]);
    }

    #[test]
    fn tables_reject_bad_keys_values_and_missing_entries() {
        assert!(execute("let t: table<i8> = table { [1.5] = 1 };").is_err());
        assert!(execute("let t: table<i8> = table { score = 1 }; t.score = 300;").is_err());
        assert!(execute("let t: table<i8> = table { score = 1 }; print(t[0]);").is_err());
        assert!(execute("let t: table<i8> = table { [0] = 1, [0] = 2 };").is_err());
    }

    #[test]
    fn structs_support_scalar_and_vector_fields() {
        let source = "let ready: bool = 1 == 1; struct Sample { signed: i16; unsigned: u32; half: f16; full: f64; ready: bool; } let item: Sample = Sample { full = 2.5, unsigned = 70000, signed = -3, ready = ready, half = 1.5 }; item.signed = -9; print(item.signed); print(item.unsigned); print(item.half); print(item.full); print(item.ready);";
        assert_eq!(execute(source).unwrap(), ["-9", "70000", "1.5", "2.5", "true"]);
    }

    #[test]
    fn structs_can_hold_and_replace_vectors() {
        let source = "struct Board { cells: vector<i8>; score: i8; function show() print(cells[1]); print(this.cells[2]); end } let board: Board = Board { score = 1, cells = [3, 4, 5] }; board.show(); board.cells[0] = 8; board.cells = [9, 10, 11]; print(board.cells[0]);";
        assert_eq!(execute(source).unwrap(), ["4", "5", "9"]);
    }

    #[test]
    fn modules_export_values_structs_and_zero_argument_functions() {
        let root = std::env::temp_dir().join(format!("rlua-module-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("game.l0"), "export let score: i8 = 0; export struct Board { cells: vector<i8>; } export function init() score = score + 1; print(score) end").unwrap();
        fs::write(root.join("main.l0"), "let game: module = require(\"game\"); let same: module = require(\"game\"); game.init(); same.init(); print(game.score); let board: game.Board = game.Board { cells = [4, 5] }; print(board.cells[1]);").unwrap();
        let output = execute_file(root.join("main.l0")).unwrap();
        let _ = fs::remove_dir_all(&root);
        assert_eq!(output, ["1", "2", "2", "5"]);
    }

    #[test]
    fn complex_values_are_shared_by_heap_reference() {
        let source = "struct Counter { value: i32; } let first: Counter = Counter { value = 1 }; let second: Counter = first; second.value = 9; print(first.value);";
        assert_eq!(execute(source).unwrap(), ["9"]);
    }

    #[test]
    fn vectors_are_shared_by_heap_reference() {
        let source = "let first: vector<i32> = [1, 2]; let second: vector<i32> = first; second[0] = 9; print(first[0]);";
        assert_eq!(execute(source).unwrap(), ["9"]);
    }

    #[test]
    fn heap_strings_keep_print_and_printf_semantics() {
        let source = "let greeting: string = \"hello\"; print(greeting); printf(\"{} {}\", greeting, 7);";
        assert_eq!(execute(source).unwrap(), ["hello", "hello 7"]);
    }
    fn rust_add(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
        let [Value::I32(left), Value::I32(right)] = arguments else { return Err(Error::Runtime("unexpected host arguments".into())); };
        Ok(Value::I32(left + right))
    }
    #[test]
    fn registered_rust_function_is_callable_from_l0() {
        let mut vm = Vm::default();
        vm.register_rust_function("host_add", vec![Type::I32, Type::I32], Type::I32, rust_add).unwrap();
        assert_eq!(vm.execute("let total: i32 = host_add(20, 22); print(total);").unwrap(), ["42"]);
    }

    unsafe extern "C" fn c_add(state: *mut L0State) -> c_int {
        let mut left = 0;
        let mut right = 0;
        if l0_to_i32(state, 0, &mut left) == 0 || l0_to_i32(state, 1, &mut right) == 0 { return 1; }
        l0_push_i32(state, left + right);
        0
    }

    #[test]
    fn registered_c_function_is_callable_from_l0() {
        let mut state = L0State { vm: Vm::default(), ffi_call: None };
        assert_eq!(state.vm.register_c_i32_function("c_add", 2, c_add), Ok(()));
        let source = b"let total: i32 = c_add(10, 32); print(total);\0";
        assert_eq!(unsafe { l0_execute(&mut state, source.as_ptr().cast()) }, 1);
        assert_eq!(state.vm.output, ["42"]);
    }

    unsafe extern "C" fn c_calculate_distance(state: *mut L0State) -> c_int {
        let mut distance = 0.0;
        let mut boosted = 0;
        if l0_to_f32(state, 0, &mut distance) == 0 || l0_to_bool(state, 1, &mut boosted) == 0 { return 1; }
        l0_push_f32(state, if boosted != 0 { distance * 1.5 } else { distance });
        0
    }

    #[test]
    fn generic_c_function_accepts_f32_and_bool() {
        let mut state = L0State { vm: Vm::default(), ffi_call: None };
        let argument_types = [L0TypeId::F32 as c_int, L0TypeId::Bool as c_int];
        assert_eq!(unsafe { l0_register_c_function(&mut state, b"calc_dist\0".as_ptr().cast(), c_calculate_distance, argument_types.as_ptr(), argument_types.len(), L0TypeId::F32 as c_int) }, 1);
        let source = b"let d: f32 = calc_dist(100.5, 1 == 1); print(d);\0";
        assert_eq!(unsafe { l0_execute(&mut state, source.as_ptr().cast()) }, 1);
        assert_eq!(state.vm.output, ["150.75"]);
    }

    #[test]
    fn generic_c_registration_rejects_invalid_type_ids_and_null_type_arrays() {
        let mut state = L0State { vm: Vm::default(), ffi_call: None };
        assert_eq!(unsafe { l0_register_c_function(&mut state, b"bad_result\0".as_ptr().cast(), c_add, std::ptr::null(), 0, 99) }, 0);
        assert_eq!(unsafe { l0_register_c_function(&mut state, b"missing_args\0".as_ptr().cast(), c_add, std::ptr::null(), 1, L0TypeId::I32 as c_int) }, 0);
    }

    #[test]
    fn mark_and_sweep_collects_unreachable_cycles() {
        let mut vm = Vm::default();
        let layout = StructLayout {
            name: "Node".into(),
            fields: vec![StructField { name: "next".into(), ty: Type::Struct("Node".into()), index: 0 }],
        };
        let first = vm.allocate(HeapObject::Struct { values: vec![Value::Bool(false)], layout: layout.clone() });
        let second = vm.allocate(HeapObject::Struct { values: vec![Value::Bool(false)], layout: layout.clone() });
        {
            let mut heap = vm.heap.borrow_mut();
            let HeapObject::Struct { values, .. } = heap.get_mut(first).unwrap() else { panic!() };
            values[0] = Value::Struct(second, layout.clone());
            let HeapObject::Struct { values, .. } = heap.get_mut(second).unwrap() else { panic!() };
            values[0] = Value::Struct(first, layout.clone());
        }
        vm.locals.push(Value::Struct(first, layout));
        assert_eq!(vm.collect_garbage(), 0);
        assert_eq!(vm.heap.borrow().allocated_count(), 2);
        vm.locals.clear();
        assert_eq!(vm.collect_garbage(), 2);
        assert_eq!(vm.heap.borrow().allocated_count(), 0);
    }

    #[test]
    fn require_needs_a_file_execution_context() {
        assert!(execute("let game: module = require(\"game\");").is_err());
    }
}

