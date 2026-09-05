use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::cell::RefCell;
use std::rc::Rc;
use std::fs;

// Подтягиваем типы из VM (lib.rs)
use crate::{
    Error, SourceLocation, StringId, StringInterner, Type, Value,
    scalar_size, types_compatible, is_numeric, is_integer, int_value, float_value,
    StructField, StructLayout,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinOp { Add, Sub, Mul, Div, Mod, Eq, Neq, Lt, Le, Gt, Ge, And, Or, BitAnd, BitOr, BitXor, Shl, Shr }

#[derive(Clone, Debug, PartialEq)]
pub enum UnOp { Neg, Not }

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Let, Print, Printf, Putc, Input, This, Function, Export, Require, If, Then, Else, ElseIf,
    While, For, Do, Break, Continue, Struct, Table, End, As, In, Ident(StringId), Integer(i128),
    Float(f64), StringLit(StringId), Colon, DoubleColon, Equal, EqualEqual, Bang, BangEq, Plus,
    Minus, Star, Slash, Percent, Ampersand, Pipe, Caret, Shl, Shr, AndAnd, OrOr, Dot, Lt, Le,
    Gt, Ge, LParen, RParen, LBracket, RBracket, LBrace, RBrace, Comma, Semi, Eof
}

#[derive(Clone, Debug)]
pub struct SpannedToken { pub kind: Token, pub location: SourceLocation }

struct TokenBuffer { tokens: Vec<SpannedToken>, location: SourceLocation }
impl TokenBuffer {
    fn new() -> Self { Self { tokens: Vec::new(), location: SourceLocation { offset: 0, line: 1, column: 1 } } }
    fn begin(&mut self, chars: &[char], offset: usize) { self.location = SourceLocation::at(chars, offset); }
    fn location(&self) -> SourceLocation { self.location }
    fn push(&mut self, kind: Token) { self.tokens.push(SpannedToken { kind, location: self.location }); }
    fn into_tokens(self) -> Vec<SpannedToken> { self.tokens }
}

#[derive(Clone, Debug)]
pub enum Expr {
    Integer(i128), Float(f64), String(StringId), Input, This, Require(String), Name(StringId),
    GlobalName(StringId),
    Array(Vec<Expr>), Table(Vec<(TableLiteralKey, Expr)>), StructLiteral(String, Vec<(String, Expr)>),
    Cast(Box<Expr>, Type),
    Binary(Box<Expr>, BinOp, Box<Expr>), Unary(UnOp, Box<Expr>),
    Index(Box<Expr>, Vec<Expr>), Field(Box<Expr>, String),
    Call(String, Vec<Expr>), TensorFactory { name: String, element: Type, shape: Box<Expr> },
    Located { node: Box<Expr>, location: SourceLocation },
}

pub struct LexedTokens { pub tokens: Vec<SpannedToken>, pub strings: StringInterner }

pub fn lex(source: &str) -> Result<LexedTokens, Error> {
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
                    "table" => Token::Table, "end" => Token::End, "as" => Token::As, "in" => Token::In, _ => Token::Ident(strings.intern(&word))
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
pub enum TableLiteralKey { Index(Expr), Name(String) }

#[derive(Clone, Debug)]
pub struct StructMethod { pub name: String, pub args: Vec<(String, Type)>, pub body: Option<Vec<Statement>> }

#[derive(Clone, Debug)]
pub enum MethodReceiver { Name(String), This }

#[derive(Clone, Debug)]
pub enum Statement {
    Struct { name: String, fields: Vec<(String, Type)>, methods: Vec<StructMethod> },
    MethodDefinition { struct_name: String, method: String, args: Vec<(String, Type)>, body: Vec<Statement> },
    ExportLet { name: StringId, ty: Type, expr: Expr },
    ExportStruct { name: String, fields: Vec<(String, Type)>, methods: Vec<StructMethod> },
    ExportFunction { name: String, body: Vec<Statement> },
    CallMethod { receiver: MethodReceiver, method: String },
    Let { name: StringId, ty: Type, expr: Expr },
    Assign { name: String, expr: Expr },
    GlobalAssign { name: String, expr: Expr },
    SetIndex { name: String, indices: Vec<Expr>, expr: Expr },
    SetField { name: String, field: String, expr: Expr },
    SetFieldIndex { name: String, field: String, index: Expr, expr: Expr },
    Print(Expr), Printf { format: Expr, args: Vec<Expr> }, Putc(Expr),
    If { condition: Expr, then_body: Vec<Statement>, else_body: Vec<Statement> },
    While { condition: Expr, body: Vec<Statement> },
    For { name: String, start: Expr, end: Expr, body: Vec<Statement> },
    ForIn { name: String, iterable: Expr, body: Vec<Statement> },
    Break,
    Continue,
    Located { node: Box<Statement>, location: SourceLocation },
}

pub struct Parser { tokens: Vec<SpannedToken>, strings: StringInterner, at: usize, last_location: SourceLocation }
impl Parser {
    pub fn new(lexed: LexedTokens) -> Self { Self { tokens: lexed.tokens, strings: lexed.strings, at: 0, last_location: SourceLocation { offset: 0, line: 1, column: 1 } } }
    fn string(&self, id: StringId) -> String { self.strings.resolve(id).to_owned() }
    fn peek(&self) -> &Token { &self.tokens[self.at].kind }
    fn location(&self) -> SourceLocation { self.tokens.get(self.at).map(|token| token.location).unwrap_or(self.last_location) }
    fn next(&mut self) -> Token { let token = self.tokens[self.at].clone(); self.at += 1; self.last_location = token.location; token.kind }
    fn need(&mut self, wanted: Token) -> Result<(), Error> { let got = self.next(); if got == wanted { Ok(()) } else { Err(Error::Parse(format!("expected {wanted:?}, got {got:?}"))) } }

    pub fn program(&mut self) -> Result<Vec<Statement>, Error> { self.block().map_err(|error| error.at(self.location())) }
    pub fn into_program(mut self) -> Result<(Vec<Statement>, StringInterner), Error> {
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
            match self.peek() {
                Token::Equal => {
                    self.next(); // Consume '='
                    let start = self.expr()?;
                    self.need(Token::Comma)?;
                    let end = self.expr()?;
                    self.need(Token::Do)?;
                    let body = self.block()?;
                    self.need(Token::End)?;
                    Ok(Statement::For { name, start, end, body })
                }
                Token::In => {
                    self.next(); // Consume 'in'
                    let iterable = self.expr()?;
                    self.need(Token::Do)?;
                    let body = self.block()?;
                    self.need(Token::End)?;
                    Ok(Statement::ForIn { name, iterable, body })
                }
                token => return Err(Error::Parse(format!("expected '=' or 'in' after loop variable, got {token:?}")))
            }
        },
        Token::Break => Ok(Statement::Break),
        Token::Continue => Ok(Statement::Continue),
        Token::This => { self.need(Token::Dot)?; let method = match self.next() { Token::Ident(method) => self.string(method), token => return Err(Error::Parse(format!("expected method name, got {token:?}"))) }; self.need(Token::LParen)?; self.need(Token::RParen)?; Ok(Statement::CallMethod { receiver: MethodReceiver::This, method }) },
        Token::DoubleColon => {
            let name = match self.next() { Token::Ident(name) => self.string(name), token => return Err(Error::Parse(format!("expected global variable name after '::', got {token:?}"))) };
            self.need(Token::Equal)?;
            Ok(Statement::GlobalAssign { name, expr: self.expr()? })
        },
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
            "f16" => Ok(Type::F16), "bf16" => Ok(Type::BF16), "f32" => Ok(Type::F32), "f64" => Ok(Type::F64), "bool" => Ok(Type::Bool),
            "table_key" => Ok(Type::TableKey), "table_keys" => Ok(Type::TableKeys),
            "vector" => { self.need(Token::Lt)?; let inner = self.ty()?; self.need(Token::Gt)?; Ok(Type::Array(Box::new(inner))) },
            "tensor" => {
                self.need(Token::Lt)?; let inner = self.ty()?; self.need(Token::Comma)?;
                let rank = match self.next() { Token::Integer(rank) if rank > 0 => usize::try_from(rank).map_err(|_| Error::Parse("tensor rank is too large".into()))?, token => return Err(Error::Parse(format!("tensor rank must be a positive integer, got {token:?}"))) };
                self.need(Token::Gt)?; scalar_size(&inner)?; Ok(Type::Tensor(Box::new(inner), rank))
            },
            "string" => Ok(Type::String),
            "dArray" => Ok(Type::DArray),
            "dTensor" => Ok(Type::DTensor),
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
            Token::DoubleColon => match self.next() {
                Token::Ident(name) => Expr::GlobalName(name),
                token => return Err(Error::Parse(format!("expected global variable name after '::', got {token:?}"))),
            },
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
pub enum TableEntry { Index, Name(Rc<str>) }

#[derive(Clone, Debug)]
pub enum ModuleExport { Value { slot: usize, ty: Type }, Function { entry: usize }, Struct(StructLayout) }

#[derive(Clone, Debug)]
pub enum BinaryOp {
    I8(BinOp), I16(BinOp), I32(BinOp), I64(BinOp),
    U8(BinOp), U16(BinOp), U32(BinOp), U64(BinOp),
    F16(BinOp), BF16(BinOp), F32(BinOp), F64(BinOp),
    Equal, NotEqual,
}

#[derive(Clone, Debug)]
pub struct ModuleArtifact { pub id: String, pub code: Rc<FlatBytecode>, pub exports: HashMap<String, ModuleExport> }

#[derive(Clone, Debug)]
pub struct HostSignature { pub arguments: Vec<Type>, pub result: Type }

#[derive(Clone, Debug)]
pub enum IrOp {
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
    TrackDestructor(usize, usize), UntrackDestructor(usize),
    Return, Print, Printf(usize), Putc
}

pub type Op = IrOp;

#[derive(Clone, Debug, Default)]
pub struct FlatBytecode { pub words: Vec<u32>, pub constants: Vec<Constant> }

#[derive(Clone, Debug)]
pub enum Constant {
    Value(Value), String(Rc<str>), Type(Rc<Type>), Module(Rc<ModuleArtifact>),
    Field(Rc<StructField>), Entries(Rc<[TableEntry]>), Layout(Rc<StructLayout>),
    Binary(BinaryOp), Unary(UnOp), Builtin(BuiltinFn), TensorInit(TensorInit),
}

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum Opcode {
    AddI32, AddF32, AddF64, Push, MakeString, Input, Require, Load, LoadCurrentReceiver,
    LoadCurrentField, Store, StoreIndex, StoreTableIndex, StoreTensorIndex, StoreTensorIndexF32,
    StoreField, StoreFieldIndex, StoreTableField, StoreCurrentField, MakeArray, MakeTable,
    MakeStruct, MakeTensor, Index, TensorIndex, TensorIndexF32, TableIndex, TableKeys,
    TableKeysIndex, TableRemove, Field, TableField, ModuleField, Binary, Unary, Len,
    ConcatString, Builtin1, Builtin2, CallExternal, JumpIfFalse, Jump, JumpIfFalseKeep,
    JumpIfTrueKeep, CallMethod, CallCurrentMethod, CallModule, Return, Print, Printf, Putc,
    Cast,
    TrackDestructor, UntrackDestructor,
}

impl Opcode {
    pub fn from_word(word: u32) -> Result<Self, Error> {
        if word > Self::UntrackDestructor as u32 { return Err(Error::Runtime("invalid bytecode opcode".into())); }
        Ok(unsafe { std::mem::transmute(word) })
    }
}

pub enum DecodedOp<'a> {
    AddI32, AddF32, AddF64, Push(&'a Value), MakeString(&'a Rc<str>), Input(&'a Rc<Type>), Require(&'a Rc<ModuleArtifact>), Load(usize), LoadCurrentReceiver,
    LoadCurrentField(&'a Rc<StructField>), Store(usize), StoreIndex(usize, &'a Rc<Type>), StoreTableIndex(usize, &'a Rc<Type>),
    StoreTensorIndex(usize, &'a Rc<Type>, usize), StoreTensorIndexF32(usize, usize),
    StoreField(usize, &'a Rc<StructField>), StoreFieldIndex(usize, &'a Rc<StructField>, &'a Rc<Type>), StoreTableField(usize, &'a Rc<str>, &'a Rc<Type>),
    StoreCurrentField(&'a Rc<StructField>), MakeArray(usize, &'a Rc<Type>), MakeTable(&'a Rc<[TableEntry]>, &'a Rc<Type>), MakeStruct(&'a Rc<StructLayout>),
    MakeTensor(TensorInit, &'a Rc<Type>, usize), Index, TensorIndex(&'a Rc<Type>, usize), TensorIndexF32(usize), TableIndex, TableKeys, TableKeysIndex, TableRemove, Field(&'a Rc<StructField>), TableField(&'a Rc<str>), ModuleField(&'a Rc<str>),
    Binary(&'a BinaryOp), Unary(&'a UnOp, &'a Rc<Type>), Len, ConcatString, Cast(&'a Rc<Type>),
    Builtin1(BuiltinFn, &'a Rc<Type>), Builtin2(BuiltinFn, &'a Rc<Type>), CallExternal(&'a Rc<str>, usize),
    JumpIfFalse(usize), Jump(usize), JumpIfFalseKeep(usize), JumpIfTrueKeep(usize),
    CallMethod(usize, usize), CallCurrentMethod(usize), CallModule(usize, &'a Rc<str>),
    TrackDestructor(usize, usize), UntrackDestructor(usize),
    Return, Print, Printf(usize), Putc,
}

impl FlatBytecode {
    fn constant(&mut self, constant: Constant) -> u32 { let index = self.constants.len() as u32; self.constants.push(constant); index }
    fn word(&mut self, value: usize) { self.words.push(value as u32); }
    fn op(&mut self, opcode: Opcode) { self.words.push(opcode as u32); }
    pub fn lower(ir: Vec<IrOp>) -> Self {
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
            IrOp::TrackDestructor(a,b) => { out.op(Opcode::TrackDestructor); out.word(a); out.word(b * 4); }, IrOp::UntrackDestructor(v) => { out.op(Opcode::UntrackDestructor); out.word(v); },
            IrOp::Return => out.op(Opcode::Return), IrOp::Print => out.op(Opcode::Print), IrOp::Printf(v) => { out.op(Opcode::Printf); out.word(v); }, IrOp::Putc => out.op(Opcode::Putc),
        } while out.words.len() % 4 != 0 { out.words.push(0); } } out
    }

    #[inline(always)]
    pub fn decode(&self, mut pc: usize) -> Result<(DecodedOp<'_>, usize), Error> {
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
            Opcode::JumpIfFalse=>DecodedOp::JumpIfFalse(word(&mut pc)?), Opcode::Jump=>DecodedOp::Jump(word(&mut pc)?), Opcode::JumpIfFalseKeep=>DecodedOp::JumpIfFalseKeep(word(&mut pc)?), Opcode::JumpIfTrueKeep=>DecodedOp::JumpIfTrueKeep(word(&mut pc)?), Opcode::CallMethod=>DecodedOp::CallMethod(word(&mut pc)?,word(&mut pc)?), Opcode::CallCurrentMethod=>DecodedOp::CallCurrentMethod(word(&mut pc)?), Opcode::CallModule=>{let a=word(&mut pc)?;c!(Constant::String(b));DecodedOp::CallModule(a,b)}, Opcode::TrackDestructor=>DecodedOp::TrackDestructor(word(&mut pc)?,word(&mut pc)?), Opcode::UntrackDestructor=>DecodedOp::UntrackDestructor(word(&mut pc)?), Opcode::Return=>DecodedOp::Return, Opcode::Print=>DecodedOp::Print, Opcode::Printf=>DecodedOp::Printf(word(&mut pc)?), Opcode::Putc=>DecodedOp::Putc,
        }; Ok((decoded,instruction_start + 4))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltinFn {
    Sqrt, Sin, Cos, Tan, Asin, Acos, Atan, Floor, Ceil, Round, Abs,
    Pow, Min, Max, Atan2,
}

impl BuiltinFn {
    pub fn unary(name: &str) -> Option<Self> {
        Some(match name {
            "sqrt" => Self::Sqrt, "sin" => Self::Sin, "cos" => Self::Cos,
            "tan" => Self::Tan, "asin" => Self::Asin, "acos" => Self::Acos,
            "atan" => Self::Atan, "floor" => Self::Floor, "ceil" => Self::Ceil,
            "round" => Self::Round, "abs" => Self::Abs,
            _ => return None,
        })
    }

    pub fn binary(name: &str) -> Option<Self> {
        Some(match name {
            "pow" => Self::Pow, "min" => Self::Min, "max" => Self::Max,
            "atan2" => Self::Atan2,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum TensorInit { Zeros, Random }

#[derive(Clone, Debug)]
pub struct DestructorLocal { pub slot: usize, pub struct_name: String }

#[derive(Default)]
pub struct ScopeContext { pub destructors: Vec<DestructorLocal> }

pub struct LoopContext {
    pub break_jumps: Vec<usize>,
    pub continue_jumps: Vec<usize>,
    pub continue_target: usize,
    pub scope_base: usize,
}

pub struct Compiler {
    pub names: HashMap<String, (usize, Type)>,
    pub structs: HashMap<String, StructLayout>,
    pub globals: HashMap<String, (usize, Type)>,
    pub methods: HashMap<(String, String), Option<usize>>,
    pub pending_method_calls: Vec<(usize, String, String)>,
    pub current_method_fields: Option<HashMap<String, StructField>>,
    pub current_method_struct: Option<String>,
    pub module_root: Option<PathBuf>,
    pub module_artifacts: HashMap<String, ModuleArtifact>,
    pub compiling_modules: Rc<RefCell<HashSet<String>>>,
    pub exports: HashMap<String, ModuleExport>,
    pub extern_functions: HashMap<String, HostSignature>,
    pub code: Vec<Op>,
    pub interned_names: HashMap<String, Rc<str>>,
    pub strings: StringInterner,
    pub next_slot: usize,
    pub scope_depth: usize,
    pub scope_stack: Vec<ScopeContext>,
    pub loops: Vec<LoopContext>
}

impl Default for Compiler {
    fn default() -> Self {
        Self {
            names: HashMap::new(), structs: HashMap::new(), globals: HashMap::new(), methods: HashMap::new(), pending_method_calls: Vec::new(), current_method_fields: None, current_method_struct: None, module_root: None, module_artifacts: HashMap::new(), compiling_modules: Rc::new(RefCell::new(HashSet::new())), exports: HashMap::new(), extern_functions: HashMap::new(), code: Vec::new(), interned_names: HashMap::new(), strings: StringInterner::new(), next_slot: 0, scope_depth: 0, scope_stack: Vec::new(), loops: Vec::new()
        }
    }
}

impl Compiler {
    pub fn with_module_root(module_root: PathBuf) -> Self { Self { module_root: Some(module_root), ..Self::default() } }
    pub fn with_extern_functions(extern_functions: HashMap<String, HostSignature>) -> Self { Self { extern_functions, ..Self::default() } }
    pub fn with_strings(mut self, strings: StringInterner) -> Self { self.strings = strings; self }
    fn string(&self, id: StringId) -> &str { self.strings.resolve(id) }

    fn intern_name(&mut self, name: &str) -> Rc<str> {
        if let Some(interned) = self.interned_names.get(name) { return interned.clone(); }
        let interned: Rc<str> = Rc::from(name);
        self.interned_names.insert(name.to_owned(), interned.clone());
        interned
    }

    fn scoped_block(&mut self, body: Vec<Statement>) -> Result<(), Error> {
        let saved_names = self.names.clone();
        let saved_next_slot = self.next_slot;
        self.scope_depth += 1;
        self.scope_stack.push(ScopeContext::default());
        let result = body.into_iter().try_for_each(|statement| self.statement(statement));
        if result.is_ok() { self.emit_current_scope_destructors()?; }
        self.scope_stack.pop();
        self.scope_depth -= 1;
        self.names = saved_names;
        self.next_slot = saved_next_slot;
        result
    }

    fn destructor_target(&self, struct_name: &str) -> Option<usize> {
        self.methods.get(&(struct_name.to_owned(), struct_name.to_owned())).copied().flatten()
    }

    fn has_destructor_declaration(&self, struct_name: &str) -> bool {
        self.methods.contains_key(&(struct_name.to_owned(), struct_name.to_owned()))
    }

    fn emit_destructor_call(&mut self, destructor: &DestructorLocal) {
        let target = self.destructor_target(&destructor.struct_name).unwrap_or(usize::MAX);
        let position = self.code.len();
        self.code.push(Op::CallMethod(destructor.slot, target));
        if target == usize::MAX {
            self.pending_method_calls.push((position, destructor.struct_name.clone(), destructor.struct_name.clone()));
        }
    }

    fn emit_destructor_sequence(&mut self, destructors: impl IntoIterator<Item = DestructorLocal>) {
        for destructor in destructors {
            self.code.push(Op::UntrackDestructor(destructor.slot));
            self.emit_destructor_call(&destructor);
        }
    }

    fn emit_current_scope_destructors(&mut self) -> Result<(), Error> {
        let destructors = self.scope_stack.last()
            .ok_or_else(|| Error::Runtime("missing compiler scope for destructors".into()))?
            .destructors.iter().cloned().rev().collect::<Vec<_>>();
        self.emit_destructor_sequence(destructors);
        Ok(())
    }

    fn emit_destructors_from_scope(&mut self, scope_base: usize) -> Result<(), Error> {
        let destructors = self.scope_stack.get(scope_base..)
            .ok_or_else(|| Error::Runtime("invalid loop scope for destructors".into()))?
            .iter().rev().flat_map(|scope| scope.destructors.iter().rev().cloned()).collect::<Vec<_>>();
        self.emit_destructor_sequence(destructors);
        Ok(())
    }

    fn track_destructor(&mut self, slot: usize, struct_name: &str) -> Result<(), Error> {
        if !self.has_destructor_declaration(struct_name) { return Ok(()); }
        let target = self.destructor_target(struct_name).unwrap_or(usize::MAX);
        let position = self.code.len();
        self.code.push(Op::TrackDestructor(slot, target));
        if target == usize::MAX {
            self.pending_method_calls.push((position, struct_name.to_owned(), struct_name.to_owned()));
        }
        self.scope_stack.last_mut()
            .ok_or_else(|| Error::Runtime("missing compiler scope for destructor".into()))?
            .destructors.push(DestructorLocal { slot, struct_name: struct_name.to_owned() });
        Ok(())
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
                Type::F16 => Ok(BinaryOp::F16(op)), Type::BF16 => Ok(BinaryOp::BF16(op)), Type::F32 => Ok(BinaryOp::F32(op)),
                Type::F64 => Ok(BinaryOp::F64(op)),
                _ => Err(Error::Type(format!("unsupported binary operand type {ty}"))),
            },
        }
    }

    pub fn compile(mut self, program: Vec<Statement>) -> Result<FlatBytecode, Error> { self.compile_program(program)?; Ok(FlatBytecode::lower(self.code)) }

    pub fn compile_module(mut self, id: String, program: Vec<Statement>) -> Result<ModuleArtifact, Error> {
        self.compile_program(program)?;
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
                Some(Op::CallMethod(_, call_target)) | Some(Op::CallCurrentMethod(call_target)) | Some(Op::TrackDestructor(_, call_target)) => *call_target = target,
                _ => return Err(Error::Runtime("invalid pending method call".into())),
            }
        }
        Ok(())
    }

    // ИЗМЕНЁННЫЙ МЕТОД: проверяет наличие нативного модуля перед попыткой чтения с диска
    fn load_module(&mut self, requested: &str) -> Result<ModuleArtifact, Error> {
        // 1. Проверяем наличие нативного расширения (например, "candle")
        let ext_names: Vec<String> = crate::ext::available_extensions()
            .into_iter().map(|e| e.name().to_string()).collect();

        if ext_names.contains(&requested.to_string()) {
            let artifact = ModuleArtifact {
                id: requested.to_string(),
                code: Rc::new(FlatBytecode::default()),
                exports: HashMap::new(),
            };
            self.module_artifacts.insert(requested.to_string(), artifact.clone());
            return Ok(artifact);
        }

        // 2. Старый код загрузки .l0 файла с диска
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
        if method_name == struct_name && !args.is_empty() {
            return Err(Error::Type(format!("destructor '{struct_name}' must not accept arguments")));
        }
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
        let saved_names = std::mem::take(&mut self.names);
        let saved_scope_depth = self.scope_depth;
        let saved_scope_stack = std::mem::take(&mut self.scope_stack);
        self.scope_depth += 1;
        self.scope_stack.push(ScopeContext::default());

        for (arg_name, arg_ty) in args.into_iter().rev() {
            let slot = self.next_slot;
            self.next_slot += 1;
            self.names.insert(arg_name, (slot, arg_ty));
            self.code.push(Op::Store(slot));
        }
        let body_result = body.into_iter().try_for_each(|statement| self.statement(statement));
        if body_result.is_ok() { self.emit_current_scope_destructors()?; }
        self.scope_stack.pop();
        self.current_method_fields = previous_fields;
        self.current_method_struct = previous_struct;
        self.names = saved_names;
        self.scope_stack = saved_scope_stack;

        self.scope_depth = saved_scope_depth;
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
                if method.name == name && !method.args.is_empty() {
                    return Err(Error::Type(format!("destructor '{name}' must not accept arguments")));
                }
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
            if is_numeric(&ty) && ty != found { self.code.push(Op::Cast(Rc::new(ty.clone()))); }
            let is_new_binding = !self.names.contains_key(&name);
            let slot = if let Some((existing_slot, existing_ty)) = self.names.get(&name) {
                if existing_ty != &ty { return Err(Error::Type(format!("cannot redefine '{name}' with a different type"))); }
                *existing_slot
            } else {
                let s = self.next_slot;
                self.next_slot += 1;
                self.names.insert(name.clone(), (s, ty.clone()));
                 if self.scope_depth == 0 {
                     self.globals.insert(name.clone(), (s, ty.clone()));
                }
                s
            };
            if let Type::Module(module_id) = &found { self.import_exported_structs(&name, module_id)?; }
            self.code.push(Op::Store(slot));
            if is_new_binding && self.scope_depth > 0 {
                if let Type::Struct(struct_name) = found { self.track_destructor(slot, &struct_name)?; }
            }
            Ok(())
        },
        Statement::Assign { name, expr } => {
            if let Some(field) = self.current_method_fields.as_ref().and_then(|fields| fields.get(&name)).cloned() {
                let found = self.expr(expr, Some(&field.ty))?;
                if !types_compatible(&field.ty, &found) { return Err(Error::Type(format!("field '{name}' is {}, but expression has type {found}", field.ty))); }
                if is_numeric(&field.ty) && field.ty != found { self.code.push(Op::Cast(Rc::new(field.ty.clone()))); }
                self.code.push(Op::StoreCurrentField(Rc::new(field)));
                Ok(())
            } else {
                let (slot, ty) = self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?;
                let found = self.expr(expr, Some(&ty))?;
                if !types_compatible(&ty, &found) { return Err(Error::Type(format!("'{name}' is {ty}, but expression has type {found}"))); }
                if is_numeric(&ty) && ty != found { self.code.push(Op::Cast(Rc::new(ty.clone()))); }
                self.code.push(Op::Store(slot));
                Ok(())
            }
        },
        Statement::GlobalAssign { name, expr } => { let (slot, ty) = self.globals.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown global name '{name}'")))?; let found = self.expr(expr, Some(&ty))?; if !types_compatible(&ty, &found) { return Err(Error::Type(format!("global '{name}' is {ty}, but expression has type {found}"))); } self.code.push(Op::Store(slot)); Ok(()) },
        Statement::SetIndex { name, indices, expr } => {
            let (slot, container_ty) = if let Some(field) = self.current_method_fields.as_ref().and_then(|fields| fields.get(&name)).cloned() {
                let temp_slot = self.next_slot;
                self.next_slot += 1;
                self.code.push(Op::LoadCurrentField(Rc::new(field.clone())));
                self.code.push(Op::Store(temp_slot));
                (temp_slot, field.ty)
            } else {
                self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?
            };

            match container_ty {
                Type::Array(inner) => {
                    if indices.len() != 1 { return Err(Error::Type("vector indexing requires exactly one index".into())); }
                    scalar_size(&inner)?; self.compile_tensor_indices(indices)?;
                    let element = *inner; let found = self.expr(expr, Some(&element))?;
                    if !types_compatible(&element, &found) { return Err(Error::Type(format!("item is {found}; expected {element}"))); }
                    self.code.push(Op::StoreIndex(slot, Rc::new(element))); Ok(())
                },
                Type::Table(inner) => {
                    if indices.len() != 1 { return Err(Error::Type("table indexing requires exactly one index".into())); }
                    self.compile_table_key(indices.into_iter().next().expect("checked table index count"))?;
                    let element = *inner; let found = self.expr(expr, Some(&element))?;
                    if !types_compatible(&element, &found) { return Err(Error::Type(format!("item is {found}; expected {element}"))); }
                    self.code.push(Op::StoreTableIndex(slot, Rc::new(element))); Ok(())
                },
                Type::Tensor(inner, rank) => {
                    if indices.len() != rank { return Err(Error::Type(format!("tensor rank {rank} requires {rank} index(es)"))); }
                    self.compile_tensor_indices(indices)?;
                    let element = *inner; let found = self.expr(expr, Some(&element))?;
                    if !types_compatible(&element, &found) { return Err(Error::Type(format!("tensor item is {found}; expected {element}"))); }
                    if element == Type::F32 { self.code.push(Op::StoreTensorIndexF32(slot, rank)); } else { self.code.push(Op::StoreTensorIndex(slot, Rc::new(element), rank)); } Ok(())
                },
                _ => Err(Error::Type(format!("'{name}' is not indexable"))),
            }
        },
        Statement::SetField { name, field, expr } => {
            let (slot, ty) = if let Some(method_field) = self.current_method_fields.as_ref().and_then(|fields| fields.get(&name)).cloned() {
                let temp_slot = self.next_slot;
                self.next_slot += 1;
                self.code.push(Op::LoadCurrentField(Rc::new(method_field.clone())));
                self.code.push(Op::Store(temp_slot));
                (temp_slot, method_field.ty)
            } else {
                self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?
            };

            match ty {
                Type::Struct(struct_name) => {
                    let layout = self.structs.get(&struct_name).ok_or_else(|| Error::Type(format!("unknown struct '{struct_name}'")))?;
                    let field = layout.fields.iter().find(|candidate| candidate.name == field).cloned().ok_or_else(|| Error::Type(format!("struct '{struct_name}' has no field '{field}'")))?;
                    let found = self.expr(expr, Some(&field.ty))?;
                    if !types_compatible(&field.ty, &found) { return Err(Error::Type("struct field type mismatch".into())); }
                    self.code.push(Op::StoreField(slot, Rc::new(field)));
                    Ok(())
                },
                Type::Table(element) => {
                    let element = *element;
                    let found = self.expr(expr, Some(&element))?;
                    if !types_compatible(&element, &found) { return Err(Error::Type("table value type mismatch".into())); }
                    let field = self.intern_name(&field);
                    self.code.push(Op::StoreTableField(slot, field, Rc::new(element)));
                    Ok(())
                },
                _ => Err(Error::Type(format!("'{name}' has no named keys"))),
            }
        },
        Statement::SetFieldIndex { name, field, index, expr } => {
            let (slot, ty) = if let Some(method_field) = self.current_method_fields.as_ref().and_then(|fields| fields.get(&name)).cloned() {
                let temp_slot = self.next_slot;
                self.next_slot += 1;
                self.code.push(Op::LoadCurrentField(Rc::new(method_field.clone())));
                self.code.push(Op::Store(temp_slot));
                (temp_slot, method_field.ty)
            } else {
                self.names.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown name '{name}'")))?
            };

            let Type::Struct(struct_name) = ty else { return Err(Error::Type(format!("'{name}' is not a struct"))); };
            let layout = self.structs.get(&struct_name).ok_or_else(|| Error::Type(format!("unknown struct '{struct_name}'")))?;
            let field = layout.fields.iter().find(|candidate| candidate.name == field).cloned().ok_or_else(|| Error::Type(format!("struct '{struct_name}' has no field '{field}'")))?;
            let Type::Array(element) = field.ty.clone() else { return Err(Error::Type(format!("field '{}' is not a vector", field.name))); };
            let element = *element; scalar_size(&element)?;
            let index_ty = self.expr(index, None)?;
            if !matches!(index_ty, Type::I8|Type::I16|Type::I32|Type::I64|Type::U8|Type::U16|Type::U32|Type::U64) { return Err(Error::Type("index must be an integer".into())); }
            let found = self.expr(expr, Some(&element))?;
            if !types_compatible(&element, &found) { return Err(Error::Type(format!("item is {found}; expected {element}"))); }
            self.code.push(Op::StoreFieldIndex(slot, Rc::new(field), Rc::new(element)));
            Ok(())
        },
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
            self.loops.push(LoopContext { break_jumps: Vec::new(), continue_jumps: Vec::new(), continue_target: loop_start, scope_base: self.scope_stack.len() });
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
            self.code.push(Op::Store(end_slot));
            self.code.push(Op::Store(index_slot));
            self.names.insert(name.clone(), (index_slot, Type::I32));
            let loop_start = self.code.len();
            self.code.push(Op::Load(index_slot));
            self.code.push(Op::Load(end_slot));
            self.code.push(Op::Binary(BinaryOp::I32(BinOp::Le)));
            let exit_jump = self.code.len();
            self.code.push(Op::JumpIfFalse(usize::MAX));
            self.loops.push(LoopContext { break_jumps: Vec::new(), continue_jumps: Vec::new(), continue_target: usize::MAX, scope_base: self.scope_stack.len() });
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
        Statement::ForIn { name, iterable, body } => {
            if self.names.contains_key(&name) { return Err(Error::Type(format!("loop variable '{name}' is already defined"))); }

            let iterable_ty = self.expr(iterable, None)?;
            let element_ty = match &iterable_ty {
                Type::Array(inner) => (**inner).clone(),
                Type::TableKeys => Type::TableKey,
                Type::Tensor(inner, 1) => (**inner).clone(),
                _ => return Err(Error::Type(format!("cannot iterate over {iterable_ty}; 'for ... in' requires a vector, 1D tensor, or table_keys"))),
            };
            let iterable_slot = self.next_slot; self.next_slot += 1;
            let len_slot      = self.next_slot; self.next_slot += 1;
            let index_slot    = self.next_slot; self.next_slot += 1;
            let item_slot     = self.next_slot; self.next_slot += 1;

            self.code.push(Op::Store(iterable_slot));

            self.code.push(Op::Load(iterable_slot));
            self.code.push(Op::Len);
            self.code.push(Op::Store(len_slot));

            self.code.push(Op::Push(Value::I32(0)));
            self.code.push(Op::Store(index_slot));

            self.names.insert(name.clone(), (item_slot, element_ty.clone()));

            let loop_start = self.code.len();

            self.code.push(Op::Load(index_slot));
            self.code.push(Op::Load(len_slot));
            self.code.push(Op::Binary(BinaryOp::I32(BinOp::Lt)));
            let exit_jump = self.code.len();
            self.code.push(Op::JumpIfFalse(usize::MAX));

            self.code.push(Op::Load(iterable_slot));
            self.code.push(Op::Load(index_slot));
            match &iterable_ty {
                Type::Array(_) => self.code.push(Op::Index),
                Type::TableKeys => self.code.push(Op::TableKeysIndex),
                Type::Tensor(inner, 1) => {
                    if **inner == Type::F32 {
                        self.code.push(Op::TensorIndexF32(1));
                    } else {
                        self.code.push(Op::TensorIndex(Rc::new((**inner).clone()), 1));
                    }
                },
                _ => unreachable!(),
            }
            self.code.push(Op::Store(item_slot));

            self.loops.push(LoopContext { break_jumps: Vec::new(), continue_jumps: Vec::new(), continue_target: usize::MAX, scope_base: self.scope_stack.len() });
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
            self.next_slot = iterable_slot;
            Ok(())
        },
        Statement::Break => {
            let scope_base = self.loops.last().ok_or_else(|| Error::Type("break is available only inside a loop".into()))?.scope_base;
            self.emit_destructors_from_scope(scope_base)?;
            let jump = self.code.len();
            self.code.push(Op::Jump(usize::MAX));
            self.loops.last_mut().expect("checked loop context").break_jumps.push(jump);
            Ok(())
        },
        Statement::Continue => {
            let scope_base = self.loops.last().ok_or_else(|| Error::Type("continue is available only inside a loop".into()))?.scope_base;
            self.emit_destructors_from_scope(scope_base)?;
            let jump = self.code.len();
            self.code.push(Op::Jump(usize::MAX));
            self.loops.last_mut().expect("checked loop context").continue_jumps.push(jump);
            Ok(())
        },
    }
    }

    fn expr(&mut self, expr: Expr, expected: Option<&Type>) -> Result<Type, Error> { match expr {
        Expr::Located { node, location } => return self.expr(*node, expected).map_err(|error| error.at(location)),
        Expr::Integer(n) => { let ty = expected.unwrap_or(&Type::I32); let val = int_value(n, ty)?; self.code.push(Op::Push(val)); Ok(ty.clone()) },
        Expr::Float(n) => {
            let ty = expected.unwrap_or(&Type::F64);
            if !matches!(ty, Type::F16 | Type::BF16 | Type::F32 | Type::F64) {
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
        Expr::GlobalName(name) => { let name = self.string(name); let (slot, ty) = self.globals.get(name).cloned().ok_or_else(|| Error::Type(format!("unknown global name '{name}'")))?; self.code.push(Op::Load(slot)); Ok(ty) },
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
                if !types_compatible(&element, &found) { return Err(Error::Type(format!("array item is {found}; expected {element}"))); }
            }
            self.code.push(Op::MakeArray(count, Rc::new(element.clone())));
            Ok(Type::Array(Box::new(element)))
        },
        Expr::Table(items) => { let element = match expected { Some(Type::Table(t)) => (**t).clone(), _ => return Err(Error::Type("table needs an explicit value type, e.g. table<i32>".into())) }; let mut entries = Vec::with_capacity(items.len()); for (key, value) in items { match key { TableLiteralKey::Name(name) => { let name = self.intern_name(&name); entries.push(TableEntry::Name(name)); }, TableLiteralKey::Index(index) => { self.compile_table_key(index)?; entries.push(TableEntry::Index); } } let found = self.expr(value, Some(&element))?; if !types_compatible(&element, &found) { return Err(Error::Type(format!("table value is {found}; expected {element}"))); } } self.code.push(Op::MakeTable(entries.into(), Rc::new(element.clone()))); Ok(Type::Table(Box::new(element))) },
        Expr::TensorFactory { name, element, shape } => {
            let Type::Tensor(expected_element, rank) = expected.cloned().ok_or_else(|| Error::Type(format!("{name}<T> needs an expected tensor type, e.g. let x: tensor<f32, 2> = {name}<f32>([2, 3])")))? else { return Err(Error::Type(format!("{name}<{}> creates a tensor, but the expected type is not tensor", element))); };
            if expected_element.as_ref() != &element { return Err(Error::Type(format!("{name} element type is {element}; expected {expected_element}"))); }
            let shape_ty = self.expr(*shape, Some(&Type::Array(Box::new(Type::U64))))?;
            if shape_ty != Type::Array(Box::new(Type::U64)) { return Err(Error::Type("tensor shape must be vector<u64>".into())); }
            let init = if name == "zeros" { TensorInit::Zeros } else { TensorInit::Random };
            if matches!(init, TensorInit::Random) && !matches!(element, Type::F16 | Type::BF16 | Type::F32 | Type::F64) { return Err(Error::Type("random<T> supports f16, bf16, f32, and f64 tensors only".into())); }
            self.code.push(Op::MakeTensor(init, Rc::new(element.clone()), rank));
            Ok(Type::Tensor(Box::new(element), rank))
        },
        Expr::StructLiteral(name, fields) => { let layout = self.structs.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown struct '{name}'")))?; if let Some(expected) = expected { if expected != &Type::Struct(name.clone()) { return Err(Error::Type(format!("expected {expected}, got {name}"))); } } if fields.len() != layout.fields.len() { return Err(Error::Type(format!("struct '{name}' needs {} field(s)", layout.fields.len()))); } let mut provided = HashMap::new(); for (field_name, field_expr) in fields { if provided.insert(field_name.clone(), field_expr).is_some() { return Err(Error::Type(format!("struct '{name}' initializes field '{field_name}' more than once"))); } } for field in &layout.fields { let expr = provided.remove(&field.name).ok_or_else(|| Error::Type(format!("struct '{name}' is missing field '{}'", field.name)))?; let found = self.expr(expr, Some(&field.ty))?; if !types_compatible(&field.ty, &found) { return Err(Error::Type(format!("field '{}' is {found}; expected {}", field.name, field.ty))); } } if let Some((unknown, _)) = provided.into_iter().next() { return Err(Error::Type(format!("struct '{name}' has no field '{unknown}'"))); } self.code.push(Op::MakeStruct(Rc::new(layout))); Ok(Type::Struct(name)) },
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
                    if !matches!(ty, Type::Array(_) | Type::String | Type::Table(_) | Type::TableKeys | Type::Tensor(_, _) | Type::DArray | Type::DTensor) { return Err(Error::Type(format!("len requires a vector, string, table, table_keys, or tensor, got {ty}"))); }
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
                    if !matches!(ty, Type::F16 | Type::BF16 | Type::F32 | Type::F64) { return Err(Error::Type(format!("{} requires a float argument, got {}", name, ty))); }
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
                    if name == "atan2" && !matches!(t1, Type::F16 | Type::BF16 | Type::F32 | Type::F64) { return Err(Error::Type("atan2 requires float arguments".into())); }
                    self.code.push(Op::Builtin2(BuiltinFn::binary(&name).expect("known binary built-in"), Rc::new(t1.clone())));
                    Ok(t1)
                },
                _ => {
                    let signature = self.extern_functions.get(&name).cloned().ok_or_else(|| Error::Type(format!("unknown function '{name}'")))?;
                    if args.len() != signature.arguments.len() { return Err(Error::Type(format!("{name} expects {} argument(s)", signature.arguments.len()))); }
                    for (argument, expected_type) in args.into_iter().zip(signature.arguments.iter()) {
                        let found = self.expr(argument, Some(expected_type))?;
                        if !types_compatible(expected_type, &found) { return Err(Error::Type(format!("{name} argument is {found}; expected {expected_type}"))); }
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
                if op == BinOp::And {
                    self.code.push(Op::JumpIfFalseKeep(usize::MAX));
                } else {
                    self.code.push(Op::JumpIfTrueKeep(usize::MAX));
                }

                let rt = self.expr(*right, Some(&Type::Bool))?;
                if rt != Type::Bool { return Err(Error::Type("logical operators require bool".into())); }

                let end_idx = self.code.len();
                if op == BinOp::And {
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
