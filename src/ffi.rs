use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};

// Импортируем типы из ядра (lib.rs) и компилятора (compiler.rs)
use crate::{ExternalFunction, FfiCall, Type, Value, Vm};
use crate::compiler::HostSignature;

pub const ABI_VERSION: u32 = 3;

/// Opaque C ABI state. Only this crate may access its interior.
#[repr(C)]
pub struct L0State {
    pub(crate) vm: Vm,
    pub(crate) ffi_call: Option<FfiCall>,
    pub(crate) is_executing: bool,
    pub(crate) last_error: Option<CString>,
}

pub type L0CFunction = unsafe extern "C" fn(*mut L0State) -> c_int;

/// Stable scalar type IDs accepted by the C FFI registration API.
///
/// C values are decoded from `c_int` before becoming this enum, so invalid C
/// input cannot create an invalid Rust enum discriminant.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum L0TypeId {
    I8 = 0, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64,
    Bool, DArray, DTensor,
}

impl L0TypeId {
    fn from_raw(value: c_int) -> Option<Self> {
        Some(match value {
            0 => Self::I8, 1 => Self::I16, 2 => Self::I32, 3 => Self::I64,
            4 => Self::U8, 5 => Self::U16, 6 => Self::U32, 7 => Self::U64,
            8 => Self::F16, 9 => Self::F32, 10 => Self::F64, 11 => Self::Bool,
            12 => Self::DArray, 13 => Self::DTensor,
            _ => return None,
        })
    }

    fn to_l0_type(self) -> Type {
        match self {
            Self::I8 => Type::I8, Self::I16 => Type::I16, Self::I32 => Type::I32, Self::I64 => Type::I64,
            Self::U8 => Type::U8, Self::U16 => Type::U16, Self::U32 => Type::U32, Self::U64 => Type::U64,
            Self::F16 => Type::F16, Self::F32 => Type::F32, Self::F64 => Type::F64, Self::Bool => Type::Bool,
            Self::DArray => Type::DArray, Self::DTensor => Type::DTensor,
        }
    }
}

fn c_scalar_type(value: c_int) -> Option<Type> {
    L0TypeId::from_raw(value).map(L0TypeId::to_l0_type)
}

fn ffi_argument(state: &L0State, index: usize) -> Option<&Value> {
    if let Some(call) = state.ffi_call.as_ref() {
        call.arguments.get(index)
    } else {
        state.vm.stack[..state.vm.stack_ptr].get(index)
    }
}

fn ffi_push(state: &mut L0State, value: Value) {
    if let Some(call) = state.ffi_call.as_mut() {
        call.results.push(value);
    } else {
        state.vm.push(value);
    }
}

macro_rules! c_scalar_helpers {
    ($push:ident, $read:ident, $variant:ident, $ty:ty) => {
        #[no_mangle]
        pub unsafe extern "C" fn $push(state: *mut L0State, value: $ty) {
            if let Some(state) = state.as_mut() {
                ffi_push(state, Value::$variant(value));
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn $read(state: *mut L0State, index: usize, out: *mut $ty) -> c_int {
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

#[no_mangle]
pub unsafe extern "C" fn l0_push_bool(state: *mut L0State, value: c_int) {
    if let Some(state) = state.as_mut() {
        ffi_push(state, Value::Bool(value != 0));
    }
}

#[no_mangle]
pub unsafe extern "C" fn l0_to_bool(state: *mut L0State, index: usize, out: *mut c_int) -> c_int {
    let Some(state) = state.as_ref() else { return 0 };
    let Some(out) = out.as_mut() else { return 0 };
    let Some(Value::Bool(value)) = ffi_argument(state, index) else { return 0 };
    *out = c_int::from(*value);
    1
}

#[no_mangle]
pub extern "C" fn l0_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn l0_new_state() -> *mut L0State {
    Box::into_raw(Box::new(L0State {
        vm: Vm::default(),
        ffi_call: None,
        is_executing: false,
        last_error: None
    }))
}

/// # Safety
/// `state` must be valid. The returned pointer remains valid until the next
/// operation on this state and must not be freed by the caller.
#[no_mangle]
pub unsafe extern "C" fn l0_last_error(state: *const L0State) -> *const c_char {
    state.as_ref()
        .and_then(|state| state.last_error.as_ref())
        .map_or(std::ptr::null(), |message| message.as_ptr())
}

/// # Safety
/// `state` must have been returned by `l0_new_state` and not freed already.
#[no_mangle]
pub unsafe extern "C" fn l0_free_state(state: *mut L0State) {
    if !state.is_null() { drop(Box::from_raw(state)); }
}

/// # Safety
/// `state` must be valid and `name` must be a NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn l0_register_i32_function(
    state: *mut L0State,
    name: *const c_char,
    function: L0CFunction,
    argument_count: usize
) -> c_int {
    let Some(state) = state.as_mut() else { return 0 };
    if name.is_null() { return 0; }
    let Ok(name) = CStr::from_ptr(name).to_str() else { return 0; };
    match state.vm.register_c_i32_function(name, argument_count, function) {
        Ok(()) => 1,
        Err(_) => 0
    }
}

/// # Safety
/// `state` must be valid, `name` must be NUL-terminated UTF-8, and `arg_types`
/// must address `argument_count` type IDs when that count is nonzero.
#[no_mangle]
pub unsafe extern "C" fn l0_register_c_function(
    state: *mut L0State,
    name: *const c_char,
    function: L0CFunction,
    arg_types: *const c_int,
    argument_count: usize,
    result_type: c_int
) -> c_int {
    let Some(state) = state.as_mut() else { return 0 };
    if name.is_null() || (argument_count != 0 && arg_types.is_null()) { return 0; }
    let Ok(name) = CStr::from_ptr(name).to_str() else { return 0; };
    let Some(result) = c_scalar_type(result_type) else { return 0; };
    let raw_arguments = if argument_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(arg_types, argument_count)
    };
    let mut arguments = Vec::with_capacity(argument_count);
    for &raw_type in raw_arguments {
        let Some(ty) = c_scalar_type(raw_type) else { return 0; };
        arguments.push(ty);
    }
    match state.vm.register_external(
        name.to_owned(),
        HostSignature { arguments, result },
        ExternalFunction::C(function)
    ) {
        Ok(()) => 1,
        Err(_) => 0
    }
}

/// # Safety
/// `state` must be valid and `source` must be a NUL-terminated UTF-8 L0 unit.
#[no_mangle]
pub unsafe extern "C" fn l0_execute(state: *mut L0State, source: *const c_char) -> c_int {
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
        Ok(Err(error)) => {
            state_ref.last_error = CString::new(error.to_string()).ok();
            0
        }
        Err(_) => {
            state_ref.is_executing = false;
            state_ref.vm.callback_state = None;
            state_ref.last_error = CString::new("panic prevented from crossing the C ABI boundary").ok();
            0
        }
    }
}
