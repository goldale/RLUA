use l0::{Error, Type, Value, Vm, HeapObject};
use std::{env, fs, process};
use std::cell::RefCell;

// Базовая встроенная функция int2str остается для примера
fn int2str(arguments: &[Value], heap: &RefCell<l0::Heap>) -> Result<Value, Error> {
    let [Value::I64(val)] = arguments else {
        return Err(Error::Runtime("int2str expects exactly one i64 argument".into()));
    };
    let str_val = val.to_string();
    let mut h = heap.borrow_mut();
    let ref_id = h.allocate(HeapObject::String(str_val));
    Ok(Value::String(ref_id))
}
// Читает аргументы командной строки по индексу для скриптов L0
fn get_cli_arg(arguments: &[Value], heap: &RefCell<l0::Heap>) -> Result<Value, l0::Error> {
    let [Value::I32(idx)] = arguments else {
        return Err(l0::Error::Runtime("get_arg expects 1 i32 argument".into()));
    };

    // В Rust args() содержит: 0 = бинарник l0, 1 = путь к скрипту, 2 = первый параметр скрипта
    let arg_str = std::env::args().nth(*idx as usize).unwrap_or_else(|| "".to_string());

    let mut h = heap.borrow_mut();
    let ref_id = h.allocate(l0::HeapObject::String(arg_str));
    Ok(Value::String(ref_id))
}
fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: l0 <file.l0>");
        process::exit(2);
    });
    if let Err(e) = fs::metadata(&path) {
        eprintln!("{path}: {e}");
        process::exit(2);
    }
    let mut vm = Vm::default();
    vm.set_interactive(true);
    // Регистрируем все нативные модули (включая Candle) до компиляции скрипта
    vm.register_rust_function("get_arg", vec![Type::I32], Type::String, get_cli_arg).unwrap();
    for ext in l0::ext::available_extensions() {
        if let Err(e) = ext.register(&mut vm) {
            eprintln!("Failed to register extension '{}': {}", ext.name(), e);
            process::exit(1);
        }
    }
    if let Err(e) = vm.register_rust_function("int2str", vec![Type::I64], Type::String, int2str) {
        eprintln!("Failed to register built-in functions: {}", e);
        process::exit(1);
    }
    if let Err(error) = vm.execute_file(&path) {
        eprintln!("{path}: {error}");
        process::exit(1);
    }
}
