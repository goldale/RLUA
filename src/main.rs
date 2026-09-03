use l0::{Error, Heap, HeapObject, Type, Value, Vm};
use std::cell::RefCell;
use std::{env, fs, process};

// Ваша функция-расширение
fn int2str(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I64(val)] = arguments else {
        return Err(Error::Runtime("int2str expects exactly one i64 argument".into()));
    };

    let str_val = val.to_string();
    let mut h = heap.borrow_mut();
    let ref_id = h.allocate(HeapObject::String(str_val));

    Ok(Value::String(ref_id))
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: l0 <file.l0>"); process::exit(2);
    });
    if let Err(e) = fs::metadata(&path) { eprintln!("{path}: {e}"); process::exit(2); }

    // Создаем экземпляр виртуальной машины
    let mut vm = Vm::default();
    vm.set_interactive(true);

    // Регистрируем функцию int2str
    if let Err(e) = vm.register_rust_function("int2str", vec![Type::I64], Type::String, int2str) {
        eprintln!("Failed to register host functions: {}", e);
        process::exit(1);
    }
    // В интерактивном режиме VM выводит текст сразу, до возможного `input`.
    match vm.execute_file(&path) {
        Ok(_) => {},
        Err(error) => { eprintln!("{path}: {error}"); process::exit(1); }
    }
}
