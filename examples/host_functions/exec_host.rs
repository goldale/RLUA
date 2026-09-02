// cargo run --example exec_host

use l0::{Error, Heap, HeapObject, Type, Value, Vm};
use std::cell::RefCell;
use std::process::Command;

/// Host-функция для выполнения бинарных файлов.
/// Принимает строку (команду с аргументами), разделяет ее по пробелам,
/// выполняет через std::process::Command и возвращает stdout в виде строки L0.
fn host_exec(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // 1. Проверка количества аргументов
    let [arg] = arguments else {
        return Err(Error::Runtime("exec expects exactly 1 argument".into()));
    };

    // 2. Чтение аргумента-строки из кучи L0[cite: 1]
    let cmd_str = {
        let h = heap.borrow();
        match arg {
            Value::String(reference) => match h.get(*reference)? {
                HeapObject::String(text) => text.clone(),
                _ => return Err(Error::Runtime("invalid string heap object".into())),
            },
            _ => return Err(Error::Type("exec expects a string".into())),
        }
    };

    // 3. Парсинг команды и аргументов
    let mut parts = cmd_str.split_whitespace();
    let program = parts.next().unwrap_or("");
    if program.is_empty() {
        return Err(Error::Runtime("empty command string provided".into()));
    }
    let args: Vec<&str> = parts.collect();

    // 4. Системный вызов (автоматический поиск в PATH)
    let output = Command::new(program)
        .args(&args)
        .output()
        .map_err(|e| Error::Runtime(format!("failed to execute process '{}': {}", program, e)))?;

    // Приведение вывода к строке UTF-8 (ошибки декодирования заменяются на U+FFFD)
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    // 5. Размещение полученной строки обратно в куче L0[cite: 1]
    let mut h = heap.borrow_mut();
    let ref_id = h.allocate(HeapObject::String(stdout));
    
    Ok(Value::String(ref_id))
}

fn main() -> Result<(), Error> {
    let mut vm = Vm::default(); //[cite: 1]

    // 6. Регистрация функции в VM
    // Функция регистрируется с именем "exec", принимает одну строку и возвращает строку[cite: 1]
    vm.register_rust_function(
        "exec",
        vec![Type::String],
        Type::String,
        host_exec,
    )?;

    // 7. Скрипт на языке L0 для проверки
    let script = r#"
        -- Вызов системных утилит (зависит от ОС: пример для Linux/macOS, на Windows используйте 'cmd.exe /C echo ...')
        let result1: string = exec("echo Hello from external binary!")
        print(result1)

        let result2: string = exec("rustc --version")
        print(result2)
    "#;

    // Выполнение скрипта и перехват вывода[cite: 1]
    for line in vm.execute(script)? {
        println!("{}", line);
    }

    Ok(())
}
