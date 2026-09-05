use l0::{Error, Heap, HeapObject, Type, Value, Vm};
use md5::{Md5, Digest};
use std::cell::RefCell;

/// Нативная Rust-функция, которая будет доступна внутри скриптов L0.
/// Сигнатура всегда одинаковая: принимает слайс аргументов `Value` и ссылку на кучу `Heap`.
fn l0_md5(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // 1. Проверяем аргументы. Ожидаем ровно одну строку.
    let [Value::String(ref_id)] = arguments else {
        return Err(Error::Runtime("md5() expects exactly one string argument".into()));
    };

    // 2. Читаем строку из кучи VM.
    // Мы ограничиваем область видимости `borrow()`, чтобы не держать кучу заблокированной,
    // пока считаем хеш и форматируем результат.
    let hash_hex = {
        let h = heap.borrow();
        let text = match h.get(*ref_id)? {
            HeapObject::String(s) => s.as_str(),
            _ => return Err(Error::Runtime("invalid string heap object".into())),
        };
        
        // Считаем MD5 хеш используя крейт `md-5`
        let mut hasher = Md5::new();
        hasher.update(text.as_bytes());
        let result = hasher.finalize();
        
        // Превращаем хеш в hex-строку
        format!("{:x}", result)
    };
    // 3. Выделяем место в куче VM для новой строки с результатом
    let mut h_mut = heap.borrow_mut();
    let new_ref = h_mut.allocate(HeapObject::String(hash_hex));
    // 4. Возвращаем указатель на новую строку
    Ok(Value::String(new_ref))
}

fn main() {
    // Создаем экземпляр виртуальной машины L0
    let mut vm = Vm::default();
    
    // Включаем интерактивный режим, чтобы `printf` и `print` сразу выводили текст в консоль
    vm.set_interactive(true);

    // Регистрируем нашу нативную функцию в VM.
    // Указываем: Имя в L0, Типы аргументов, Тип возвращаемого значения, Сама функция.
    vm.register_rust_function(
        "md5", 
        vec![Type::String], 
        Type::String, 
        l0_md5
    ).expect("Failed to register native function 'md5'");

    // Пишем скрипт на языке L0, который использует нашу функцию
    let script = r#"
        printf("=== MD5 Hash Test ===\n")
        
        let text1: string = "hello world"
        let hash1: string = md5(text1)
        printf("MD5('{}') = {}\n", text1, hash1)
        
        let text2: string = "Rust + L0 = <3"
        let hash2: string = md5(text2)
        printf("MD5('{}') = {}\n", text2, hash2)
        
        printf("=====================\n")
    "#;
    println!("Running L0 script from Rust host...\n");

    // Исполняем скрипт
    if let Err(error) = vm.execute(script) {
        eprintln!("Execution error: {}", error);
    }
}
