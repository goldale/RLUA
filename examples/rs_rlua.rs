use l0::{execute, Error};

fn main() -> Result<(), Error> {
    // L0-скрипт, который будет выполнен внутри Rust-хоста
    let script = "
        let a: i32 = 10;
        let b: i32 = 15;
        print(a + b);
    ";

    // Компиляция и запуск скрипта с получением массива выведенных строк[cite: 1]
    let output = execute(script)?; //[cite: 1]

    // Чтение строк, которые скрипт передал в функцию print() внутри VM
    for line in output {
        println!("Вывод L0-скрипта: {}", line);
    }

    Ok(())
}
