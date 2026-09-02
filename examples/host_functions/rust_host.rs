use l0::{Error, Type, Value, Vm};

fn rust_multiply(arguments: &[Value]) -> Result<Value, Error> {
    let [Value::I32(left), Value::I32(right)] = arguments else {
        return Err(Error::Runtime("rust_multiply expects two i32 values".into()));
    };
    Ok(Value::I32(left * right))
}

fn main() -> Result<(), Error> {
    let mut vm = Vm::default();
    vm.register_rust_function(
        "rust_multiply",
        vec![Type::I32, Type::I32],
        Type::I32,
        rust_multiply,
    )?;

    for line in vm.execute(include_str!("rust_host.l0"))? {
        println!("{line}");
    }
    Ok(())
}
