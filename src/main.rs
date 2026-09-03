use l0::{Error, Heap, HeapObject, Type, Value, Vm};
use std::cell::RefCell;
use std::{env, fs, process};
use std::rc::Rc;
use ndarray::{ArrayView1, ArrayView2, Axis};
use ndarray_linalg::SVD;

// Вспомогательная функция для извлечения массива f32 из dTensor L0
fn extract_tensor_f32(heap: &Heap, reference: l0::HeapRef) -> Result<(Vec<f32>, Vec<usize>), Error> {
    match heap.get(reference)? {
        HeapObject::Tensor { bytes, element, shape } if *element == Type::F32 => {
            let mut floats = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let val = f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap()));
                floats.push(val);
            }
            Ok((floats, shape.clone()))
        }
        _ => Err(Error::Type("expected tensor<f32>".into())),
    }
}

// 1. Настоящее SVD: torch_svdvals(h_centered: dTensor) -> dTensor
fn torch_svdvals(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_id, _)] = arguments else {
        return Err(Error::Runtime("torch_svdvals expects 1 dTensor".into()));
    };

    let (floats, shape) = extract_tensor_f32(&heap.borrow(), *ref_id)?;
    if shape.len() != 2 {
        return Err(Error::Runtime("SVD requires a 2D tensor".into()));
    }

    // Создаем матрицу ndarray из извлеченных данных
    let matrix = ArrayView2::from_shape((shape[0], shape[1]), &floats)
        .map_err(|_| Error::Runtime("Failed to create matrix view".into()))?;

    // Вычисляем сингулярные числа (без вычисления матриц U и VT для ускорения)
    let (_, s, _) = matrix.svd(false, false)
        .map_err(|_| Error::Runtime("SVD computation failed".into()))?;
    
    let svd_result: Vec<f32> = s.to_vec();
    let new_shape = vec![svd_result.len()];

    let mut bytes = Vec::with_capacity(svd_result.len() * 4);
    for val in svd_result {
        bytes.extend_from_slice(&val.to_bits().to_le_bytes());
    }

    let mut h = heap.borrow_mut();
    let new_ref = h.allocate(HeapObject::Tensor { 
        bytes, element: Type::F32, shape: new_shape.clone() 
    });

    let tensor_type = Rc::new(Type::Tensor(Box::new(Type::F32), new_shape.len()));
    Ok(Value::Tensor(new_ref, tensor_type))
}

// 2. Среднее по нулевому измерению: torch_mean_dim0(a: dTensor) -> dTensor
fn torch_mean_dim0(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_id, _)] = arguments else {
        return Err(Error::Runtime("torch_mean_dim0 expects 1 dTensor".into()));
    };

    let (floats, shape) = extract_tensor_f32(&heap.borrow(), *ref_id)?;
    let matrix = ArrayView2::from_shape((shape[0], shape[1]), &floats).unwrap();
    
    // Считаем среднее по строкам (Axis 0)
    let mean_array = matrix.mean_axis(Axis(0)).unwrap();
    
    let mut bytes = Vec::with_capacity(mean_array.len() * 4);
    for val in mean_array.iter() {
        bytes.extend_from_slice(&val.to_bits().to_le_bytes());
    }

    let mut h = heap.borrow_mut();
    let new_ref = h.allocate(HeapObject::Tensor { 
        bytes, element: Type::F32, shape: vec![shape[1]] 
    });
    
    let tensor_type = Rc::new(Type::Tensor(Box::new(Type::F32), 1));
    Ok(Value::Tensor(new_ref, tensor_type))
}

// 3. Вычитание тензора (броадкастинг 1D из 2D): torch_sub(a: dTensor, b: dTensor) -> dTensor
fn torch_sub(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_a, ty), Value::Tensor(ref_b, _)] = arguments else {
        return Err(Error::Runtime("torch_sub expects 2 dTensor arguments".into()));
    };

    let (floats_a, shape_a) = extract_tensor_f32(&heap.borrow(), *ref_a)?;
    let (floats_b, shape_b) = extract_tensor_f32(&heap.borrow(), *ref_b)?;

    let matrix_a = ArrayView2::from_shape((shape_a[0], shape_a[1]), &floats_a).unwrap();
    let vec_b = ArrayView1::from_shape(shape_b[0], &floats_b).unwrap();

    // Броадкастинг и вычитание
    let result = &matrix_a - &vec_b;

    let mut bytes = Vec::with_capacity(result.len() * 4);
    for val in result.iter() {
        bytes.extend_from_slice(&val.to_bits().to_le_bytes());
    }

    let mut h = heap.borrow_mut();
    let new_ref = h.allocate(HeapObject::Tensor { bytes, element: Type::F32, shape: shape_a });
    Ok(Value::Tensor(new_ref, ty.clone()))
}

// 2. Поэлементное возведение в квадрат: torch_square(a: dTensor) -> dTensor
fn torch_square(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_id, ty)] = arguments else {
        return Err(Error::Runtime("torch_square expects 1 dTensor".into()));
    };

    let (floats, shape) = extract_tensor_f32(&heap.borrow(), *ref_id)?;
    let mut bytes       = Vec::with_capacity(floats.len() * 4);
    for val in floats {
        let squared = val * val;
        bytes.extend_from_slice(&squared.to_bits().to_le_bytes());
    }

    let mut h = heap.borrow_mut();
    let new_ref = h.allocate(HeapObject::Tensor { bytes, element: Type::F32, shape });
    Ok(Value::Tensor(new_ref, ty.clone()))
}

// 3. Сумма элементов тензора: torch_sum(a: dTensor) -> f32
fn torch_sum(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_id, _)] = arguments else {
        return Err(Error::Runtime("torch_sum expects 1 dTensor".into()));
    };

    let (floats, _) = extract_tensor_f32(&heap.borrow(), *ref_id)?;
    let sum: f32 = floats.iter().sum();
    
    Ok(Value::F32(sum))
}

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

    // Регистрация математических оберток для L0
    vm.register_rust_function("torch_svdvals", vec![Type::DTensor], Type::DTensor, torch_svdvals).unwrap();
    vm.register_rust_function("torch_square", vec![Type::DTensor], Type::DTensor, torch_square).unwrap();
    vm.register_rust_function("torch_sum", vec![Type::DTensor], Type::F32, torch_sum).unwrap();
    vm.register_rust_function("torch_mean_dim0", vec![Type::DTensor], Type::DTensor, torch_mean_dim0).unwrap();
    vm.register_rust_function("torch_sub", vec![Type::DTensor, Type::DTensor], Type::DTensor, torch_sub).unwrap();
    // ... регистрация остальных функций моста ...

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
