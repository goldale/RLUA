use std::rc::Rc;
use std::cell::RefCell;
use std::{env, fs, process};
use ndarray::{ArrayView1, ArrayView2, Axis};
use l0::{Error, Heap, HeapObject, Type, Value, Vm};
use ndarray_linalg::SVD;

use std::sync::OnceLock;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

static CUDA_DEV: OnceLock<std::sync::Arc<CudaDevice>> = OnceLock::new();

// Настоящий код ядра на CUDA C++
const CUDA_KERNELS: &str = r#"
extern "C" __global__ void kernel_sub(const float* a, const float* b, float* out, int num_elements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = a[idx] - b[idx];
    }
}
"#;
fn get_cuda_device() -> std::sync::Arc<CudaDevice> {
    CUDA_DEV.get_or_init(|| {
        let dev = CudaDevice::new(0).expect("Не удалось инициализировать CUDA GPU");
        // Используем встроенное поле `arch` вместо вектора строк
        let opts = CompileOptions {arch: Some("sm_61"), ..Default::default()};
        let ptx = compile_ptx_with_opts(CUDA_KERNELS, opts)
            .expect("Не удалось скомпилировать PTX ядро");
        dev.load_ptx(ptx, "my_kernels", &["kernel_sub"]).expect("Не удалось загрузить ядро");
        dev
    }).clone()
}
// Отправка CPU-тензора в GPU
fn to_cuda(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_id, _)] = arguments else {
        return Err(Error::Runtime("to_cuda expects 1 dTensor".into()));
    };

    let dev = get_cuda_device();

    // Блочная область видимости для чтения из кучи
    let (gpu_data, element_cloned, shape_cloned) = {
        let h = heap.borrow();
        let (bytes, element, shape) = match h.get(*ref_id)? {
            HeapObject::Tensor { bytes, element, shape } => (bytes, element, shape),
            _ => return Err(Error::Runtime("invalid tensor".into())),
        };
        let gpu_data: CudaSlice<u8> = dev.htod_sync_copy(bytes)
            .map_err(|e| Error::Runtime(format!("CUDA HTOD: {}", e)))?;

        // Клонируем нужные данные ДО того, как `h` будет уничтожен
        (gpu_data, element.clone(), shape.clone())
    }; // <- Здесь Rust автоматически делает drop(h)

    let mut h_mut = heap.borrow_mut();
    let new_ref = h_mut.allocate(HeapObject::CudaTensor {
        data: gpu_data,
        element: element_cloned,
        shape: shape_cloned,
    });
    Ok(Value::CudaTensor(new_ref, std::rc::Rc::new(Type::DCudaTensor)))
}

// Возврат GPU-тензора обратно в CPU
fn to_cpu(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::CudaTensor(ref_id, _)] = arguments else {
        return Err(Error::Runtime("to_cpu expects 1 dCudaTensor".into()));
    };

    let dev = get_cuda_device();

    let (cpu_bytes, element_cloned, shape_cloned) = {
        let h = heap.borrow();
        let (data, element, shape) = match h.get(*ref_id)? {
            HeapObject::CudaTensor { data, element, shape } => (data, element, shape),
            _ => return Err(Error::Runtime("invalid CudaTensor".into())),
        };

        let cpu_bytes = dev.dtoh_sync_copy(data)
            .map_err(|e| Error::Runtime(format!("CUDA DTOH: {}", e)))?;

        (cpu_bytes, element.clone(), shape.clone())
    }; // <- drop(h)
    let mut h_mut = heap.borrow_mut();
    let new_ref = h_mut.allocate(HeapObject::Tensor {
        bytes: cpu_bytes,
        element: element_cloned,
        shape: shape_cloned,
    });
    Ok(Value::Tensor(new_ref, std::rc::Rc::new(Type::DTensor)))
}

// Запуск CUDA ядра
fn cuda_go(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::CudaTensor(ref_a, _), Value::CudaTensor(ref_b, _)] = arguments else {
        return Err(Error::Runtime("cuda_go expects 2 dCudaTensors".into()));
    };

    let dev = get_cuda_device();

    // Переменные, которые мы заполним внутри блока и передадим в новый объект
    let mut out_data: CudaSlice<u8>;
    let shape_cloned;

    {
        let h = heap.borrow();

        let (data_a, shape_a) = match h.get(*ref_a)? {
            HeapObject::CudaTensor { data, shape, element } if *element == Type::F32 => (data, shape),
            _ => return Err(Error::Runtime("Invalid cuda tensor A".into())),
        };

        let (data_b, _) = match h.get(*ref_b)? {
            HeapObject::CudaTensor { data, .. } => (data, ()),
            _ => return Err(Error::Runtime("Invalid cuda tensor B".into())),
        };

        let num_elements = shape_a.iter().product::<usize>();
        shape_cloned = shape_a.clone();

        // Выделяем память (x4, так как каждый float занимает 4 байта)
        out_data = dev.alloc_zeros::<u8>(num_elements * 4)
            .map_err(|e| Error::Runtime(e.to_string()))?;

        let f = dev.get_func("my_kernels", "kernel_sub").unwrap();
        let cfg = LaunchConfig::for_num_elems(num_elements as u32);

        // cudarc позволяет напрямую передавать CudaSlice<u8> как указатели на память.
        // Видеокарта сама прочитает их как `float*`, потому что мы так написали в ядре PTX.
        unsafe { f.launch(cfg, (data_a, data_b, &mut out_data, num_elements as i32)) }
            .map_err(|e| Error::Runtime(format!("CUDA launch failed: {}", e)))?;

    } // <- drop(h)
    let mut h_mut = heap.borrow_mut();
    let new_ref = h_mut.allocate(HeapObject::CudaTensor {
        data: out_data,
        element: Type::F32,
        shape: shape_cloned,
    });

    Ok(Value::CudaTensor(new_ref, std::rc::Rc::new(Type::DCudaTensor)))
}

// Help функция для извлечения массива f32 из dTensor L0
fn get_tensor_f32(heap: &Heap, reference: l0::HeapRef) -> Result<(Vec<f32>, Vec<usize>), Error> {
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

fn svdvals(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_id, _)] = arguments else {
        return Err(Error::Runtime("svdvals expects 1 dTensor".into()));
    };
    let (floats, shape) = get_tensor_f32(&heap.borrow(), *ref_id)?;
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
    // Explicitly return Type::DTensor to satisfy the FFI boundary signature
    let tensor_type = Rc::new(Type::DTensor);
    Ok(Value::Tensor(new_ref, tensor_type))
}

// 2. Среднее - mean_dim0(a: dTensor) -> dTensor
fn mean_dim0(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_id, _)] = arguments else {
        return Err(Error::Runtime("mean_dim0 expects 1 dTensor".into()));
    };

    let (floats, shape) = get_tensor_f32(&heap.borrow(), *ref_id)?;
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
    let tensor_type = Rc::new(Type::DTensor);
    Ok(Value::Tensor(new_ref, tensor_type))
}

// 3. Вычитание тензора (броадкастинг 1D из 2D): t_sub(a: dTensor, b: dTensor) -> dTensor
fn t_sub(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_a, ty), Value::Tensor(ref_b, _)] = arguments else {
        return Err(Error::Runtime("t_sub expects 2 dTensor arguments".into()));
    };
    let (floats_a, shape_a) = get_tensor_f32(&heap.borrow(), *ref_a)?;
    let (floats_b, shape_b) = get_tensor_f32(&heap.borrow(), *ref_b)?;

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

// 2. Поэлементное возведение в квадрат: t_square(a: dTensor) -> dTensor
fn t_square(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_id, ty)] = arguments else {
        return Err(Error::Runtime("t_square expects 1 dTensor".into()));
    };
    let (floats, shape) = get_tensor_f32(&heap.borrow(), *ref_id)?;
    let mut bytes       = Vec::with_capacity(floats.len() * 4);
    for val in floats {
        let squared = val * val;
        bytes.extend_from_slice(&squared.to_bits().to_le_bytes());
    }
    let mut h = heap.borrow_mut();
    let new_ref = h.allocate(HeapObject::Tensor { bytes, element: Type::F32, shape });
    Ok(Value::Tensor(new_ref, ty.clone()))
}

// 3. Сумма элементов тензора: t_sum(a: dTensor) -> f32
fn t_sum(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_id, _)] = arguments else {
        return Err(Error::Runtime("t_sum expects 1 dTensor".into()));
    };
    let (floats, _) = get_tensor_f32(&heap.borrow(), *ref_id)?;
    let sum: f32 = floats.iter().sum();

    Ok(Value::F32(sum))
}

fn t_reshape(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_id, _ty), Value::I32(dim1), Value::I32(dim2)] = arguments else {
        return Err(Error::Runtime("t_reshape expects 1 dTensor and 2 i32 arguments".into()));
    };
    // Extract existing tensor data from the L0 heap
    let (bytes, element, old_shape) = {
        let h = heap.borrow();
        match h.get(*ref_id)? {
            HeapObject::Tensor { bytes, element, shape } => (bytes.clone(), element.clone(), shape.clone()),
            _ => return Err(Error::Runtime("invalid tensor object".into())),
        }
    };
    // Calculate total elements to properly resolve the -1 wildcard
    let total_elements: isize = old_shape.iter().product::<usize>() as isize;
    let mut d1 = *dim1 as isize;
    let mut d2 = *dim2 as isize;
    if d1 == -1 {
        d1 = total_elements / d2;
    } else if d2 == -1 {
        d2 = total_elements / d1;
    }
    let new_shape = vec![d1 as usize, d2 as usize];
    // Allocate a new tensor in the L0 heap with the exact same bytes but a new shape
    let mut h = heap.borrow_mut();
    let new_ref = h.allocate(HeapObject::Tensor {
        bytes,
        element: element.clone(),
        shape: new_shape.clone()
    });
    let tensor_type = Rc::new(Type::DTensor);
    Ok(Value::Tensor(new_ref, tensor_type))
}

fn t_scalar_add(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // 1. Accept a dTensor and an f32 scalar
    let [Value::Tensor(ref_id, ty), Value::F32(scalar)] = arguments else {
        return Err(Error::Runtime("t_scalar_add expects 1 dTensor and 1 f32 argument".into()));
    };
    // 2. Extract the raw bytes and shape from the tensor in the heap
    let (floats, shape) = {
        let h = heap.borrow();
        match h.get(*ref_id)? {
            HeapObject::Tensor { bytes, element, shape } if *element == Type::F32 => {
                let mut temp = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    temp.push(f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())));
                }
                (temp, shape.clone())
            }
            _ => return Err(Error::Runtime("invalid tensor heap object or element type".into())),
        }
    };
    // 3. Add the scalar to each element and pack back into bytes
    let mut new_bytes = Vec::with_capacity(floats.len() * 4);
    for val in floats {
        let add_val = val + scalar;
        new_bytes.extend_from_slice(&add_val.to_bits().to_le_bytes());
    }
    // 4. Allocate a new tensor in the VM heap
    let mut h = heap.borrow_mut();
    let new_ref = h.allocate(HeapObject::Tensor {
        bytes: new_bytes,
        element: Type::F32,
        shape
    });
    // 5. Return the new dynamic tensor
    Ok(Value::Tensor(new_ref, ty.clone()))
}

fn t_scalar_div(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // 1. Accept a dTensor and an f32 scalar
    let [Value::Tensor(ref_id, ty), Value::F32(scalar)] = arguments else {
        return Err(Error::Runtime("t_scalar_div expects 1 dTensor and 1 f32 argument".into()));
    };
    // 2. Extract the raw bytes and shape from the tensor
    let (floats, shape) = {
        let h = heap.borrow();
        match h.get(*ref_id)? {
            HeapObject::Tensor { bytes, element, shape } if *element == Type::F32 => {
                let mut temp = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    temp.push(f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())));
                }
                (temp, shape.clone())
            }
            _ => return Err(Error::Runtime("invalid tensor heap object or element type".into())),
        }
    };
    // 3. Divide each element by the scalar
    let mut new_bytes = Vec::with_capacity(floats.len() * 4);
    for val in floats {
        let div_val = val / scalar;
        new_bytes.extend_from_slice(&div_val.to_bits().to_le_bytes());
    }
    // 4. Allocate a new tensor in the VM heap
    let mut h = heap.borrow_mut();
    let new_ref = h.allocate(HeapObject::Tensor {
        bytes: new_bytes,
        element: Type::F32,
        shape
    });
    // 5. Return the new tensor, preserving the original dynamic tensor type
    Ok(Value::Tensor(new_ref, ty.clone()))
}

fn t_log(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // 1. Accept a single dTensor argument
    let [Value::Tensor(ref_id, ty)] = arguments else {
        return Err(Error::Runtime("t_log expects exactly 1 dTensor argument".into()));
    };

    // 2. Extract the raw bytes and shape from the tensor in the heap
    let (floats, shape) = {
        let h = heap.borrow();
        match h.get(*ref_id)? {
            HeapObject::Tensor { bytes, element, shape } if *element == Type::F32 => {
                let mut temp = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    temp.push(f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())));
                }
                (temp, shape.clone())
            }
            _ => return Err(Error::Runtime("invalid tensor heap object or element type".into())),
        }
    };

    // 3. Apply the natural logarithm (.ln()) to each element and pack back into bytes
    let mut new_bytes = Vec::with_capacity(floats.len() * 4);
    for val in floats {
        let log_val = val.ln();
        new_bytes.extend_from_slice(&log_val.to_bits().to_le_bytes());
    }

    // 4. Allocate a new tensor in the VM heap
    let mut h = heap.borrow_mut();
    let new_ref = h.allocate(HeapObject::Tensor {
        bytes: new_bytes,
        element: Type::F32,
        shape
    });
    // 5. Return the new dynamic tensor
    Ok(Value::Tensor(new_ref, ty.clone()))
}

fn t_mul(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // 1. Accept two dTensor arguments
    let [Value::Tensor(ref_a, ty_a), Value::Tensor(ref_b, _)] = arguments else {
        return Err(Error::Runtime("t_mul expects exactly 2 dTensor arguments".into()));
    };

    // 2. Extract the raw bytes and shape from the first tensor
    let (floats_a, shape_a) = {
        let h = heap.borrow();
        match h.get(*ref_a)? {
            HeapObject::Tensor { bytes, element, shape } if *element == Type::F32 => {
                let mut temp = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    temp.push(f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())));
                }
                (temp, shape.clone())
            }
            _ => return Err(Error::Runtime("invalid first tensor heap object or element type".into())),
        }
    };

    // 3. Extract the raw bytes from the second tensor
    let floats_b = {
        let h = heap.borrow();
        match h.get(*ref_b)? {
            HeapObject::Tensor { bytes, element, .. } if *element == Type::F32 => {
                let mut temp = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    temp.push(f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())));
                }
                temp
            }
            _ => return Err(Error::Runtime("invalid second tensor heap object or element type".into())),
        }
    };
    if floats_a.len() != floats_b.len() {
        return Err(Error::Runtime("t_mul expects tensors of the same size for element-wise multiplication".into()));
    }
    // 4. Perform element-wise multiplication and pack back into bytes
    let mut new_bytes = Vec::with_capacity(floats_a.len() * 4);
    for (a, b) in floats_a.iter().zip(floats_b.iter()) {
        let mul_val = a * b;
        new_bytes.extend_from_slice(&mul_val.to_bits().to_le_bytes());
    }
    // 5. Allocate a new tensor in the VM heap
    let mut h = heap.borrow_mut();
    let new_ref = h.allocate(HeapObject::Tensor {
        bytes: new_bytes,
        element: Type::F32,
        shape: shape_a
    });
    // 6. Return the new dynamic tensor
    Ok(Value::Tensor(new_ref, ty_a.clone()))
}

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

    vm.register_rust_function("to_cuda", vec![Type::DTensor], Type::DCudaTensor, to_cuda).unwrap();
    vm.register_rust_function("to_cpu", vec![Type::DCudaTensor], Type::DTensor, to_cpu).unwrap();
    vm.register_rust_function("cuda_go", vec![Type::DCudaTensor, Type::DCudaTensor], Type::DCudaTensor,
                              cuda_go).unwrap();

    // Регистрация математических оберток для L0
    vm.register_rust_function("svdvals", vec![Type::DTensor], Type::DTensor, svdvals).unwrap();
    vm.register_rust_function("t_square", vec![Type::DTensor], Type::DTensor, t_square).unwrap();
    vm.register_rust_function("t_sum", vec![Type::DTensor], Type::F32, t_sum).unwrap();
    vm.register_rust_function("mean_dim0", vec![Type::DTensor], Type::DTensor, mean_dim0).unwrap();
    vm.register_rust_function("t_sub", vec![Type::DTensor, Type::DTensor], Type::DTensor, t_sub).unwrap();
    vm.register_rust_function("t_reshape", vec![Type::DTensor, Type::I32, Type::I32],
                              Type::DTensor, t_reshape).unwrap();
    vm.register_rust_function("t_scalar_div", vec![Type::DTensor, Type::F32], Type::DTensor,
                              t_scalar_div).unwrap();
    vm.register_rust_function("t_scalar_add", vec![Type::DTensor, Type::F32],
        Type::DTensor, t_scalar_add).unwrap();
    vm.register_rust_function("t_log", vec![Type::DTensor],
        Type::DTensor, t_log).unwrap();
    vm.register_rust_function("t_mul", vec![Type::DTensor, Type::DTensor],
        Type::DTensor, t_mul).unwrap();
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
