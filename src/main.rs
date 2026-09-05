use half::f16;
use std::rc::Rc;
use std::cell::RefCell;
use std::{env, fs, process};
use ndarray::{ArrayView1, ArrayView2, Axis};
use l0::{Error, Heap, HeapObject, Type, Value, Vm};
use ndarray_linalg::SVD;

use std::sync::OnceLock;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
//use cudarc::cusolver::{Cusolver, SafeCusolver};
//use cudarc::nvrtc::compile_ptx;

// === CANDLE ===
use candle_core::{Tensor, Device, DType};

////////// candle_core ////////////////////
use candle_core::{Tensor, Device};

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

fn int2str(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I64(val)] = arguments else {
        return Err(Error::Runtime("int2str expects exactly one i64 argument".into()));
    };

    let str_val = val.to_string();
    let mut h = heap.borrow_mut();
    let ref_id = h.allocate(HeapObject::String(str_val));

    Ok(Value::String(ref_id))
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
// Извлекает массив f16 и форму (shape) из dTensor L0
fn get_tensor_f16(heap: &Heap, reference: l0::HeapRef) -> Result<(Vec<f16>, Vec<usize>), Error> {
    match heap.get(reference)? {
        HeapObject::Tensor { bytes, element, shape } if *element == Type::F16 => {
            let mut floats = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                floats.push(f16::from_bits(bits));
            }
            Ok((floats, shape.clone()))
        }
        _ => Err(Error::Type("expected tensor<f16>".into())),
    }
}
/// Извлекает сырые данные из vector<f32> (dArray)
fn get_array_f32(heap: &Heap, reference: l0::HeapRef) -> Result<Vec<f32>, Error> {
    match heap.get(reference)? {
        HeapObject::Array { bytes, element } if *element == Type::F32 => {
            let mut floats = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let bits = u32::from_le_bytes(chunk.try_into().unwrap());
                floats.push(f32::from_bits(bits));
            }
            Ok(floats)
        }
        _ => Err(Error::Type("expected vector<f32> or dArray".into())),
    }
}
/// Извлекает сырые данные из vector<f16> (dArray)
fn get_array_f16(heap: &Heap, reference: l0::HeapRef) -> Result<Vec<f16>, Error> {
    match heap.get(reference)? {
        HeapObject::Array { bytes, element } if *element == Type::F16 => {
            let mut floats = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                floats.push(f16::from_bits(bits));
            }
            Ok(floats)
        }
        _ => Err(Error::Type("expected vector<f16> or dArray".into())),
    }
}
///////////////////////////////////////////////////////////

fn get_cuda_device() -> std::sync::Arc<CudaDevice> {
    CUDA_DEV.get_or_init(|| {
        let dev = CudaDevice::new(0).expect("Не удалось инициализировать CUDA GPU");
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

// Cuda substraction
fn cuda_sub(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::CudaTensor(ref_a, _), Value::CudaTensor(ref_b, _)] = arguments else {
        return Err(Error::Runtime("cuda_sub expects 2 dCudaTensors".into()));
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

// Возвращает dCudaTensor (сингулярные числа), принимая dCudaTensor
/*
fn cuda_svdvals(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::CudaTensor(ref_id, _)] = arguments else {
        return Err(Error::Runtime("svdvals_cuda expects 1 dCudaTensor".into()));
    };
    let dev = get_cuda_device();
    
    // Инициализируем обертку над cuSOLVER
    let solver = Cusolver::new(dev.clone())
        .map_err(|e| Error::Runtime(format!("cuSOLVER init failed: {}", e)))?;
    let (matrix_data, shape_cloned) = {
        let h = heap.borrow();
        let (data, shape, element) = match h.get(*ref_id)? {
            HeapObject::CudaTensor { data, shape, element } if *element == Type::F32 => (data, shape, element),
            _ => return Err(Error::Runtime("Invalid cuda tensor".into())),
        };
        if shape.len() != 2 {
            return Err(Error::Runtime("SVD requires a 2D tensor".into()));
        }
        // Алгоритм SVD разрушает исходную матрицу в процессе вычислений.
        // Чтобы не испортить тензор пользователя, делаем его копию прямо в VRAM.
        let mut data_clone = dev.alloc_zeros::<u8>(data.len())
            .map_err(|e| Error::Runtime(e.to_string()))?;
        dev.dtod_sync_copy(data, &mut data_clone)
            .map_err(|e| Error::Runtime(e.to_string()))?;
        (data_clone, shape.clone())
    };
    let m = shape_cloned[0];
    let n = shape_cloned[1];
    let min_mn = m.min(n);
    // cudarc позволяет безопасно трансмутировать слайсы байтов в f32 для вычислений
    let mut matrix_f32: CudaSlice<f32> = unsafe { std::mem::transmute(matrix_data) };
    // Выделяем память в VRAM для итогового вектора сингулярных чисел
    let mut s_f32 = dev.alloc_zeros::<f32>(min_mn)
        .map_err(|e| Error::Runtime(e.to_string()))?;
    // Выделяем пустые буферы (мы не вычисляем матрицы U и VT, только сингулярные числа)
    let mut empty_u = dev.alloc_zeros::<f32>(1).unwrap();
    let mut empty_vt = dev.alloc_zeros::<f32>(1).unwrap();
    // Запускаем аппаратный SVD. Передаем (n, m) вместо (m, n) из-за Column-Major
    solver.sgesvd(
        b'N', b'N', // 'N' означает, что мы не сохраняем сингулярные векторы
        n as i32, m as i32, 
        &mut matrix_f32, n as i32, // LDA (Leading Dimension) = n
        &mut s_f32, 
        &mut empty_u, 1, 
        &mut empty_vt, 1
    ).map_err(|e| Error::Runtime(format!("cuSOLVER SVD failed: {}", e)))?;
    // Преобразуем результат обратно в байты для структуры L0
    let s_bytes: CudaSlice<u8> = unsafe { std::mem::transmute(s_f32) };
    // Аллоцируем новый объект CudaTensor в куче виртуальной машины L0[cite: 1]
    let mut h_mut = heap.borrow_mut();
    let new_ref = h_mut.allocate(HeapObject::CudaTensor {
        data: s_bytes,
        element: Type::F32,
        shape: vec![min_mn],
    });
    Ok(Value::CudaTensor(new_ref, std::rc::Rc::new(Type::DCudaTensor)))
} */
/////////// End Cuda /////////////////////////
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
    let [Value::Tensor(ref_a, _), Value::Tensor(ref_b, _)] = arguments else {
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
    Ok(Value::Tensor(new_ref, std::rc::Rc::new(Type::DTensor)))
}
// 2. Поэлементное возведение в квадрат: t_square(a: dTensor) -> dTensor
fn t_square(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_id, _)] = arguments else {
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
    Ok(Value::Tensor(new_ref, std::rc::Rc::new(Type::DTensor)))
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
    let [Value::Tensor(ref_id, _), Value::I32(dim1), Value::I32(dim2)] = arguments else {
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
    let _tensor_type = Rc::new(Type::DTensor);
    Ok(Value::Tensor(new_ref, std::rc::Rc::new(Type::DTensor)))
}

fn t_scalar_add(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // 1. Accept a dTensor and an f32 scalar
    let [Value::Tensor(ref_id, _), Value::F32(scalar)] = arguments else {
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

    Ok(Value::Tensor(new_ref, std::rc::Rc::new(Type::DTensor)))
}

fn t_scalar_div(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // 1. Accept a dTensor and an f32 scalar
    let [Value::Tensor(ref_id, _), Value::F32(scalar)] = arguments else {
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
    Ok(Value::Tensor(new_ref, std::rc::Rc::new(Type::DTensor)))
}

fn t_log(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // 1. Accept a single dTensor argument
    let [Value::Tensor(ref_id, _)] = arguments else {
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
    Ok(Value::Tensor(new_ref, std::rc::Rc::new(Type::DTensor)))
}

fn t_mul(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // 1. Accept two dTensor arguments
    let [Value::Tensor(ref_a, _), Value::Tensor(ref_b, _)] = arguments else {
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
    Ok(Value::Tensor(new_ref, std::rc::Rc::new(Type::DTensor)))
}

// === CANDLE ===

/// Извлекает тензор из L0 и превращает его в Candle Tensor (поддерживает f32 и f16)
fn tensor_to_candle(value: &Value, heap: &Heap, device: &Device) -> Result<Tensor, Error> {
    let Value::Tensor(ref_id, ty) = value else {
        return Err(Error::Runtime("Expected tensor".into()));
    };
    
    let Type::Tensor(element_type, _) = ty.as_ref() else {
        return Err(Error::Runtime("Invalid tensor type".into()));
    };

    match element_type.as_ref() {
        Type::F32 => {
            let (floats, shape) = get_tensor_f32(heap, *ref_id)?;
            Tensor::from_vec(floats, shape.as_slice(), device)
                .map_err(|e| Error::Runtime(format!("Candle f32 error: {}", e)))
        },
        Type::F16 => {
            let (floats, shape) = get_tensor_f16(heap, *ref_id)?;
            Tensor::from_vec(floats, shape.as_slice(), device)
                .map_err(|e| Error::Runtime(format!("Candle f16 error: {}", e)))
        },
        _ => Err(Error::Runtime("Unsupported tensor type. Use f32 or f16.".into()))
    }
}
/// Превращает Candle Tensor обратно в тензор L0 с сохранением размерности и типа
fn candle_to_tensor(tensor: &Tensor, heap: &RefCell<Heap>) -> Result<Value, Error> {
    let cpu_tensor = tensor.to_device(&Device::Cpu)
        .map_err(|e| Error::Runtime(e.to_string()))?;
    
    let shape = tensor.dims().to_vec();
    
    match tensor.dtype() {
        DType::F32 => {
            let floats = cpu_tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut bytes = Vec::with_capacity(floats.len() * 4);
            for val in floats { bytes.extend_from_slice(&val.to_bits().to_le_bytes()); }
            
            let mut h = heap.borrow_mut();
            let new_ref = h.allocate(HeapObject::Tensor { bytes, element: Type::F32, shape });
            Ok(Value::Tensor(new_ref, Rc::new(Type::DTensor)))
        },
        DType::F16 => {
            let floats = cpu_tensor.flatten_all().unwrap().to_vec1::<f16>().unwrap();
            let mut bytes = Vec::with_capacity(floats.len() * 2);
            for val in floats { bytes.extend_from_slice(&val.to_bits().to_le_bytes()); }
            
            let mut h = heap.borrow_mut();
            let new_ref = h.allocate(HeapObject::Tensor { bytes, element: Type::F16, shape });
            Ok(Value::Tensor(new_ref, Rc::new(Type::DTensor)))
        },
        _ => Err(Error::Runtime("Unsupported output dtype from Candle".into()))
    }
}
/// Извлекает вектор из L0 и превращает его в 1D Candle Tensor
fn vector_to_candle(value: &Value, heap: &Heap, device: &Device) -> Result<Tensor, Error> {
    let Value::Array(ref_id, element_type) = value else {
        return Err(Error::Runtime("Expected vector".into()));
    };
    
    match element_type.as_ref() {
        Type::F32 => {
            let floats = get_array_f32(heap, *ref_id)?;
            let shape = [floats.len()];
            Tensor::from_vec(floats, shape.as_slice(), device)
                .map_err(|e| Error::Runtime(format!("Candle f32 error: {}", e)))
        },
        Type::F16 => {
            let floats = get_array_f16(heap, *ref_id)?;
            let shape = [floats.len()];
            Tensor::from_vec(floats, shape.as_slice(), device)
                .map_err(|e| Error::Runtime(format!("Candle f16 error: {}", e)))
        },
        _ => Err(Error::Runtime("Unsupported vector type. Use vector<f32> or vector<f16>.".into()))
    }
}
/// Превращает 1D Candle Tensor обратно в вектор L0
fn candle_to_vector(tensor: &Tensor, heap: &RefCell<Heap>) -> Result<Value, Error> {
    let cpu_tensor = tensor.to_device(&Device::Cpu)
        .map_err(|e| Error::Runtime(e.to_string()))?;
        
    match tensor.dtype() {
        DType::F32 => {
            let floats = cpu_tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut bytes = Vec::with_capacity(floats.len() * 4);
            for val in floats { bytes.extend_from_slice(&val.to_bits().to_le_bytes()); }
            
            let mut h = heap.borrow_mut();
            let new_ref = h.allocate(HeapObject::Array { bytes, element: Type::F32 });
            Ok(Value::Array(new_ref, Box::new(Type::F32)))
        },
        DType::F16 => {
            let floats = cpu_tensor.flatten_all().unwrap().to_vec1::<f16>().unwrap();
            let mut bytes = Vec::with_capacity(floats.len() * 2);
            for val in floats { bytes.extend_from_slice(&val.to_bits().to_le_bytes()); }
            
            let mut h = heap.borrow_mut();
            let new_ref = h.allocate(HeapObject::Array { bytes, element: Type::F16 });
            Ok(Value::Array(new_ref, Box::new(Type::F16)))
        },
        _ => Err(Error::Runtime("Unsupported output dtype from Candle".into()))
    }
}

fn candle_matmul(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let device = Device::new_cuda(0).map_err(|e| Error::Runtime(e.to_string()))?;
    let tensor_a = tensor_to_candle(&arguments[0], &heap.borrow(), &device)?;
    let tensor_b = tensor_to_candle(&arguments[1], &heap.borrow(), &device)?;
    let result = tensor_a.matmul(&tensor_b)
        .map_err(|e| Error::Runtime(format!("Candle matmul failed: {}", e)))?;
    candle_to_tensor(&result, heap)
}

fn candle_l0_layer(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // Ожидаем два аргумента: веса (dTensor) и входной вектор (dArray)
    let [arg_w, arg_x] = arguments else {
        return Err(Error::Runtime("candle_l0_layer expects 1 tensor and 1 vector".into()));
    };
    let device = Device::new_cuda(0).map_err(|e| Error::Runtime(e.to_string()))?;
    // 1. Извлекаем матрицу весов: размерность [out_features, in_features]
    let weights = tensor_to_candle(arg_w, &heap.borrow(), &device)?;
    // 2. Извлекаем входной вектор: размерность [in_features]
    let input_vec = vector_to_candle(arg_x, &heap.borrow(), &device)?;
    // 3. Превращаем вектор [in_features] в матрицу-столбец [in_features, 1] для умножения
    let input_col = input_vec.unsqueeze(1)
        .map_err(|e| Error::Runtime(format!("Candle unsqueeze error: {}", e)))?;
    // 4. Умножаем матрицу на вектор: [out_features, in_features] x [in_features, 1] -> [out_features, 1]
    let result_col = weights.matmul(&input_col)
        .map_err(|e| Error::Runtime(format!("Candle matmul error: {}", e)))?;

    // 5. Убираем лишнюю размерность: [out_features, 1] -> [out_features]
    let result_vec = result_col.squeeze(1)
        .map_err(|e| Error::Runtime(format!("Candle squeeze error: {}", e)))?;
    // 6. Упаковываем результат обратно в dArray для L0
    candle_to_vector(&result_vec, heap)
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
    vm.register_rust_function("cuda_sub", vec![Type::DCudaTensor, Type::DCudaTensor], Type::DCudaTensor, cuda_sub).unwrap();
    //vm.register_rust_function("cuda_svdvals", vec![Type::DCudaTensor], Type::DCudaTensor, cuda_svdvals).unwrap();

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
    ////////////// candle //////////////
    vm.register_rust_function("candle_l0_layer", vec![Type::DTensor, Type::DArray], 
                              Type::Array(Box::new(Type::F32)), candle_l0_layer).unwrap();
    vm.register_rust_function("candle_matmul", vec![Type::DTensor, Type::DTensor], 
                              Type::DTensor, candle_matmul).unwrap();
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
