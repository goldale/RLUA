use crate::{Error, Heap, HeapObject, HeapRef, Type, Value, Vm};
use super::NativeModule;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::OnceLock;

use half::f16;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use candle_core::{Tensor, Device, DType};

static CUDA_DEV: OnceLock<std::sync::Arc<CudaDevice>> = OnceLock::new();

// Код ядра на CUDA C++
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
        let opts = CompileOptions {arch: Some("sm_61"), ..Default::default()};
        let ptx = compile_ptx_with_opts(CUDA_KERNELS, opts)
            .expect("Не удалось скомпилировать PTX ядро");
        dev.load_ptx(ptx, "my_kernels", &["kernel_sub"]).expect("Не удалось загрузить ядро");
        dev
    }).clone()
}

// --- Хелперы для извлечения данных из кучи VM ---

fn get_tensor_f32(heap: &Heap, reference: HeapRef) -> Result<(Vec<f32>, Vec<usize>), Error> {
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

fn get_tensor_f16(heap: &Heap, reference: HeapRef) -> Result<(Vec<f16>, Vec<usize>), Error> {
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

fn get_array_f32(heap: &Heap, reference: HeapRef) -> Result<Vec<f32>, Error> {
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

fn get_array_f16(heap: &Heap, reference: HeapRef) -> Result<Vec<f16>, Error> {
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

// --- Хелперы для конвертации между L0 и Candle ---

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

// --- Нативные функции (L0RustFunction) ---
fn to_cuda(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::Tensor(ref_id, _)] = arguments else {
        return Err(Error::Runtime("to_cuda expects 1 dTensor".into()));
    };

    let dev = get_cuda_device();

    let (gpu_data, element_cloned, shape_cloned) = {
        let h = heap.borrow();
        let (bytes, element, shape) = match h.get(*ref_id)? {
            HeapObject::Tensor { bytes, element, shape } => (bytes, element, shape),
            _ => return Err(Error::Runtime("invalid tensor".into())),
        };
        let gpu_data: CudaSlice<u8> = dev.htod_sync_copy(bytes)
            .map_err(|e| Error::Runtime(format!("CUDA HTOD: {}", e)))?;
        (gpu_data, element.clone(), shape.clone())
    };

    let mut h_mut = heap.borrow_mut();
    let new_ref = h_mut.allocate(HeapObject::CudaTensor {
        data: gpu_data,
        element: element_cloned,
        shape: shape_cloned,
    });
    Ok(Value::CudaTensor(new_ref, std::rc::Rc::new(Type::DCudaTensor)))
}

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
    };

    let mut h_mut = heap.borrow_mut();
    let new_ref = h_mut.allocate(HeapObject::Tensor {
        bytes: cpu_bytes,
        element: element_cloned,
        shape: shape_cloned,
    });
    Ok(Value::Tensor(new_ref, std::rc::Rc::new(Type::DTensor)))
}

fn cuda_sub(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::CudaTensor(ref_a, _), Value::CudaTensor(ref_b, _)] = arguments else {
        return Err(Error::Runtime("cuda_sub expects 2 dCudaTensors".into()));
    };
    let dev = get_cuda_device();
    
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
        
        out_data = dev.alloc_zeros::<u8>(num_elements * 4)
            .map_err(|e| Error::Runtime(e.to_string()))?;
        let f = dev.get_func("my_kernels", "kernel_sub").unwrap();
        let cfg = LaunchConfig::for_num_elems(num_elements as u32);
        
        unsafe { f.launch(cfg, (data_a, data_b, &mut out_data, num_elements as i32)) }
            .map_err(|e| Error::Runtime(format!("CUDA launch failed: {}", e)))?;
    }
    
    let mut h_mut = heap.borrow_mut();
    let new_ref = h_mut.allocate(HeapObject::CudaTensor {
        data: out_data,
        element: Type::F32,
        shape: shape_cloned,
    });
    Ok(Value::CudaTensor(new_ref, std::rc::Rc::new(Type::DCudaTensor)))
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
    let [arg_w, arg_x] = arguments else {
        return Err(Error::Runtime("candle_l0_layer expects 1 tensor and 1 vector".into()));
    };
    let device = Device::new_cuda(0).map_err(|e| Error::Runtime(e.to_string()))?;
    let weights = tensor_to_candle(arg_w, &heap.borrow(), &device)?;
    let input_vec = vector_to_candle(arg_x, &heap.borrow(), &device)?;
    
    let input_col = input_vec.unsqueeze(1)
        .map_err(|e| Error::Runtime(format!("Candle unsqueeze error: {}", e)))?;
    let result_col = weights.matmul(&input_col)
        .map_err(|e| Error::Runtime(format!("Candle matmul error: {}", e)))?;
    let result_vec = result_col.squeeze(1)
        .map_err(|e| Error::Runtime(format!("Candle squeeze error: {}", e)))?;
        
    candle_to_vector(&result_vec, heap)
}

// --- Регистрация модуля ---
pub struct CandleExtension;

impl NativeModule for CandleExtension {
    fn name(&self) -> &str {
        "candle"
    }

    fn register(&self, vm: &mut Vm) -> Result<(), Error> {
        vm.register_rust_function("to_cuda", vec![Type::DTensor], Type::DCudaTensor, to_cuda)?;
        vm.register_rust_function("to_cpu", vec![Type::DCudaTensor], Type::DTensor, to_cpu)?;
        vm.register_rust_function("cuda_sub", vec![Type::DCudaTensor, Type::DCudaTensor], Type::DCudaTensor, cuda_sub)?;
        
        vm.register_rust_function("candle_l0_layer", vec![Type::DTensor, Type::DArray], 
                                  Type::Array(Box::new(Type::F32)), candle_l0_layer)?;
        vm.register_rust_function("candle_matmul", vec![Type::DTensor, Type::DTensor], 
                                  Type::DTensor, candle_matmul)?;
        Ok(())
    }
}
