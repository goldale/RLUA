use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use half::{bf16, f16};

use tokenizers::Tokenizer;
use candle_core::safetensors::{Load, MmapedSafetensors};
use candle_core::{DType, Device, IndexOp, Tensor};

use crate::ext::NativeModule;
use crate::{Error, Heap, HeapObject, HeapRef, TableKey, Type, Value, Vm};

thread_local! {
    static TENSORS: RefCell<HashMap<i32, Tensor>> = RefCell::new(HashMap::new());
    static NEXT_TENSOR_ID: RefCell<i32> = RefCell::new(1);
    // ДОБАВЛЯЕМ Реестр для токенизаторов:
    static TOKENIZERS: RefCell<HashMap<i32, Tokenizer>> = RefCell::new(HashMap::new());
    static NEXT_TOKENIZER_ID: RefCell<i32> = RefCell::new(1);
}

fn insert_tensor(t: Tensor) -> i32 {
    NEXT_TENSOR_ID.with(|counter| {
        let mut id = counter.borrow_mut();
        let current = *id;
        *id += 1;
        TENSORS.with(|reg| reg.borrow_mut().insert(current, t));
        current
    })
}
fn get_tensor(id: i32) -> Result<Tensor, Error> {
    TENSORS.with(|reg| {
        reg.borrow()
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::Runtime(format!("Tensor ID {} not found in registry", id)))
    })
}
fn get_tensor_i32(heap: &Heap, reference: HeapRef) -> Result<(Vec<i32>, Vec<usize>), Error> {
    match heap.get(reference)? {
        HeapObject::Tensor { bytes, element, shape } if *element == Type::I32 => {
            let mut ints = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                ints.push(i32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok((ints, shape.clone()))
        }
        _ => Err(Error::Type("expected tensor<i32>".into())),
    }
}

// =========================================================================
// ФУНКЦИИ ИЗВЛЕЧЕНИЯ ИЗ ПАМЯТИ L0
// =========================================================================
fn get_tensor_f32(heap: &Heap, reference: HeapRef) -> Result<(Vec<f32>, Vec<usize>), Error> {
    match heap.get(reference)? {
        HeapObject::Tensor { bytes, element, shape } if *element == Type::F32 => {
            let mut floats = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                floats.push(f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())));
            }
            Ok((floats, shape.clone()))
        }
        _ => Err(Error::Type("expected tensor<f32>".into())),
    }
}

fn get_tensor_bf16(heap: &Heap, reference: HeapRef) -> Result<(Vec<bf16>, Vec<usize>), Error> {
    match heap.get(reference)? {
        HeapObject::Tensor { bytes, element, shape } if *element == Type::BF16 => {
            let mut floats = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                floats.push(bf16::from_bits(bits));
            }
            Ok((floats, shape.clone()))
        }
        _ => Err(Error::Type("expected tensor<bf16>".into())),
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
                floats.push(f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())));
            }
            Ok(floats)
        }
        _ => Err(Error::Type("expected vector<f32>".into())),
    }
}

fn get_array_bf16(heap: &Heap, reference: HeapRef) -> Result<Vec<bf16>, Error> {
    match heap.get(reference)? {
        HeapObject::Array { bytes, element } if *element == Type::BF16 => {
            let mut floats = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                floats.push(bf16::from_bits(bits));
            }
            Ok(floats)
        }
        _ => Err(Error::Type("expected vector<bf16>".into())),
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
        _ => Err(Error::Type("expected vector<f16>".into())),
    }
}

// =========================================================================
// КОНВЕРТЕРЫ ТЕНЗОРОВ (L0 <-> Candle)
// =========================================================================

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
            Tensor::from_vec(floats, shape.as_slice(), device).map_err(|e| Error::Runtime(e.to_string()))
        },
        Type::F16 => {
            let (floats, shape) = get_tensor_f16(heap, *ref_id)?;
            Tensor::from_vec(floats, shape.as_slice(), device).map_err(|e| Error::Runtime(e.to_string()))
        },
        Type::BF16 => {
            let (floats, shape) = get_tensor_bf16(heap, *ref_id)?;
            Tensor::from_vec(floats, shape.as_slice(), device).map_err(|e| Error::Runtime(e.to_string()))
        },
        _ => Err(Error::Runtime("Unsupported tensor element type".into()))
    }
}

fn candle_to_tensor(tensor: &Tensor, heap: &RefCell<Heap>) -> Result<Value, Error> {
    let cpu_tensor = tensor.to_device(&Device::Cpu).map_err(|e| Error::Runtime(e.to_string()))?;
    let shape = tensor.dims().to_vec();
    let rank = shape.len();

    match tensor.dtype() {
        DType::F32 => {
            let floats = cpu_tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut bytes = Vec::with_capacity(floats.len() * 4);
            for val in floats { bytes.extend_from_slice(&val.to_bits().to_le_bytes()); }
            let mut h = heap.borrow_mut();
            let new_ref = h.allocate(HeapObject::Tensor { bytes, element: Type::F32, shape: shape.clone() });
            Ok(Value::Tensor(new_ref, Rc::new(Type::Tensor(Box::new(Type::F32), rank))))
        },
        DType::F16 => {
            let floats = cpu_tensor.flatten_all().unwrap().to_vec1::<f16>().unwrap();
            let mut bytes = Vec::with_capacity(floats.len() * 2);
            for val in floats { bytes.extend_from_slice(&val.to_bits().to_le_bytes()); }
            let mut h = heap.borrow_mut();
            let new_ref = h.allocate(HeapObject::Tensor { bytes, element: Type::F16, shape: shape.clone() });
            Ok(Value::Tensor(new_ref, Rc::new(Type::Tensor(Box::new(Type::F16), rank))))
        },
        DType::BF16 => {
            let floats = cpu_tensor.flatten_all().unwrap().to_vec1::<bf16>().unwrap();
            let mut bytes = Vec::with_capacity(floats.len() * 2);
            for val in floats { bytes.extend_from_slice(&val.to_bits().to_le_bytes()); }
            let mut h = heap.borrow_mut();
            let new_ref = h.allocate(HeapObject::Tensor { bytes, element: Type::BF16, shape: shape.clone() });
            Ok(Value::Tensor(new_ref, Rc::new(Type::Tensor(Box::new(Type::BF16), rank))))
        },
        _ => Err(Error::Runtime("Unsupported Candle dtype".into()))
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
            Tensor::from_vec(floats, shape.as_slice(), device).map_err(|e| Error::Runtime(e.to_string()))
        },
        Type::F16 => {
            let floats = get_array_f16(heap, *ref_id)?;
            let shape = [floats.len()];
            Tensor::from_vec(floats, shape.as_slice(), device).map_err(|e| Error::Runtime(e.to_string()))
        },
        Type::BF16 => {
            let floats = get_array_bf16(heap, *ref_id)?;
            let shape = [floats.len()];
            Tensor::from_vec(floats, shape.as_slice(), device).map_err(|e| Error::Runtime(e.to_string()))
        },
        _ => Err(Error::Runtime("Unsupported vector type".into()))
    }
}

fn candle_to_vector(tensor: &Tensor, heap: &RefCell<Heap>) -> Result<Value, Error> {
    let cpu_tensor = tensor.to_device(&Device::Cpu).map_err(|e| Error::Runtime(e.to_string()))?;

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
        DType::BF16 => {
            let floats = cpu_tensor.flatten_all().unwrap().to_vec1::<bf16>().unwrap();
            let mut bytes = Vec::with_capacity(floats.len() * 2);
            for val in floats { bytes.extend_from_slice(&val.to_bits().to_le_bytes()); }
            let mut h = heap.borrow_mut();
            let new_ref = h.allocate(HeapObject::Array { bytes, element: Type::BF16 });
            Ok(Value::Array(new_ref, Box::new(Type::BF16)))
        },
        _ => Err(Error::Runtime("Unsupported Candle dtype".into()))
    }
}

// =========================================================================
// ВЫСОКОУРОВНЕВЫЕ ОПЕРАЦИИ (Legacy)
// =========================================================================

fn candle_load_model(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::String(ref_path)] = arguments else {
        return Err(Error::Runtime("candle_load_model expects 1 string (path)".into()));
    };
    let path = match heap.borrow().get(*ref_path)? {
        HeapObject::String(text) => text.clone(),
        _ => return Err(Error::Runtime("invalid string heap object".into())),
    };

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let safetensors = unsafe { MmapedSafetensors::new(&path) }.map_err(|e| Error::Runtime(e.to_string()))?;
    let mut table_entries = HashMap::new();

    for (name, tensor_view) in safetensors.tensors() {
        let tensor = tensor_view.load(&device).map_err(|e| Error::Runtime(e.to_string()))?;
        let id = insert_tensor(tensor);
        table_entries.insert(TableKey::Name(Rc::from(name.as_str())), Value::I32(id));
    }
    let mut h = heap.borrow_mut();
    let table_ref = h.allocate(HeapObject::Table { entries: table_entries, element: Type::I32 });
    Ok(Value::Table(table_ref, Box::new(Type::I32)))
}
fn candle_save_model(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // Ожидаем таблицу с тензорами и путь сохранения
    let [Value::Table(ref_table, _), Value::String(ref_path)] = arguments else {
        return Err(Error::Runtime("candle_save_model expects (table<i32>, string)".into()));
    };
    
    // 1. Достаем путь сохранения из кучи L0
    let path = match heap.borrow().get(*ref_path)? {
        HeapObject::String(text) => text.clone(),
        _ => return Err(Error::Runtime("invalid string heap object".into())),
    };

    let mut tensors_to_save = std::collections::HashMap::new();
    
    // 2. Итерируемся по таблице L0
    if let HeapObject::Table { entries, .. } = heap.borrow().get(*ref_table)? {
        for (key, val) in entries {
            // Извлекаем имя тензора и его ID в реестре Candle
            if let (TableKey::Name(name), Value::I32(id)) = (key, val) {
                let mut tensor = get_tensor(*id)?;
                
                // АВТОКОНВЕРТАЦИЯ: Если тензор в BF16, переводим его в F32
                if tensor.dtype() == candle_core::DType::BF16 {
                    tensor = tensor.to_dtype(candle_core::DType::F32).map_err(|e| Error::Runtime(e.to_string()))?;
                }
                
                tensors_to_save.insert(name.to_string(), tensor);
            }
        }
    } else {
        return Err(Error::Runtime("invalid table heap object".into()));
    }
    // 3. Сохраняем в формат safetensors средствами Candle
    candle_core::safetensors::save(&tensors_to_save, &path).map_err(|e| Error::Runtime(e.to_string()))?;
    
    Ok(Value::Bool(true))
}

fn candle_embedding(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(weights_id), Value::I32(token_id)] = arguments else { return Err(Error::Runtime("expects (i32, i32)".into())); };
    let weights = get_tensor(*weights_id)?;
    let result = weights.i((*token_id as usize, ..)).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(Value::I32(insert_tensor(result)))
}

fn candle_softmax(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(tensor_id)] = arguments else { return Err(Error::Runtime("expects i32".into())); };
    let tensor = get_tensor(*tensor_id)?;
    let result = candle_nn::ops::softmax(&tensor, candle_core::D::Minus1).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(Value::I32(insert_tensor(result)))
}

fn candle_rmsnorm(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(tensor_id), Value::I32(weight_id), Value::F32(eps)] = arguments else { return Err(Error::Runtime("expects (i32, i32, f32)".into())); };
    let x = get_tensor(*tensor_id)?;
    let weight = get_tensor(*weight_id)?;

    let x_sq = x.sqr().map_err(|e| Error::Runtime(e.to_string()))?;
    let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1).map_err(|e| Error::Runtime(e.to_string()))?;
    let std = (mean_sq + *eps as f64).unwrap().sqrt().map_err(|e| Error::Runtime(e.to_string()))?;
    let norm = x.broadcast_div(&std).map_err(|e| Error::Runtime(e.to_string()))?;
    let result = norm.broadcast_mul(&weight).map_err(|e| Error::Runtime(e.to_string()))?;

    Ok(Value::I32(insert_tensor(result)))
}

fn candle_free_tensor(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(id)] = arguments else { return Err(Error::Runtime("expects i32".into())); };
    TENSORS.with(|reg| reg.borrow_mut().remove(id));
    Ok(Value::Bool(true))
}

fn candle_tensor_shape(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [arg] = arguments else { return Err(Error::Runtime("expects 1 dTensor".into())); };
    let device = Device::Cpu;
    let tensor = tensor_to_candle(arg, &heap.borrow(), &device)?;
    let shape_f32: Vec<f32> = tensor.dims().iter().map(|&d| d as f32).collect();
    let shape_tensor = Tensor::from_vec(shape_f32, vec![tensor.dims().len()], &device).map_err(|e| Error::Runtime(e.to_string()))?;
    candle_to_vector(&shape_tensor, heap)
}

fn candle_l0_layer(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let [arg_w, arg_x] = arguments else { return Err(Error::Runtime("expects 1 tensor and 1 vector".into())); };
    let weights = tensor_to_candle(arg_w, &heap.borrow(), &device)?;
    let input_vec = vector_to_candle(arg_x, &heap.borrow(), &device)?;

    let input_col = input_vec.unsqueeze(1).map_err(|e| Error::Runtime(e.to_string()))?;
    let result_col = weights.matmul(&input_col).map_err(|e| Error::Runtime(e.to_string()))?;
    let result_vec = result_col.squeeze(1).map_err(|e| Error::Runtime(e.to_string()))?;
    candle_to_vector(&result_vec, heap)
}

fn candle_vector_add(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let [arg_a, arg_b] = arguments else { return Err(Error::Runtime("expects 2 dArrays".into())); };
    let vec_a = vector_to_candle(arg_a, &heap.borrow(), &device)?;
    let vec_b = vector_to_candle(arg_b, &heap.borrow(), &device)?;
    let result = vec_a.broadcast_add(&vec_b).map_err(|e| Error::Runtime(e.to_string()))?;
    candle_to_vector(&result, heap)
}

fn candle_vector_relu(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let [arg] = arguments else { return Err(Error::Runtime("expects 1 dArray".into())); };
    let input_vec = vector_to_candle(arg, &heap.borrow(), &device)?;
    let result = input_vec.relu().map_err(|e| Error::Runtime(e.to_string()))?;
    candle_to_vector(&result, heap)
}

fn candle_matmul(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let tensor_a = tensor_to_candle(&arguments[0], &heap.borrow(), &device)?;
    let tensor_b = tensor_to_candle(&arguments[1], &heap.borrow(), &device)?;
    let result = tensor_a.matmul(&tensor_b).map_err(|e| Error::Runtime(e.to_string()))?;
    candle_to_tensor(&result, heap)
}

fn candle_load_safetensor(arguments: &[Value], heap: &RefCell<crate::Heap>) -> Result<Value, Error> {
    let [arg_path, arg_name] = arguments else { return Err(Error::Runtime("expects 2 strings".into())); };
    let h = heap.borrow();
    let path = match arg_path {
        Value::String(ref_id) => match h.get(*ref_id)? { HeapObject::String(text) => text.clone(), _ => return Err(Error::Runtime("invalid string".into())) },
        _ => return Err(Error::Type("path must be a string".into())),
    };
    let name = match arg_name {
        Value::String(ref_id) => match h.get(*ref_id)? { HeapObject::String(text) => text.clone(), _ => return Err(Error::Runtime("invalid string".into())) },
        _ => return Err(Error::Type("tensor_name must be a string".into())),
    };
    drop(h);

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let tensors = candle_core::safetensors::load(&path, &device).map_err(|e| Error::Runtime(e.to_string()))?;
    let tensor = tensors.get(&name).ok_or_else(|| Error::Runtime(format!("Tensor '{}' not found", name)))?;
    candle_to_tensor(tensor, heap)
}

// =========================================================================
// ФУНКЦИИ ВЫЧИСЛЕНИЙ ПО ID (LLM Инференс в VRAM)
// =========================================================================

fn candle_linear_id(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(w_id), Value::I32(x_id)] = arguments else { return Err(Error::Runtime("expects (i32, i32)".into())); };
    let w = get_tensor(*w_id)?;
    let x = get_tensor(*x_id)?;
    let x_col = x.unsqueeze(1).map_err(|e| Error::Runtime(e.to_string()))?;
    let res_col = w.matmul(&x_col).map_err(|e| Error::Runtime(e.to_string()))?;
    let res = res_col.squeeze(1).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(Value::I32(insert_tensor(res)))
}

fn candle_silu(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(id)] = arguments else { return Err(Error::Runtime("expects i32".into())); };
    let t = get_tensor(*id)?;
    let res = candle_nn::ops::silu(&t).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(Value::I32(insert_tensor(res)))
}

fn candle_mul_id(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(a_id), Value::I32(b_id)] = arguments else { return Err(Error::Runtime("expects (i32, i32)".into())); };
    let a = get_tensor(*a_id)?;
    let b = get_tensor(*b_id)?;
    let res = a.broadcast_mul(&b).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(Value::I32(insert_tensor(res)))
}

fn candle_get_darray(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(id)] = arguments else { return Err(Error::Runtime("expects i32".into())); };
    let t = get_tensor(*id)?;
    candle_to_vector(&t, heap)
}

fn candle_reshape_id(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(t_id), Value::Array(ref_id, _)] = arguments else { return Err(Error::Runtime("expects (i32, dArray)".into())); };
    let t = get_tensor(*t_id)?;
    let floats = get_array_f32(&heap.borrow(), *ref_id)?;
    let dims: Vec<usize> = floats.into_iter().map(|f| f as usize).collect();
    let res = t.reshape(dims.as_slice()).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(Value::I32(insert_tensor(res)))
}

fn candle_transpose_id(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(t_id), Value::I32(dim1), Value::I32(dim2)] = arguments else { return Err(Error::Runtime("expects (i32, i32, i32)".into())); };
    let t = get_tensor(*t_id)?;
    let res = t.transpose(*dim1 as usize, *dim2 as usize).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(Value::I32(insert_tensor(res)))
}

fn candle_repeat_kv_id(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(t_id), Value::I32(n_rep)] = arguments else { return Err(Error::Runtime("expects (i32, i32)".into())); };
    let t = get_tensor(*t_id)?;
    let dims = t.dims();
    let n_rep = *n_rep as usize;

    if dims.len() == 3 {
        let (b, heads, hd) = (dims[0], dims[1], dims[2]);
        let res = t.unsqueeze(2).map_err(|e| Error::Runtime(e.to_string()))?
            .broadcast_as((b, heads, n_rep, hd)).map_err(|e| Error::Runtime(e.to_string()))?
            .reshape((b, heads * n_rep, hd)).map_err(|e| Error::Runtime(e.to_string()))?;
        Ok(Value::I32(insert_tensor(res)))
    } else if dims.len() == 4 {
        let (b, heads, seq, hd) = (dims[0], dims[1], dims[2], dims[3]);
        let res = t.unsqueeze(2).map_err(|e| Error::Runtime(e.to_string()))?
            .broadcast_as((b, heads, n_rep, seq, hd)).map_err(|e| Error::Runtime(e.to_string()))?
            .reshape((b, heads * n_rep, seq, hd)).map_err(|e| Error::Runtime(e.to_string()))?;
        Ok(Value::I32(insert_tensor(res)))
    } else {
        Err(Error::Runtime("repeat_kv expects 3D or 4D tensor".into()))
    }
}

fn candle_scale_id(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(t_id), Value::F32(scale)] = arguments else { return Err(Error::Runtime("expects (i32, f32)".into())); };
    let t = get_tensor(*t_id)?;
    let res = (&t * (*scale as f64)).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(Value::I32(insert_tensor(res)))
}

fn candle_matmul_id(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(a_id), Value::I32(b_id)] = arguments else { return Err(Error::Runtime("expects (i32, i32)".into())); };
    let a = get_tensor(*a_id)?;
    let b = get_tensor(*b_id)?;
    let res = a.broadcast_matmul(&b).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(Value::I32(insert_tensor(res)))
}

fn candle_apply_rope_id(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(t_id), Value::I32(pos), Value::I32(head_dim)] = arguments else { return Err(Error::Runtime("expects (i32, i32, i32)".into())); };
    let t = get_tensor(*t_id)?;
    let device = t.device().clone();
    let pos = *pos as usize;
    let head_dim = *head_dim as usize;

    let mut inv_freqs = Vec::new();
    for i in (0..head_dim).step_by(2) {
        inv_freqs.push(1.0 / 10000.0_f32.powf(i as f32 / head_dim as f32));
    }
    let inv_freqs = Tensor::from_vec(inv_freqs, (1, head_dim / 2), &device).map_err(|e| Error::Runtime(e.to_string()))?;
    let t_pos = Tensor::new(&[pos as f32], &device).unwrap().reshape((1, 1)).unwrap();
    let freqs = t_pos.matmul(&inv_freqs).map_err(|e| Error::Runtime(e.to_string()))?;

    let emb = Tensor::cat(&[&freqs, &freqs], 1).map_err(|e| Error::Runtime(e.to_string()))?;
    let cos = emb.cos().unwrap().broadcast_as(t.shape()).map_err(|e| Error::Runtime(e.to_string()))?;
    let sin = emb.sin().unwrap().broadcast_as(t.shape()).map_err(|e| Error::Runtime(e.to_string()))?;

    let t_half1 = t.narrow(candle_core::D::Minus1, 0, head_dim / 2).map_err(|e| Error::Runtime(e.to_string()))?;
    let t_half2 = t.narrow(candle_core::D::Minus1, head_dim / 2, head_dim / 2).map_err(|e| Error::Runtime(e.to_string()))?;
    let t_half2_neg = t_half2.neg().map_err(|e| Error::Runtime(e.to_string()))?;
    let t_rotated = Tensor::cat(&[&t_half2_neg, &t_half1], candle_core::D::Minus1).map_err(|e| Error::Runtime(e.to_string()))?;

    let res = t.broadcast_mul(&cos).unwrap().broadcast_add(&t_rotated.broadcast_mul(&sin).unwrap()).unwrap();
    Ok(Value::I32(insert_tensor(res)))
}

fn candle_softmax_dim_id(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(t_id), Value::I32(dim)] = arguments else { return Err(Error::Runtime("expects (i32, i32)".into())); };
    let t = get_tensor(*t_id)?;
    let res = candle_nn::ops::softmax(&t, *dim as usize).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(Value::I32(insert_tensor(res)))
}

fn candle_add_id(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(a_id), Value::I32(b_id)] = arguments else { return Err(Error::Runtime("expects (i32, i32)".into())); };
    let a = get_tensor(*a_id)?;
    let b = get_tensor(*b_id)?;
    let res = a.broadcast_add(&b).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(Value::I32(insert_tensor(res)))
}

fn candle_cat_id(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(a_id), Value::I32(b_id), Value::I32(dim)] = arguments else { return Err(Error::Runtime("expects (i32, i32, i32)".into())); };
    let a = get_tensor(*a_id)?;
    let b = get_tensor(*b_id)?;
    let res = Tensor::cat(&[&a, &b], *dim as usize).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(Value::I32(insert_tensor(res)))
}

// Token Gen
// Загружает tokenizer.json с диска
fn host_tokenizer_load(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::String(ref_path)] = arguments else {
        return Err(Error::Runtime("tokenizer_load expects 1 string (path)".into()));
    };

    let path = {
        let h = heap.borrow();
        match h.get(*ref_path)? {
            HeapObject::String(text) => text.clone(),
            _ => return Err(Error::Runtime("invalid string heap object".into())),
        }
    };
    // Загружаем токенизатор
    let tokenizer = Tokenizer::from_file(&path)
        .map_err(|e| Error::Runtime(format!("Failed to load tokenizer: {}", e)))?;

    // Сохраняем в реестр и возвращаем ID
    let id = NEXT_TOKENIZER_ID.with(|counter| {
        let mut c = counter.borrow_mut();
        let current = *c;
        *c += 1;
        TOKENIZERS.with(|reg| reg.borrow_mut().insert(current, tokenizer));
        current
    });
    Ok(Value::I32(id))
}
// Переводит ID токена в строку
fn host_tokenizer_decode(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(tok_id), Value::I32(token)] = arguments else {
        return Err(Error::Runtime("tokenizer_decode expects (tokenizer_id: i32, token: i32)".into()));
    };
    let text = TOKENIZERS.with(|reg| {
        let r = reg.borrow();
        let tokenizer = r.get(tok_id).ok_or_else(|| Error::Runtime("Tokenizer not found".into()))?;
        // Получаем сырое представление токена напрямую из словаря
        let raw_token = tokenizer.id_to_token(*token as u32).unwrap_or_default();
        // Вручную меняем спецсимволы токенизатора на пробелы и переносы строк
        let clean_text = raw_token
            .replace('\u{2581}', " ") // SentencePiece (используется в TinyLlama/Llama)
            .replace('Ġ', " ")        // ByteLevel BPE (используется в GPT-2/Llama 3)
            .replace("<0x0A>", "\n"); // Явный токен переноса строки
            
        Ok::<String, Error>(clean_text)
    })?;
    // Аллоцируем строку в куче L0
    let mut h = heap.borrow_mut();
    let ref_id = h.allocate(HeapObject::String(text));
    Ok(Value::String(ref_id))
}
// Переводит строку текста в массив ID токенов (vector<i32>)
fn host_tokenizer_encode(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(tok_id), Value::String(ref_text)] = arguments else {
        return Err(Error::Runtime("tokenizer_encode expects (tokenizer_id: i32, text: string)".into()));
    };

    // 1. Извлекаем строку промпта из кучи L0
    let text = {
        let h = heap.borrow();
        match h.get(*ref_text)? {
            HeapObject::String(s) => s.clone(),
            _ => return Err(Error::Runtime("invalid string heap object".into())),
        }
    };
    // 2. Кодируем текст через HuggingFace Tokenizer
    let ids = TOKENIZERS.with(|reg| {
        let r = reg.borrow();
        let tokenizer = r.get(tok_id).ok_or_else(|| Error::Runtime("Tokenizer not found".into()))?;
        // true означает добавление спец-токенов (например <|endoftext|> или <BOS>)
        let encoding = tokenizer.encode(text, true)
            .map_err(|e| Error::Runtime(format!("Encode error: {}", e)))?;
        Ok::<Vec<u32>, Error>(encoding.get_ids().to_vec())
    })?;
    // 3. Упаковываем массив токенов (u32 -> i32) в кучу L0
    let mut bytes = Vec::with_capacity(ids.len() * 4);
    for id in ids {
        bytes.extend_from_slice(&(id as i32).to_le_bytes());
    }
    let mut h = heap.borrow_mut();
    let new_ref = h.allocate(HeapObject::Array { bytes, element: Type::I32 });
    // Возвращаем строго типизированный vector<i32>
    Ok(Value::Array(new_ref, Box::new(Type::I32)))
}
// Выбор токена с максимальной вероятностью (работает по ID тензора)
fn candle_argmax_token(arguments: &[Value], _heap: &RefCell<Heap>) -> Result<Value, Error> {
    let [Value::I32(t_id)] = arguments else {
        return Err(Error::Runtime("candle_argmax_token expects 1 i32 (tensor ID)".into()));
    };

    // Достаем тензор напрямую из нашего реестра
    let t = get_tensor(*t_id)?;

    // Безопасно переводим в f32 на CPU (работает и для f32, и для f16 моделей)
    let cpu_logits = t.to_device(&Device::Cpu).map_err(|e| Error::Runtime(e.to_string()))?
        .to_dtype(DType::F32).map_err(|e| Error::Runtime(e.to_string()))?;

    let res = cpu_logits.flatten_all().map_err(|e| Error::Runtime(e.to_string()))?
        .argmax(0).map_err(|e| Error::Runtime(e.to_string()))?;

    let token_id = res.to_scalar::<u32>().map_err(|e| Error::Runtime(e.to_string()))? as i32;
    Ok(Value::I32(token_id))
}
fn candle_sample_token(arguments: &[Value], heap: &RefCell<Heap>) -> Result<Value, Error> {
    // Новая сигнатура: (logits_id: i32, temp: f32, context: dTensor, ctx_len: i32, penalty: f32)
    let [Value::I32(t_id), Value::F32(temperature), Value::Tensor(ref_ctx, _), Value::I32(ctx_len), Value::F32(penalty)] = arguments else {
        return Err(Error::Runtime("candle_sample_token expects (i32, f32, dTensor, i32, f32)".into()));
    };
    let temp = *temperature;
    let penalty = *penalty;
    let ctx_len = *ctx_len as usize;

    // Достаем историю токенов из скрипта L0
    let (context, _) = get_tensor_i32(&heap.borrow(), *ref_ctx)?;

    let t = get_tensor(*t_id)?;
    let cpu_logits = t.to_device(&Device::Cpu).map_err(|e| Error::Runtime(e.to_string()))?
                      .to_dtype(DType::F32).map_err(|e| Error::Runtime(e.to_string()))?;

    let mut logits_vec = cpu_logits.flatten_all().map_err(|e| Error::Runtime(e.to_string()))?
                                   .to_vec1::<f32>().map_err(|e| Error::Runtime(e.to_string()))?;

    // 1. ПРИМЕНЯЕМ REPETITION PENALTY
    if penalty > 1.0 {
        for i in 0..ctx_len {
            if i < context.len() {
                let tok = context[i] as usize;
                if tok < logits_vec.len() {
                    let logit = logits_vec[tok];
                    // Если вероятность положительная, делим её на штраф, если отрицательная — умножаем
                    if logit > 0.0 {
                        logits_vec[tok] = logit / penalty;
                    } else {
                        logits_vec[tok] = logit * penalty;
                    }
                }
            }
        }
    }
    // Если температура <= 0, делаем обычный argmax с учетом штрафа
    if temp <= 0.0 {
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, &val) in logits_vec.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        return Ok(Value::I32(max_idx as i32));
    }

    // 2. Температура
    for val in logits_vec.iter_mut() { *val /= temp; }

    // 3. Softmax
    let max_logit = logits_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum_exp = 0.0;
    for val in logits_vec.iter_mut() {
        *val = (*val - max_logit).exp();
        sum_exp += *val;
    }
    for val in logits_vec.iter_mut() { *val /= sum_exp; }

    // 4. Случайная выборка (Categorical Sampling)
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros();
    let mut state = now as u64;
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    let random_p = (state as f64) / (u64::MAX as f64);

    let mut cdf = 0.0;
    let mut selected_token = 0;
    for (i, &prob) in logits_vec.iter().enumerate() {
        cdf += prob as f64;
        if random_p < cdf {
            selected_token = i as i32;
            break;
        }
    }
    Ok(Value::I32(selected_token))
}

// =========================================================================
// РЕГИСТРАЦИЯ МОДУЛЯ
// =========================================================================

pub struct CandleExtension;
impl NativeModule for CandleExtension {
    fn name(&self) -> &str { "candle" }
    fn register(&self, vm: &mut Vm) -> Result<(), Error> {
        vm.register_rust_function("candle_l0_layer", vec![Type::DTensor, Type::DArray], Type::DArray, candle_l0_layer)?;
        vm.register_rust_function("candle_matmul", vec![Type::DTensor, Type::DTensor], Type::DTensor, candle_matmul)?;
        vm.register_rust_function("tensor_shape", vec![Type::DTensor], Type::DArray, candle_tensor_shape)?;
        vm.register_rust_function("candle_vector_add", vec![Type::DArray, Type::DArray], Type::DArray, candle_vector_add)?;
        vm.register_rust_function("candle_vector_relu", vec![Type::DArray], Type::DArray, candle_vector_relu)?;
        vm.register_rust_function("candle_load_safetensor", vec![Type::String, Type::String], Type::DTensor, candle_load_safetensor)?;
        vm.register_rust_function("candle_load_model", vec![Type::String], Type::Table(Box::new(Type::I32)), candle_load_model)?;
        vm.register_rust_function("candle_save_model", vec![Type::Table(Box::new(Type::I32)), Type::String], Type::Bool, candle_save_model)?;
        vm.register_rust_function("candle_embedding", vec![Type::I32, Type::I32], Type::I32, candle_embedding)?;
        vm.register_rust_function("candle_softmax", vec![Type::I32], Type::I32, candle_softmax)?;
        vm.register_rust_function("candle_rmsnorm", vec![Type::I32, Type::I32, Type::F32], Type::I32, candle_rmsnorm)?;
        vm.register_rust_function("candle_free_tensor", vec![Type::I32], Type::Bool, candle_free_tensor)?;

        vm.register_rust_function("candle_linear_id", vec![Type::I32, Type::I32], Type::I32, candle_linear_id)?;
        vm.register_rust_function("candle_silu", vec![Type::I32], Type::I32, candle_silu)?;
        vm.register_rust_function("candle_mul_id", vec![Type::I32, Type::I32], Type::I32, candle_mul_id)?;
        vm.register_rust_function("candle_get_darray", vec![Type::I32], Type::DArray, candle_get_darray)?;
        vm.register_rust_function("candle_reshape_id", vec![Type::I32, Type::DArray], Type::I32, candle_reshape_id)?;
        vm.register_rust_function("candle_transpose_id", vec![Type::I32, Type::I32, Type::I32], Type::I32, candle_transpose_id)?;
        vm.register_rust_function("candle_repeat_kv_id", vec![Type::I32, Type::I32], Type::I32, candle_repeat_kv_id)?;
        vm.register_rust_function("candle_scale_id", vec![Type::I32, Type::F32], Type::I32, candle_scale_id)?;
        vm.register_rust_function("candle_matmul_id", vec![Type::I32, Type::I32], Type::I32, candle_matmul_id)?;
        vm.register_rust_function("candle_apply_rope_id", vec![Type::I32, Type::I32, Type::I32], Type::I32, candle_apply_rope_id)?;
        vm.register_rust_function("candle_softmax_dim_id", vec![Type::I32, Type::I32], Type::I32, candle_softmax_dim_id)?;
        vm.register_rust_function("candle_add_id", vec![Type::I32, Type::I32], Type::I32, candle_add_id)?;
        vm.register_rust_function("candle_cat_id", vec![Type::I32, Type::I32, Type::I32], Type::I32, candle_cat_id)?;
        vm.register_rust_function("tokenizer_load", vec![Type::String], Type::I32, host_tokenizer_load)?;
        // Обратите внимание: возвращаем Type::String для вывода в консоль
        vm.register_rust_function("tokenizer_decode", vec![Type::I32, Type::I32], Type::String, host_tokenizer_decode)?;
        vm.register_rust_function(
            "candle_argmax_token",
            vec![Type::I32], // <--- ИСПРАВЛЕНО НА I32
            Type::I32,
            candle_argmax_token
        ).unwrap();
        vm.register_rust_function(
            "tokenizer_encode",
            vec![Type::I32, Type::String],
            Type::Array(Box::new(Type::I32)),
            host_tokenizer_encode
        )?;
        vm.register_rust_function(
            "candle_sample_token",
            vec![Type::I32, Type::F32, Type::DTensor, Type::I32, Type::F32], // <--- Обновленные типы
            Type::I32,
            candle_sample_token
        ).unwrap();

        Ok(())
    }
}
