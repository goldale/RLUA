use super::*;
use crate::compiler::*;

#[test]
fn float_literals_cannot_claim_non_float_types() {
    assert!(matches!(execute("let value: i32 = 1.0"), Err(Error::Located { source, location: SourceLocation { line: 1, column: 18, .. } }) if matches!(*source, Error::Type(_))));
    assert!(matches!(execute("let value: bool = 1.0"), Err(Error::Located { source, location: SourceLocation { line: 1, column: 19, .. } }) if matches!(*source, Error::Type(_))));
}
#[test]
fn specialized_i32_addition_reports_overflow() {
    assert!(matches!(
        execute("let value: i32 = 2147483647; print(value + 1)"),
        Err(Error::Runtime(message)) if message == "addition overflow"
    ));
}
#[test]
fn unary_not_is_lexed_parsed_and_evaluated() {
    assert_eq!(execute("let no: bool = 1 == 0; print(!no)").unwrap(), vec!["true".to_owned()]);
}
#[test]
fn strings_compare_by_content() {
    assert_eq!(
        execute("let left: string = \"same\"; let right: string = \"same\"; print(left == right)").unwrap(),
        vec!["true".to_owned()]
    );
}
#[test]
fn block_locals_do_not_escape_or_consume_slots_forever() {
    assert!(matches!(
        execute("if 1 == 1 then let inner: i32 = 7 end print(inner)"),
        Err(Error::Located { source, location: SourceLocation { line: 1, column: 45, .. } }) if matches!(&*source, Error::Type(message) if message == "unknown name 'inner'")
    ));
    assert!(matches!(
        execute("for i = 1, 2 do let temporary: i32 = i end print(i)"),
        Err(Error::Located { source, location: SourceLocation { line: 1, column: 50, .. } }) if matches!(&*source, Error::Type(message) if message == "unknown name 'i'")
    ));
}
#[test]
fn scope_test_covers_all_normal_raii_exit_paths() {
    let expected = vec![
        "RAII: declaration order", "block body", "drop 2", "drop 1",
        "RAII: if then", "drop 3", "RAII: if else", "drop 4",
        "RAII: nested scopes", "drop 6", "drop 5",
        "RAII: while fallthrough", "drop 10", "drop 11",
        "RAII: while continue", "drop 21", "drop 22",
        "RAII: while break", "drop 30",
        "RAII: nested break", "drop 41", "drop 40",
        "RAII: for fallthrough", "drop 51", "drop 52",
        "RAII: for continue", "drop 61", "drop 62",
        "RAII: for break", "drop 66",
        "RAII: method return", "method body", "drop 70",
        "RAII: no destructor", "80", "RAII scope tests completed",
    ].into_iter().map(str::to_owned).collect::<Vec<_>>();
    let output = execute(include_str!("../examples/scope_tst.l0")).unwrap();
    let raii_start = output.iter().position(|line| line == "RAII: declaration order")
        .expect("scope test must reach the RAII section");
    assert_eq!(&output[raii_start..], expected);
}
#[test]
fn runtime_error_unwinds_live_structs_in_reverse_order() {
    let source = r#"
            struct CrashGuard {
                id: i32;
                function CrashGuard() print(id) end
            }
            struct Crasher { function fail() end }
            function Crasher::fail()
                let outer: CrashGuard = CrashGuard { id = 90 }
                if 1 == 1 then
                    let inner: CrashGuard = CrashGuard { id = 91 }
                    let zero: i32 = 0
                    print(1 / zero)
                end
            end
            let crasher: Crasher = Crasher { }
            crasher.fail()
        "#;
    let mut vm = Vm::default();
    assert!(matches!(vm.execute(source), Err(Error::Runtime(message)) if message.contains("division by zero")));
    assert_eq!(vm.output, ["91", "90"]);
    assert!(vm.active_destructors.is_empty());
}
#[test]
fn exported_function_drops_its_local_before_implicit_return() {
    let source = r#"
            struct ExportGuard {
                function ExportGuard() print(100) end
            }
            export function work()
                let guard: ExportGuard = ExportGuard { }
                print(101)
            end
        "#;
    let (program, strings) = Parser::new(lex(source).unwrap()).into_program().unwrap();
    let artifact = Compiler::default().with_strings(strings).compile_module("test-module".into(), program).unwrap();
    let ModuleExport::Function { entry } = artifact.exports.get("work").unwrap() else { panic!("missing exported function") };
    let mut vm = Vm::default();
    assert_eq!(vm.run_from(&artifact.code, *entry, true).unwrap(), ["101", "100"]);
}
#[test]
fn destructor_cannot_accept_arguments() {
    assert!(matches!(
        execute("struct Invalid { function Invalid(value: i32) end }"),
        Err(Error::Located { source, .. }) if matches!(&*source, Error::Type(message) if message == "destructor 'Invalid' must not accept arguments")
    ));
}
#[test]
fn failed_initializer_does_not_register_or_destroy_a_partial_struct() {
    let source = r#"
            struct PartialGuard {
                id: i32;
                function PartialGuard() print(id) end
            }
            if 1 == 1 then
                let guard: PartialGuard = PartialGuard { id = 1 / 0 }
            end
        "#;
    let mut vm = Vm::default();
    assert!(matches!(vm.execute(source), Err(Error::Runtime(message)) if message.contains("division by zero")));
    assert!(vm.output.is_empty());
    assert!(vm.active_destructors.is_empty());
}
#[test]
fn methods_are_isolated_and_can_explicitly_access_globals() {
    let source = r#"
            let counter: i32 = 7
            struct Meter { value: i32; function advance() end }
            function Meter::advance()
                let step: i32 = 1
                ::counter = ::counter + step
                value = value + ::counter
            end
            let meter: Meter = Meter { value = 0 }
            meter.advance()
            print(meter.value)
            print(counter)
        "#;
    assert_eq!(execute(source).unwrap(), ["8", "8"]);
    assert!(matches!(
        execute("let counter: i32 = 7; struct Meter { value: i32; function read() print(counter) end }"),
        Err(Error::Located { source, .. }) if matches!(&*source, Error::Type(message) if message == "unknown name 'counter'")
    ));
}
#[test]
fn len_supports_all_sized_runtime_values() {
    assert_eq!(
        execute("let text: string = \"ёж\"; let map: table<i32> = table { a = 1, b = 2 }; print(len(text)); print(len(map))").unwrap(),
        vec!["2".to_owned(), "2".to_owned()]
    );
}
#[test]
fn packed_scalars_are_little_endian() {
    let mut bytes = Vec::new();
    encode_scalar(&Value::I32(0x0102_0304), &Type::I32, &mut bytes).unwrap();
    assert_eq!(bytes, [4, 3, 2, 1]);
    assert_eq!(decode_scalar(&bytes, 0, &Type::I32).unwrap(), Value::I32(0x0102_0304));
}
#[test]
fn queued_input_is_consumed_before_stdin() {
    let mut vm = Vm::default();
    vm.push_input("42".to_owned());
    assert_eq!(vm.execute("let answer: i32 = input; print(answer)").unwrap(), vec!["42"]);
}
#[test]
fn public_input_helper_supports_strings_and_numbers() {
    assert_eq!(
        execute_with_input(
            "let label: string = input; let answer: i32 = input; print(label); print(answer)",
            ["ready", "42"],
        ).unwrap(),
        vec!["ready", "42"],
    );
}
#[test]
fn tables_accept_string_keys_and_support_removal_and_iteration() {
    let source = r#"
            let map: table<i32> = table { ["alpha"] = 2, [7] = 3 };
            let name: string = "alpha";
            map[name] = 5;
            print(len(map));
            print(remove(map, name));
            print(len(map));
            let table_keys: table_keys = keys(map);
            for i = 0, len(table_keys) - 1 do
                let key: table_key = table_keys[i];
                print(map[key]);
            end
        "#;
    assert_eq!(execute(source).unwrap(), vec!["2", "true", "1", "3"]);
}
#[test]
fn printf_validates_arguments_and_brace_escapes() {
    assert_eq!(execute("printf(\"{{{}}}\", 7)").unwrap(), vec!["{7}"]);
    assert!(matches!(execute("printf(\"{}\", 7, 8)"), Err(Error::Runtime(message)) if message == "printf has more arguments than placeholders"));
    assert!(matches!(execute("printf(\"{\", 7)"), Err(Error::Runtime(message)) if message.contains("invalid '{' escape")));
}
#[test]
fn putc_rejects_invalid_unicode_scalars() {
    assert!(matches!(execute("putc(1114112)"), Err(Error::Runtime(message)) if message.contains("valid Unicode scalar")));
}
#[test]
fn zero_tensor_bytes_need_no_scalar_encoding() {
    let mut vm = Vm::default();
    assert_eq!(vm.make_tensor_bytes(TensorInit::Zeros, &Type::I32, &[2, 3]).unwrap(), vec![0; 24]);
}

#[test]
fn stack_value_stays_within_three_machine_words() {
    // This protects the VM stack from a future metadata-bearing variant
    // accidentally turning every stack slot into a cache-line fragment.
    assert!(std::mem::size_of::<Value>() <= 3 * std::mem::size_of::<usize>());
}
#[test]
fn tensor_value_keeps_its_complete_static_type() {
    let ty = Rc::new(Type::Tensor(Box::new(Type::F32), 2));
    let value = Value::Tensor(HeapRef(0), ty);
    assert_eq!(value.ty(), Type::Tensor(Box::new(Type::F32), 2));
}
#[test]
fn garbage_collection_tracks_payload_bytes() {
    let mut heap = Heap::default();
    heap.allocate(HeapObject::String("payload".repeat(128)));
    assert!(heap.allocated_bytes > 0);
    assert_eq!(heap.collect(Vec::new()), 1);
    assert_eq!(heap.allocated_bytes, 0);
}
#[test]
fn garbage_collection_traverses_references_without_cloning_values() {
    let mut heap = Heap::default();
    let child = heap.allocate(HeapObject::String("reachable".into()));
    let parent = heap.allocate(HeapObject::Table {
        entries: HashMap::from([(TableKey::Name(Rc::from("child")), Value::String(child))]),
        element: Type::String,
    });
    assert_eq!(heap.collect([parent]), 0);
    assert!(matches!(heap.get(child), Ok(HeapObject::String(text)) if text == "reachable"));
}
#[test]
fn f16_conversion_uses_standard_binary16_encoding() {
    assert_eq!(f32_to_f16(1.0), 0x3c00);
    assert_eq!(f16_to_f32(0xc000), -2.0);
}
#[test]
fn builtins_are_resolved_during_compilation() {
    assert_eq!(execute("let value: f32 = 4.0; print(sqrt(value)); print(min(value, 3.0))").unwrap(), vec!["2", "3"]);
}
#[test]
fn lexer_and_core_ast_names_share_string_ids() {
    let lexed = lex("let repeated: i32 = 1; print(repeated)").unwrap();
    let ids: Vec<_> = lexed.tokens.iter().filter_map(|token| match token.kind {
        Token::Ident(id) => Some(id), _ => None,
    }).collect();
    assert_eq!(ids[0], ids[2]);
    assert_eq!(lexed.strings.resolve(ids[0]), "repeated");
}
#[test]
fn executable_bytecode_is_a_flat_u32_stream() {
    let (program, strings) = Parser::new(lex("let answer: i32 = 42; print(answer)").unwrap()).into_program().unwrap();
    let code = Compiler::default().with_strings(strings).compile(program).unwrap();
    assert!(code.words.iter().all(|word| *word <= u32::MAX));
    assert_eq!(code.words.len() % 4, 0);
    assert_eq!(Vm::default().run(&code).unwrap(), ["42"]);
}
#[test]
fn concat_allocates_exact_payload_capacity() {
    let mut vm = Vm::default();
    assert_eq!(vm.execute("let left: string = \"left\"; let right: string = \"right\"; let joined: string = left + right; print(joined)").unwrap(), ["leftright"]);
    let Value::String(reference) = vm.locals[2] else { panic!("joined local is not a string") };
    let HeapObject::String(joined) = vm.heap_ref().get(reference).unwrap() else { panic!("joined value is not a heap string") };
    assert_eq!(joined.capacity(), joined.len());
}
#[test]
fn numeric_casts_work_correctly() {
    let source = "let a: i8 = 10; let b: i32 = a as i32; print(b)";
    assert_eq!(execute(source).unwrap(), vec!["10".to_owned()]);
    let source_float = "let f: f32 = 42.5; let i: i32 = f as i32; print(i)";
    assert_eq!(execute(source_float).unwrap(), vec!["42".to_owned()]);
}
#[test]
fn elseif_chain_evaluates_correctly() {
    let source = "
            let x: i32 = 2
            if x == 1 then
                print(10)
            elseif x == 2 then
                print(20)
            elseif x == 3 then
                print(30)
            else
                print(40)
            end
        ";
    assert_eq!(execute(source).unwrap(), vec!["20".to_owned()]);
}
#[test]
fn cast_various_numeric_types_and_chains() {
    let source = r#"
            -- знаковое в беззнаковое с циклическим переносом (-1 -> 255)
            let neg: i32 = -1;
            let u: u8 = neg as u8;
            print(u);
            -- усечение старших битов (300 % 256 = 44)
            let big: i32 = 300;
            let wrapped: u8 = big as u8;
            print(wrapped);
            -- усечение дробной части к нулю (-3.7 -> -3)
            let f_neg: f64 = -3.7;
            let i_trunc: i32 = f_neg as i32;
            print(i_trunc);
            -- конвертация с участием f16 (binary16)
            let orig_f32: f32 = 2.5;
            let half: f16 = orig_f32 as f16;
            let back_f64: f64 = half as f64;
            print(back_f64);
            -- цепочки приведений
            let chained: i32 = 123.75 as f32 as i32;
            print(chained);
            -- приведение результата выражения в скобках
            let expr_cast: f64 = (10 + 5) as f64 / 2.0;
            print(expr_cast);
        "#;
    assert_eq!(
        execute(source).unwrap(),
        vec!["255", "44", "-3", "2.5", "123", "7.5"]
    );
}
#[test]
fn cast_rejects_non_numeric_types() {
    // bool нельзя приводить к числовым типам (используем 1 == 1 вместо true)
    assert!(matches!(
        execute("let b: bool = 1 == 1; let i: i32 = b as i32"),
        Err(Error::Located { source, .. }) if matches!(*source, Error::Type(ref msg) if msg.contains("cannot cast bool to i32"))
    ));
    // string нельзя приводить через cast
    assert!(matches!(
        execute("let s: string = \"hello\"; let i: i32 = s as i32"),
        Err(Error::Located { source, .. }) if matches!(*source, Error::Type(ref msg) if msg.contains("cannot cast string to i32"))
    ));
    // числа нельзя приводить к bool
    assert!(matches!(
        execute("let i: i32 = 42; let b: bool = i as bool"),
        Err(Error::Located { source, .. }) if matches!(*source, Error::Type(ref msg) if msg.contains("cannot cast i32 to bool"))
    ));
}
#[test]
fn garbage_collection_allocates_and_frees_999999_chunks() {
    let mut heap = Heap::default();
    let mut refs = Vec::new();
    for i in 0..999999 {
        let k:  i32 = i + 979999999 % 1333;
        let j:   i64 = (i as i64) * (k as i64);
        let fv: f32 = j  as f32;
        refs.push(heap.allocate(HeapObject::String(format!("mem-Chunk {i}-{k}-{j}-{fv}"))));
    }
    assert_eq!(heap.slots.len(), 999999);
    let roots = vec![refs[0], refs[999998]];
    let reclaimed = heap.collect(roots);
    assert_eq!(reclaimed, 999997);
    assert_eq!(heap.slots.len(), 999999);
    assert!(heap.free_head.is_some());
    let reused_ref = heap.allocate(HeapObject::String("reused".into()));
    assert!(reused_ref.0 > 0 && reused_ref.0 < 999998);
    assert_eq!(heap.slots.len(), 999999);
    let reclaimed_final = heap.collect(Vec::new());
    assert_eq!(reclaimed_final, 3);
    assert_eq!(heap.slots.len(), 0);
    assert_eq!(heap.free_head, None);
    assert_eq!(heap.allocated_bytes, 0);
}
#[test]
fn dynamic_types_binding_and_len() {
    let source = r#"
            -- Проверка привязки вектора к dArray и вычисления длины
            let vec: vector<i32> = [10, 20, 30, 40]
            let d_vec: dArray = vec
            print(len(d_vec))

            -- Проверка привязки тензора к dTensor и вычисления длины
            let t: tensor<f32, 2> = zeros<f32>([2, 3])
            let d_t: dTensor = t
            print(len(d_t))
        "#;
    assert_eq!(
        execute(source).unwrap(),
        vec!["4".to_owned(), "6".to_owned()]
    );
}
#[test]
fn dynamic_types_reject_invalid_assignments() {
    // Скалярные значения не должны приводиться к dArray
    assert!(matches!(
        execute("let num: i32 = 42; let d: dArray = num"),
        Err(Error::Located { source, .. }) if matches!(*source, Error::Type(ref msg) if msg.contains("expression has type i32"))
    ));
    // Векторы не должны приводиться к dTensor
    assert!(matches!(
        execute("let v: vector<i32> = [1]; let t: dTensor = v"),
        Err(Error::Located { source, .. }) if matches!(*source, Error::Type(ref msg) if msg.contains("expression has type vector<i32>"))
    ));
}
#[test]
fn dynamic_types_ffi_passing() {
    let mut vm = Vm::default();
    // Регистрируем внешнюю функцию, которая ожидает динамический массив
    vm.register_rust_function(
        "is_dynamic_array",
        vec![Type::DArray],
        Type::Bool,
        |args, _| {
            // На уровне Rust проверяем, что в рантайме действительно пришел Array
            match args[0] {
                Value::Array(_, _) => Ok(Value::Bool(true)),
                _ => Ok(Value::Bool(false)),
            }
        }
    ).unwrap();
    // Регистрируем функцию, ожидающую динамический тензор
    vm.register_rust_function(
        "is_dynamic_tensor",
        vec![Type::DTensor],
        Type::Bool,
        |args, _| {
            match args[0] {
                Value::Tensor(_, _) => Ok(Value::Bool(true)),
                _ => Ok(Value::Bool(false)),
            }
        }
    ).unwrap();
    let source = r#"
            let vec: vector<f64> = [1.0, 2.0]
            let t: tensor<u8, 1> = zeros<u8>([5])
            -- Передаем строгие типы во внешнюю функцию без явного приведения
            print(is_dynamic_array(vec))
            print(is_dynamic_tensor(t))
        "#;
    assert_eq!(
        vm.execute(source).unwrap(),
        vec!["true".to_owned(), "true".to_owned()]
    );
}
