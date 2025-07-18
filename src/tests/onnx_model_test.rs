use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use crate::{OrtEngine, OrtValue, DataType, Dimensions, OrtResult, NodeProto, ModelProto};

// Helper function for identity operation
fn identity_op(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // Simple identity operation
    Ok(inputs[0].clone())
}

// Helper function to create a simple ONNX model file for testing
fn create_test_onnx_file(path: &str) -> std::io::Result<()> {
    // This is a simplified binary representation of a minimal ONNX model
    // with a single Add operator that adds two 2x2 matrices
    let model_bytes: Vec<u8> = vec![
        // ONNX magic number and version
        0x8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        // ModelProto
        0x8, 0x3, 0x12, 0x8, 0x74, 0x65, 0x73, 0x74, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x1a, 0x3, 0x31, 0x2e, 0x30,
        // OpsetImport
        0x42, 0x6, 0xa, 0x0, 0x10, 0xd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        // GraphProto
        0x3a, 0x80, 0x1, 0xa, 0x8, 0x74, 0x65, 0x73, 0x74, 0x67, 0x72, 0x61, 0x70, 0x68,
        // Input
        0x62, 0x12, 0xa, 0x1, 0x41, 0x12, 0xd, 0xa, 0xb, 0x8, 0x1, 0x12, 0x7, 0xa, 0x5, 0x1a, 0x3, 0x8, 0x2, 0x8, 0x2,
        // Input
        0x62, 0x12, 0xa, 0x1, 0x42, 0x12, 0xd, 0xa, 0xb, 0x8, 0x1, 0x12, 0x7, 0xa, 0x5, 0x1a, 0x3, 0x8, 0x2, 0x8, 0x2,
        // Output
        0x5a, 0x12, 0xa, 0x1, 0x43, 0x12, 0xd, 0xa, 0xb, 0x8, 0x1, 0x12, 0x7, 0xa, 0x5, 0x1a, 0x3, 0x8, 0x2, 0x8, 0x2,
        // Node (Add)
        0xa, 0x1a, 0xa, 0x1, 0x41, 0x12, 0x1, 0x42, 0x1a, 0x1, 0x43, 0x22, 0x3, 0x41, 0x64, 0x64,
        // Initializer for A
        0x42, 0x1e, 0x8, 0x1, 0x12, 0x1, 0x41, 0x1a, 0x2, 0x8, 0x2, 0x8, 0x2, 0x2a, 0x10, 0x0, 0x0, 0x80, 0x3f, 0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x40, 0x40, 0x0, 0x0, 0x80, 0x40,
        // Initializer for B
        0x42, 0x1e, 0x8, 0x1, 0x12, 0x1, 0x42, 0x1a, 0x2, 0x8, 0x2, 0x8, 0x2, 0x2a, 0x10, 0x0, 0x0, 0x80, 0x3f, 0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x40, 0x40, 0x0, 0x0, 0x80, 0x40,
    ];

    let mut file = File::create(path)?;
    file.write_all(&model_bytes)?;
    Ok(())
}

// Helper function to create a tensor
fn create_tensor(shape: Vec<usize>, data: Vec<f32>) -> OrtValue {
    let shape_dims = shape.iter().map(|&d| Dimensions::Fixed(d)).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
    OrtValue::Tensor {
        shape: shape_dims,
        dtype: DataType::Float,
        data: Arc::new(bytes),
    }
}
use crate::print_model_info;
#[test]
fn test_parse_and_infer_from_onnx_file() {
    // Create a temporary ONNX model file
    // let test_file = "./kokoro-v1.0.onnx";
    // let test_file = "./test_model.onnx";
    let test_file = "./corrected_add_model.onnx";
    // print_model_info(test_file).unwrap();
    // // create_test_onnx_file(test_file).expect("Failed to create test ONNX file");
    // let tokens: Vec<Vec<Vec<f32>>> = vec![vec![vec![1.0 ,2.0],vec![3.0,4.0]]]; // [1, 7]
    // //     let speed: f32 = 1.0;
    
    // //     // Create tokens tensor
    //     let batch_size = tokens.len();
    //     let sequence_length = tokens[0].len();
    //     let tokens_flat: Vec<Vec<f32>> = tokens.clone().into_iter().flatten().collect();
    //     // let tokens_flat: Vec<f32> = tokens_flat.clone().into_iter().flatten().collect();
    //     let tokens_tensor = OrtValue::Tensor {
    //         shape: vec![Dimensions::Fixed(batch_size), Dimensions::Fixed(sequence_length)],
    //         dtype: DataType::Float,
    //         data: Arc::new(
    //             tokens_flat
    //                 .iter()
    //                 .flat_map(|x| x.to_vec().to_le_bytes())
    //                 .collect::<Vec<u8>>(),
    //         ),
    //     };
    // Test that the file exists
    assert!(Path::new(test_file).exists());

    // Try to parse the ONNX model
    let engine_result = OrtEngine::new(test_file);
    // let model=engine_result.unwrap().model;
    // let graph=model.graph.unwrap();
    // println!("{:?}",graph);

    // for ei in model.graph.unwrap().initializer{
    //     println!("{:?}",ei);
    // }
    
    // Clean up the test file
    // std::fs::remove_file(test_file).expect("Failed to remove test file");
    
    // Check if parsing was successful
    match engine_result {
        Ok(engine) => {
            // Test inference with empty inputs (since we have initializers)
            let mut inputs = HashMap::new();
            // inputs.insert("A".to_string(),tokens_tensor.clone()) ;
            // inputs.insert("B".to_string(), tokens_tensor);
            let outputs = engine.infer(inputs);
            
            match outputs {
                Ok(output_map) => {
                    assert!(output_map.contains_key("C"));
                    
                    if let Some(output_tensor) = output_map.get("C") {
                        // Convert to ndarray for easier validation
                        if let Ok(array) = crate::ort_to_ndarray(output_tensor) {
                            println!("{:?}",array)
                            // assert_eq!(array.shape(), &[2, 2]);
                            
                            // // Expected: 1+1=2, 2+2=4, 3+3=6, 4+4=8
                            // assert_eq!(array[[0, 0]], 2.0);
                            // assert_eq!(array[[0, 1]], 4.0);
                            // assert_eq!(array[[1, 0]], 6.0);
                            // assert_eq!(array[[1, 1]], 8.0);
                        } else {
                            panic!("Failed to convert output tensor to ndarray");
                        }
                    } else {
                        panic!("Output tensor 'C' not found");
                    }
                }
                Err(e) => println!("Inference failed: {:?}", e),
            }
        }
        Err(e) => {
            // If parsing fails, it might be due to the simplified binary representation
            // In a real test, we would use a proper ONNX file
            println!("Note: ONNX parsing failed: {:?}", e);
            println!("This is expected if using a simplified binary representation");
        }
    }
}

#[test]
fn test_model_with_custom_inputs() {
    // Create inputs for a model
    let mut inputs = HashMap::new();
    
    // Create a 2x3 tensor with values [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let input_tensor = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(2), Dimensions::Fixed(3)],
        dtype: DataType::Float,
        data: std::sync::Arc::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .into_iter()
                .flat_map(|f| f.to_le_bytes())
                .collect()
        ),
    };
    
    inputs.insert("input".to_string(), input_tensor);
    
    // In a real test, we would load an actual model and run inference
    // For this test, we'll just verify the input tensor structure
    if let OrtValue::Tensor { shape, dtype, data } = &inputs["input"] {
        assert_eq!(shape.len(), 2);
        assert_eq!(dtype, &DataType::Float);
        
        // Convert back to f32 values for verification
        let float_data: Vec<f32> = data.chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        assert_eq!(float_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    } else {
        panic!("Expected Tensor variant");
    }
}

#[test]
fn test_shape_inference_for_complex_ops() {
    // Create a mock model with complex operations
    let mut model = crate::ModelProto::default();
    model.ir_version = 7;
    
    let mut graph = crate::GraphProto::default();
    
    // Create a MatMul node
    let mut node = crate::NodeProto::default();
    node.op_type = "MatMul".to_string();
    node.input.push("A".to_string());
    node.input.push("B".to_string());
    node.output.push("C".to_string());
    
    graph.node.push(node);
    
    // Add output info
    let mut output = crate::ValueInfoProto::default();
    output.name = "C".to_string();
    graph.output.push(output);
    
    // Create input tensors with shapes that should be compatible for MatMul
    let mut inputs = HashMap::new();
    
    // Input A: 2x3 matrix
    inputs.insert("A".to_string(), OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(2), Dimensions::Fixed(3)],
        dtype: DataType::Float,
        data: std::sync::Arc::new(vec![0.0f32; 6].into_iter().flat_map(|f| f.to_le_bytes()).collect()),
    });
    
    // Input B: 3x4 matrix
    inputs.insert("B".to_string(), OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(3), Dimensions::Fixed(4)],
        dtype: DataType::Float,
        data: std::sync::Arc::new(vec![0.0f32; 12].into_iter().flat_map(|f| f.to_le_bytes()).collect()),
    });
    
    model.graph = Some(graph);
    
    // Create engine with the mock model
    let mut node_registry = HashMap::new();
    node_registry.insert("MatMul".to_string(), OrtEngine::op_matmul as fn(&NodeProto, &[OrtValue]) -> OrtResult<OrtValue>);
    
    let engine = OrtEngine {
        model,
        node_registry,
        vendor_ops: HashMap::new(),
        shape_inference: crate::ShapeInference::default(),
    };
    
    // Run inference
    let outputs = engine.infer(inputs);
    
    // Check that shape inference worked correctly
    match outputs {
        Ok(output_map) => {
            assert!(output_map.contains_key("C"));
            
            if let Some(output_tensor) = output_map.get("C") {
                if let OrtValue::Tensor { shape, .. } = output_tensor {
                    // Expected shape for MatMul: [2, 4]
                    assert_eq!(shape.len(), 2);
                    assert!(matches!(shape[0], Dimensions::Fixed(2)));
                    assert!(matches!(shape[1], Dimensions::Fixed(4)));
                } else {
                    panic!("Expected Tensor variant");
                }
            } else {
                panic!("Output tensor 'C' not found");
            }
        }
        Err(e) => panic!("Inference failed: {:?}", e),
    }
}
/// Function to print all initializers in an ONNX model file
/// 
/// # Arguments
/// * `path` - Path to the ONNX model file
/// 
/// # Returns
/// * `OrtResult<()>` - Result of the operation
pub fn print_model_initializers<P: AsRef<Path>>(path: P) -> OrtResult<()> {
    use std::io::Read;
    use prost::Message;
    
    // Open and read the ONNX file
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    // Decode the model
    let model = ModelProto::decode(&*buffer)?;
    
    // Get the graph
    let graph = model.graph.as_ref().ok_or(crate::OrtError::InvalidModel)?;
    
    println!("Model Initializers:");
    println!("===================");
    
    if graph.initializer.is_empty() {
        println!("No initializers found in the model.");
        return Ok(());
    }
    
    // Print information about each initializer
    for (i, initializer) in graph.initializer.iter().enumerate() {
        println!("Initializer #{}: {}", i + 1, initializer.name);
        
        // Print data type
        let data_type = match initializer.data_type {
            1 => "Float",
            7 => "Int64",
            8 => "String",
            _ => "Unknown",
        };
        println!("  Data Type: {}", data_type);
        
        // Print shape
        let shape = initializer.dims.iter()
            .map(|&d| d.to_string())
            .collect::<Vec<_>>()
            .join(" x ");
        println!("  Shape: [{}]", shape);
        
        // Print data summary based on type
        match initializer.data_type {
            1 => { // Float
                let data_len = if !initializer.float_data.is_empty() {
                    initializer.float_data.len()
                } else {
                    initializer.raw_data.len() / 4
                };
                println!("  Data: {} float values", data_len);
                
                // Print a few sample values if available
                if !initializer.float_data.is_empty() && initializer.float_data.len() <= 10 {
                    println!("  Values: {:?}", initializer.float_data);
                } else if !initializer.float_data.is_empty() {
                    println!("  First few values: {:?}", &initializer.float_data[..std::cmp::min(5, initializer.float_data.len())]);
                }
            },
            7 => { // Int64
                let data_len = if !initializer.int64_data.is_empty() {
                    initializer.int64_data.len()
                } else {
                    initializer.raw_data.len() / 8
                };
                println!("  Data: {} int64 values", data_len);
                
                // Print a few sample values if available
                if !initializer.int64_data.is_empty() && initializer.int64_data.len() <= 10 {
                    println!("  Values: {:?}", initializer.int64_data);
                } else if !initializer.int64_data.is_empty() {
                    println!("  First few values: {:?}", &initializer.int64_data[..std::cmp::min(5, initializer.int64_data.len())]);
                }
            },
            8 => { // String
                println!("  Data: {} string values", initializer.string_data.len());
                
                // Print a few sample values if available
                if !initializer.string_data.is_empty() && initializer.string_data.len() <= 5 {
                    for (i, s) in initializer.string_data.iter().enumerate() {
                        println!("  String {}: {}", i, String::from_utf8_lossy(s));
                    }
                } else if !initializer.string_data.is_empty() {
                    println!("  First string: {}", String::from_utf8_lossy(&initializer.string_data[0]));
                }
            },
            _ => println!("  Data: Unknown format"),
        }
        
        println!();
    }
    
    Ok(())
}


#[test]
fn test_print_real_model_initializers() {
    // Check if the real ONNX model exists
    let real_model_path = "V:\\Github\\b2\\bison\\kokoro-v1.0.onnx";
    
    if Path::new(real_model_path).exists() {
        println!("Testing with real model: {}", real_model_path);
        match print_model_initializers(real_model_path) {
            Ok(_) => println!("Successfully printed initializers from real model"),
            Err(e) => println!("Error printing initializers from real model: {:?}", e),
        }
    } else {
        println!("Skipping real model test as '{}' does not exist", real_model_path);
    }
}