use std::collections::HashMap;
use std::sync::Arc;
use crate::{OrtEngine, OrtValue, DataType, Dimensions, ModelProto, GraphProto, NodeProto, ValueInfoProto, TensorProto, TypeProto, TensorTypeProto, TensorShapeProto, TensorDimension, AttributeProto, OrtResult, OrtError};

// Helper function for identity operation
fn identity_op(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // Simple identity operation
    Ok(inputs[0].clone())
}

// Helper function to create a simple model proto for testing
fn create_test_model() -> ModelProto {
    let mut model = ModelProto::default();
    model.ir_version = 7;
    model.producer_name = "Test".to_string();
    model.producer_version = "1.0".to_string();
    
    let mut graph = GraphProto::default();
    graph.name = "test_graph".to_string();
    
    // Create input
    let mut input = ValueInfoProto::default();
    input.name = "input".to_string();
    let mut input_type = TypeProto::default();
    let mut tensor_type = TensorTypeProto::default();
    tensor_type.elem_type = 1; // FLOAT
    let mut shape = TensorShapeProto::default();
    let mut dim1 = TensorDimension::default();
    dim1.dim_value = 1;
    let mut dim2 = TensorDimension::default();
    dim2.dim_value = 3;
    shape.dim = vec![dim1, dim2];
    tensor_type.shape = Some(shape);
    input_type.tensor_type = Some(tensor_type);
    input.type_proto = Some(input_type);
    graph.input.push(input);
    
    // Create output
    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    let mut output_type = TypeProto::default();
    let mut tensor_type = TensorTypeProto::default();
    tensor_type.elem_type = 1; // FLOAT
    let mut shape = TensorShapeProto::default();
    let mut dim1 = TensorDimension::default();
    dim1.dim_value = 1;
    let mut dim2 = TensorDimension::default();
    dim2.dim_value = 3;
    shape.dim = vec![dim1, dim2];
    tensor_type.shape = Some(shape);
    output_type.tensor_type = Some(tensor_type);
    output.type_proto = Some(output_type);
    graph.output.push(output);
    
    // Create a simple node (Identity)
    let mut node = NodeProto::default();
    node.op_type = "Identity".to_string();
    node.input.push("input".to_string());
    node.output.push("output".to_string());
    graph.node.push(node);
    
    model.graph = Some(graph);
    model
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

// Helper function to create a tensor proto
fn create_tensor_proto(name: &str, dims: Vec<i64>, data: Vec<f32>) -> TensorProto {
    let mut tensor = TensorProto::default();
    tensor.name = name.to_string();
    tensor.data_type = 1; // FLOAT
    tensor.dims = dims;
    tensor.float_data = data;
    tensor
}

#[test]
fn test_parse_tensor() {
    let tensor_proto = create_tensor_proto("test_tensor", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    
    let model = create_test_model();
    let engine = OrtEngine {
        model,
        node_registry: HashMap::new(),
        vendor_ops: HashMap::new(),
        shape_inference: crate::ShapeInference::default(),
    };
    
    let result = engine.parse_tensor(&tensor_proto).unwrap();
    
    if let OrtValue::Tensor { shape, dtype, .. } = &result {
        assert_eq!(shape.len(), 2);
        assert_eq!(dtype, &DataType::Float);
        
        // Convert to ndarray for easier verification
        let array = crate::ort_to_ndarray(&result).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array[[0, 0]], 1.0);
        assert_eq!(array[[0, 1]], 2.0);
        assert_eq!(array[[0, 2]], 3.0);
        assert_eq!(array[[1, 0]], 4.0);
        assert_eq!(array[[1, 1]], 5.0);
        assert_eq!(array[[1, 2]], 6.0);
    } else {
        panic!("Expected Tensor variant");
    }
}

#[test]
fn test_model_proto_message() {
    let model = create_test_model();
    
    assert_eq!(model.ir_version, 7);
    assert_eq!(model.producer_name, "Test");
    assert_eq!(model.producer_version, "1.0");
    
    let graph = model.graph.as_ref().unwrap();
    assert_eq!(graph.name, "test_graph");
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Identity");
    assert_eq!(graph.input.len(), 1);
    assert_eq!(graph.input[0].name, "input");
    assert_eq!(graph.output.len(), 1);
    assert_eq!(graph.output[0].name, "output");
}

#[test]
fn test_shape_inference() {
    let model = create_test_model();
    let graph = model.graph.as_ref().unwrap();
    
    let mut shape_inference = crate::ShapeInference::new(graph);
    
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), create_tensor(vec![1, 3], vec![1.0, 2.0, 3.0]));
    
    shape_inference.infer_shapes(&inputs).unwrap();
    
    // Check that the output shape is correctly inferred
    assert!(shape_inference.shapes.contains_key("output"));
    let output_shape = shape_inference.shapes.get("output").unwrap();
    assert_eq!(output_shape.len(), 2);
    assert!(matches!(output_shape[0], Dimensions::Fixed(1)));
    assert!(matches!(output_shape[1], Dimensions::Fixed(3)));
}

#[test]
fn test_infer_with_identity_op() {
    let model = create_test_model();
    
    let mut node_registry = HashMap::new();
    node_registry.insert("Identity".to_string(), identity_op as fn(&NodeProto, &[OrtValue]) -> OrtResult<OrtValue>);
    
    let engine = OrtEngine {
        model,
        node_registry,
        vendor_ops: HashMap::new(),
        shape_inference: crate::ShapeInference::default(),
    };
    
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), create_tensor(vec![1, 3], vec![1.0, 2.0, 3.0]));
    
    let outputs = engine.infer(inputs).unwrap();
    
    assert!(outputs.contains_key("output"));
    let output = outputs.get("output").unwrap();
    
    let array = crate::ort_to_ndarray(output).unwrap();
    assert_eq!(array.shape(), &[1, 3]);
    assert_eq!(array[[0, 0]], 1.0);
    assert_eq!(array[[0, 1]], 2.0);
    assert_eq!(array[[0, 2]], 3.0);
}

#[test]
fn test_infer_with_add_op() {
    let mut model = create_test_model();
    let graph = model.graph.as_mut().unwrap();
    
    // Change the op_type to Add
    graph.node[0].op_type = "Add".to_string();
    
    // Add a second input
    let mut input2 = ValueInfoProto::default();
    input2.name = "input2".to_string();
    let mut input_type = TypeProto::default();
    let mut tensor_type = TensorTypeProto::default();
    tensor_type.elem_type = 1; // FLOAT
    let mut shape = TensorShapeProto::default();
    let mut dim1 = TensorDimension::default();
    dim1.dim_value = 1;
    let mut dim2 = TensorDimension::default();
    dim2.dim_value = 3;
    shape.dim = vec![dim1, dim2];
    tensor_type.shape = Some(shape);
    input_type.tensor_type = Some(tensor_type);
    input2.type_proto = Some(input_type);
    graph.input.push(input2);
    
    // Update the node to use both inputs
    graph.node[0].input.push("input2".to_string());
    
    let mut node_registry = HashMap::new();
    node_registry.insert("Add".to_string(), OrtEngine::op_add as fn(&NodeProto, &[OrtValue]) -> OrtResult<OrtValue>);
    
    let engine = OrtEngine {
        model,
        node_registry,
        vendor_ops: HashMap::new(),
        shape_inference: crate::ShapeInference::default(),
    };
    
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), create_tensor(vec![1, 3], vec![1.0, 2.0, 3.0]));
    inputs.insert("input2".to_string(), create_tensor(vec![1, 3], vec![4.0, 5.0, 6.0]));
    
    let outputs = engine.infer(inputs).unwrap();
    
    assert!(outputs.contains_key("output"));
    let output = outputs.get("output").unwrap();
    
    let array = crate::ort_to_ndarray(output).unwrap();
    assert_eq!(array.shape(), &[1, 3]);
    assert_eq!(array[[0, 0]], 5.0); // 1 + 4
    assert_eq!(array[[0, 1]], 7.0); // 2 + 5
    assert_eq!(array[[0, 2]], 9.0); // 3 + 6
}

#[test]
fn test_infer_with_matmul_op() {
    let mut model = create_test_model();
    let graph = model.graph.as_mut().unwrap();
    
    // Change the op_type to MatMul
    graph.node[0].op_type = "MatMul".to_string();
    
    // Update input shape to [1, 2]
    if let Some(type_proto) = &mut graph.input[0].type_proto {
        if let Some(tensor_type) = &mut type_proto.tensor_type {
            if let Some(shape) = &mut tensor_type.shape {
                shape.dim[1].dim_value = 2;
            }
        }
    }
    
    // Add a second input with shape [2, 3]
    let mut input2 = ValueInfoProto::default();
    input2.name = "input2".to_string();
    let mut input_type = TypeProto::default();
    let mut tensor_type = TensorTypeProto::default();
    tensor_type.elem_type = 1; // FLOAT
    let mut shape = TensorShapeProto::default();
    let mut dim1 = TensorDimension::default();
    dim1.dim_value = 2;
    let mut dim2 = TensorDimension::default();
    dim2.dim_value = 3;
    shape.dim = vec![dim1, dim2];
    tensor_type.shape = Some(shape);
    input_type.tensor_type = Some(tensor_type);
    input2.type_proto = Some(input_type);
    graph.input.push(input2);
    
    // Update the node to use both inputs
    graph.node[0].input.push("input2".to_string());
    
    // Update output shape to [1, 3]
    if let Some(type_proto) = &mut graph.output[0].type_proto {
        if let Some(tensor_type) = &mut type_proto.tensor_type {
            if let Some(shape) = &mut tensor_type.shape {
                shape.dim[1].dim_value = 3;
            }
        }
    }
    
    let mut node_registry = HashMap::new();
    node_registry.insert("MatMul".to_string(), OrtEngine::op_matmul as fn(&NodeProto, &[OrtValue]) -> OrtResult<OrtValue>);
    
    let engine = OrtEngine {
        model,
        node_registry,
        vendor_ops: HashMap::new(),
        shape_inference: crate::ShapeInference::default(),
    };
    
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), create_tensor(vec![1, 2], vec![1.0, 2.0]));
    inputs.insert("input2".to_string(), create_tensor(vec![2, 3], vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
    
    let outputs = engine.infer(inputs).unwrap();
    
    assert!(outputs.contains_key("output"));
    let output = outputs.get("output").unwrap();
    
    let array = crate::ort_to_ndarray(output).unwrap();
    assert_eq!(array.shape(), &[1, 3]);
    
    // [1, 2] · [3, 4, 5; 6, 7, 8] = [1*3 + 2*6, 1*4 + 2*7, 1*5 + 2*8] = [15, 18, 21]
    assert_eq!(array[[0, 0]], 15.0);
    assert_eq!(array[[0, 1]], 18.0);
    assert_eq!(array[[0, 2]], 21.0);
}

#[test]
fn test_symbolic_dimensions() {
    let mut model = create_test_model();
    let graph = model.graph.as_mut().unwrap();
    
    // Set a symbolic dimension (-1) in the input
    if let Some(type_proto) = &mut graph.input[0].type_proto {
        if let Some(tensor_type) = &mut type_proto.tensor_type {
            if let Some(shape) = &mut tensor_type.shape {
                shape.dim[0].dim_value = -1; // Symbolic dimension
            }
        }
    }
    
    let mut node_registry = HashMap::new();
    node_registry.insert("Identity".to_string(), identity_op as fn(&NodeProto, &[OrtValue]) -> OrtResult<OrtValue>);
    
    let engine = OrtEngine {
        model,
        node_registry,
        vendor_ops: HashMap::new(),
        shape_inference: crate::ShapeInference::default(),
    };
    
    // Create a tensor with batch size 2 (different from the symbolic -1)
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), create_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    
    let outputs = engine.infer(inputs).unwrap();
    
    assert!(outputs.contains_key("output"));
    let output = outputs.get("output").unwrap();
    
    let array = crate::ort_to_ndarray(output).unwrap();
    assert_eq!(array.shape(), &[2, 3]); // The symbolic dimension should adapt to the input
}

#[test]
fn test_error_handling_missing_input() {
    let model = create_test_model();
    
    let mut node_registry = HashMap::new();
    node_registry.insert("Identity".to_string(), identity_op as fn(&NodeProto, &[OrtValue]) -> OrtResult<OrtValue>);
    
    let engine = OrtEngine {
        model,
        node_registry,
        vendor_ops: HashMap::new(),
        shape_inference: crate::ShapeInference::default(),
    };
    
    // Empty inputs - should fail because "input" is missing
    let inputs = HashMap::new();
    let result = engine.infer(inputs);
    
    assert!(result.is_err());
    match result {
        Err(crate::OrtError::MissingInput(_)) => {
            // Expected error
        }
        _ => panic!("Expected MissingInput error"),
    }
}

#[test]
fn test_error_handling_unsupported_op() {
    let mut model = create_test_model();
    let graph = model.graph.as_mut().unwrap();
    
    // Change to an unsupported op type
    graph.node[0].op_type = "UnsupportedOp".to_string();
    
    let engine = OrtEngine {
        model,
        node_registry: HashMap::new(), // Empty registry
        vendor_ops: HashMap::new(),
        shape_inference: crate::ShapeInference::default(),
    };
    
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), create_tensor(vec![1, 3], vec![1.0, 2.0, 3.0]));
    
    let result = engine.infer(inputs);
    
    assert!(result.is_err());
    match result {
        Err(crate::OrtError::UnsupportedOp(_)) => {
            // Expected error
        }
        _ => panic!("Expected UnsupportedOp error"),
    }
}