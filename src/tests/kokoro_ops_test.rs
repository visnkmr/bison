use std::sync::Arc;
use ndarray::{Array, ArrayD};
use crate::{OrtEngine, OrtValue, DataType, Dimensions, NodeProto, AttributeProto, OrtResult};

// Helper function to create a tensor for testing
fn create_tensor(shape: Vec<usize>, data: Vec<f32>) -> OrtValue {
    let shape_dims = shape.iter().map(|&d| Dimensions::Fixed(d)).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
    OrtValue::Tensor {
        shape: shape_dims,
        dtype: DataType::Float,
        data: Arc::new(bytes),
    }
}

// Helper function to create an int64 tensor for testing
fn create_int64_tensor(shape: Vec<usize>, data: Vec<i64>) -> OrtValue {
    let shape_dims = shape.iter().map(|&d| Dimensions::Fixed(d)).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|&i| i.to_le_bytes()).collect();
    OrtValue::Tensor {
        shape: shape_dims,
        dtype: DataType::Int64,
        data: Arc::new(bytes),
    }
}

// Helper function to add an attribute to a node
fn add_attribute(node: &mut NodeProto, name: &str, value: i64) {
    let mut attr = AttributeProto::default();
    attr.name = name.to_string();
    attr.i = value;
    node.attributes.push(attr);
}

// Helper function to add a float attribute to a node
fn add_float_attribute(node: &mut NodeProto, name: &str, value: f32) {
    let mut attr = AttributeProto::default();
    attr.name = name.to_string();
    attr.f = value;
    node.attributes.push(attr);
}

#[test]
fn test_op_embedding() {
    // Test the embedding operation
    let indices = create_int64_tensor(vec![2, 3], vec![
        1, 2, 3,
        4, 5, 0
    ]);
    
    // Create embedding weights
    let weights = create_tensor(vec![6, 4], vec![
        0.1, 0.2, 0.3, 0.4,  // embedding for index 0
        0.5, 0.6, 0.7, 0.8,  // embedding for index 1
        0.9, 1.0, 1.1, 1.2,  // embedding for index 2
        1.3, 1.4, 1.5, 1.6,  // embedding for index 3
        1.7, 1.8, 1.9, 2.0,  // embedding for index 4
        2.1, 2.2, 2.3, 2.4   // embedding for index 5
    ]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_embedding(&node, &[indices, weights]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[2, 3, 4]);
    
    // Check embeddings for first sequence
    // Index 1 -> [0.5, 0.6, 0.7, 0.8]
    assert_eq!(array[[0, 0, 0]], 0.5);
    assert_eq!(array[[0, 0, 1]], 0.6);
    assert_eq!(array[[0, 0, 2]], 0.7);
    assert_eq!(array[[0, 0, 3]], 0.8);
    
    // Index 2 -> [0.9, 1.0, 1.1, 1.2]
    assert_eq!(array[[0, 1, 0]], 0.9);
    assert_eq!(array[[0, 1, 1]], 1.0);
    assert_eq!(array[[0, 1, 2]], 1.1);
    assert_eq!(array[[0, 1, 3]], 1.2);
}

#[test]
fn test_op_constant_of_shape_int64() {
    // Test creating a constant tensor with a specified shape
    let shape = create_int64_tensor(vec![3], vec![2, 3, 4]);
    
    let mut node = NodeProto::default();
    
    // Create a tensor attribute for the value
    let mut tensor = crate::TensorProto::default();
    tensor.data_type = 7; // INT64
    tensor.int64_data = vec![1]; // Fill with 1s
    
    let mut attr = AttributeProto::default();
    attr.name = "value".to_string();
    attr.t = Some(tensor);
    node.attributes.push(attr);
    
    let result = OrtEngine::op_constant_of_shape_int64(&node, &[shape]).unwrap();
    
    if let OrtValue::Tensor { shape, dtype, data } = result {
        assert_eq!(shape.len(), 3);
        assert_eq!(dtype, DataType::Int64);
        
        // Check dimensions
        if let Dimensions::Fixed(dim) = shape[0] {
            assert_eq!(dim, 2);
        }
        if let Dimensions::Fixed(dim) = shape[1] {
            assert_eq!(dim, 3);
        }
        if let Dimensions::Fixed(dim) = shape[2] {
            assert_eq!(dim, 4);
        }
    }
}

#[test]
fn test_op_layer_normalization_with_epsilon() {
    // Test layer normalization with epsilon
    let input = create_tensor(vec![2, 3, 4], vec![
        // Batch 1
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
        // Batch 2
        1.3, 1.4, 1.5, 1.6,
        1.7, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4
    ]);
    
    let scale = create_tensor(vec![4], vec![1.0, 1.0, 1.0, 1.0]);
    let bias = create_tensor(vec![4], vec![0.0, 0.0, 0.0, 0.0]);
    
    let mut node = NodeProto::default();
    add_float_attribute(&mut node, "epsilon", 1e-5);
    
    let result = OrtEngine::op_layer_normalization_with_epsilon(&node, &[input, scale, bias]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[2, 3, 4]);
}

#[test]
fn test_op_expand() {
    // Test expanding a tensor to a larger shape
    let input = create_tensor(vec![2, 1], vec![1.0, 2.0]);
    let shape = create_int64_tensor(vec![3], vec![2, 3, 2]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_expand(&node, &[input, shape]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[2, 3, 2]);
}

#[test]
fn test_kokoro_bert_pipeline() {
    // Test a complete BERT pipeline for Kokoro
    
    // 1. Create token IDs
    let input_ids = create_int64_tensor(vec![2, 3], vec![
        101, 2054, 102, // [CLS] who [SEP]
        101, 2128, 102  // [CLS] how [SEP]
    ]);
    
    // 2. Create token type IDs
    let token_type_ids = create_int64_tensor(vec![2, 3], vec![
        0, 0, 0,
        0, 0, 0
    ]);
    
    // 3. Create attention mask
    let attention_mask = create_tensor(vec![2, 3], vec![
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0
    ]);
    
    // 4. Create word embeddings (simplified)
    let word_embeddings = create_tensor(vec![30522, 4], vec![
        // Only define embeddings for tokens we need
        0.01, 0.02, 0.03, 0.04, // 0 (padding)
        0.05, 0.06, 0.07, 0.08, // 101 ([CLS])
        0.09, 0.10, 0.11, 0.12, // 102 ([SEP])
        0.13, 0.14, 0.15, 0.16, // 103
        // ... (many zeros for tokens we don't use)
        // Fill with zeros up to token 2054
        0.0, 0.0, 0.0, 0.0,
        // Token 2054 (who)
        0.17, 0.18, 0.19, 0.20,
        // Fill with zeros up to token 2128
        0.0, 0.0, 0.0, 0.0,
        // Token 2128 (how)
        0.21, 0.22, 0.23, 0.24,
        // Fill the rest with zeros
        0.0, 0.0, 0.0, 0.0
    ]);
    
    // 5. Create position embeddings
    let position_embeddings = create_tensor(vec![512, 4], vec![
        // Only define embeddings for positions we need
        0.001, 0.002, 0.003, 0.004, // Position 0
        0.005, 0.006, 0.007, 0.008, // Position 1
        0.009, 0.010, 0.011, 0.012, // Position 2
        // Fill the rest with zeros
        0.0, 0.0, 0.0, 0.0
    ]);
    
    // 6. Create token type embeddings
    let token_type_embeddings = create_tensor(vec![2, 4], vec![
        0.0001, 0.0002, 0.0003, 0.0004, // Type 0
        0.0005, 0.0006, 0.0007, 0.0008  // Type 1
    ]);
    
    // 7. Apply embedding
    let embedding_node = NodeProto::default();
    let word_embeddings_result = OrtEngine::op_embedding(&embedding_node, &[input_ids, word_embeddings]).unwrap();
    
    // 8. Add position embeddings
    let position_node = NodeProto::default();
    let position_result = OrtEngine::op_position_embeddings(&position_node, &[word_embeddings_result, position_embeddings]).unwrap();
    
    // 9. Add token type embeddings
    let token_type_node = NodeProto::default();
    let embeddings_result = OrtEngine::op_token_type_embeddings(&token_type_node, &[position_result, token_type_ids, token_type_embeddings]).unwrap();
    
    // 10. Apply layer normalization
    let scale = create_tensor(vec![4], vec![1.0, 1.0, 1.0, 1.0]);
    let bias = create_tensor(vec![4], vec![0.0, 0.0, 0.0, 0.0]);
    
    let mut layer_norm_node = NodeProto::default();
    add_float_attribute(&mut layer_norm_node, "epsilon", 1e-12);
    
    let embeddings_output = OrtEngine::op_layer_normalization_with_epsilon(&layer_norm_node, &[embeddings_result, scale, bias]).unwrap();
    
    // Verify that the embeddings have the right shape
    let embeddings_array = crate::ort_to_ndarray(&embeddings_output).unwrap();
    assert_eq!(embeddings_array.shape(), &[2, 3, 4]);
}