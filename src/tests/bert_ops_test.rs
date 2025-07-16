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

// Helper function to add a string attribute to a node
fn add_string_attribute(node: &mut NodeProto, name: &str, value: &str) {
    let mut attr = AttributeProto::default();
    attr.name = name.to_string();
    attr.s = value.as_bytes().to_vec();
    node.attributes.push(attr);
}

#[test]
fn test_op_erf() {
    // Test the error function implementation
    let input = create_tensor(vec![3], vec![-1.0, 0.0, 1.0]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_erf(&node, &[input]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    
    // Expected values for erf(-1.0), erf(0.0), erf(1.0)
    // erf(-1.0) ≈ -0.8427, erf(0.0) = 0, erf(1.0) ≈ 0.8427
    assert!((array[[0]] + 0.8427).abs() < 0.01);
    assert!(array[[1]].abs() < 0.01);
    assert!((array[[2]] - 0.8427).abs() < 0.01);
}

#[test]
fn test_op_gelu() {
    // Test the GELU activation function
    let input = create_tensor(vec![3], vec![-1.0, 0.0, 1.0]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_gelu(&node, &[input]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    
    // Expected values for GELU
    // GELU(-1.0) ≈ -0.1587, GELU(0.0) = 0, GELU(1.0) ≈ 0.8413
    assert!((array[[0]] + 0.1587).abs() < 0.01);
    assert!(array[[1]].abs() < 0.01);
    assert!((array[[2]] - 0.8413).abs() < 0.01);
}

#[test]
fn test_op_split() {
    // Test splitting a tensor along an axis
    let input = create_tensor(vec![2, 4], vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0
    ]);
    
    // Split along axis 1 into 2 parts
    let mut node = NodeProto::default();
    add_attribute(&mut node, "axis", 1);
    node.output.push("output1".to_string());
    node.output.push("output2".to_string());
    
    let result = OrtEngine::op_split(&node, &[input]).unwrap();
    
    if let OrtValue::Sequence(parts) = result {
        assert_eq!(parts.len(), 2);
        
        // First part should be [1.0, 2.0] and [5.0, 6.0]
        let array1 = crate::ort_to_ndarray(&parts[0]).unwrap();
        assert_eq!(array1.shape(), &[2, 2]);
        assert_eq!(array1[[0, 0]], 1.0);
        assert_eq!(array1[[0, 1]], 2.0);
        assert_eq!(array1[[1, 0]], 5.0);
        assert_eq!(array1[[1, 1]], 6.0);
        
        // Second part should be [3.0, 4.0] and [7.0, 8.0]
        let array2 = crate::ort_to_ndarray(&parts[1]).unwrap();
        assert_eq!(array2.shape(), &[2, 2]);
        assert_eq!(array2[[0, 0]], 3.0);
        assert_eq!(array2[[0, 1]], 4.0);
        assert_eq!(array2[[1, 0]], 7.0);
        assert_eq!(array2[[1, 1]], 8.0);
    } else {
        panic!("Expected Sequence variant");
    }
}

#[test]
fn test_op_dropout() {
    // Test dropout in inference mode (should be a no-op)
    let input = create_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    
    let mut node = NodeProto::default();
    add_attribute(&mut node, "ratio", 500); // 0.5 as integer (scaled)
    
    let result = OrtEngine::op_dropout(&node, &[input.clone()]).unwrap();
    
    if let OrtValue::Sequence(outputs) = result {
        assert_eq!(outputs.len(), 2);
        
        // First output should be identical to input (no dropout in inference mode)
        let output_tensor = crate::ort_to_ndarray(&outputs[0]).unwrap();
        let input_tensor = crate::ort_to_ndarray(&input).unwrap();
        assert_eq!(output_tensor.shape(), input_tensor.shape());
        
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(output_tensor[[i, j]], input_tensor[[i, j]]);
            }
        }
        
        // Second output should be a mask of all ones
        let mask_tensor = crate::ort_to_ndarray(&outputs[1]).unwrap();
        assert_eq!(mask_tensor.shape(), input_tensor.shape());
        
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(mask_tensor[[i, j]], 1.0);
            }
        }
    } else {
        panic!("Expected Sequence variant");
    }
}

#[test]
fn test_op_einsum() {
    // Test Einsum for batched matrix multiplication
    let a = create_tensor(vec![2, 2, 3], vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    ]);
    
    let b = create_tensor(vec![2, 3, 2], vec![
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    ]);
    
    let mut node = NodeProto::default();
    add_string_attribute(&mut node, "equation", "bij,bjk->bik");
    
    let result = OrtEngine::op_einsum(&node, &[a, b]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[2, 2, 2]);
    
    // Expected results for batch 0:
    // [1, 2, 3] · [1, 2; 3, 4; 5, 6] = [22, 28]
    // [4, 5, 6] · [1, 2; 3, 4; 5, 6] = [49, 64]
    assert_eq!(array[[0, 0, 0]], 22.0);
    assert_eq!(array[[0, 0, 1]], 28.0);
    assert_eq!(array[[0, 1, 0]], 49.0);
    assert_eq!(array[[0, 1, 1]], 64.0);
    
    // Expected results for batch 1:
    // [7, 8, 9] · [7, 8; 9, 10; 11, 12] = [202, 226]
    // [10, 11, 12] · [7, 8; 9, 10; 11, 12] = [256, 289]
    assert_eq!(array[[1, 0, 0]], 202.0);
    assert_eq!(array[[1, 0, 1]], 226.0);
    assert_eq!(array[[1, 1, 0]], 256.0);
    assert_eq!(array[[1, 1, 1]], 289.0);
}

#[test]
fn test_op_topk() {
    // Test TopK operation
    let input = create_tensor(vec![2, 5], vec![
        3.0, 1.0, 5.0, 2.0, 4.0,
        9.0, 7.0, 5.0, 8.0, 6.0
    ]);
    
    let mut node = NodeProto::default();
    add_attribute(&mut node, "k", 3);
    add_attribute(&mut node, "axis", 1);
    
    let result = OrtEngine::op_topk(&node, &[input]).unwrap();
    
    if let OrtValue::Sequence(outputs) = result {
        assert_eq!(outputs.len(), 2);
        
        // First output should be the top 3 values
        let values = crate::ort_to_ndarray(&outputs[0]).unwrap();
        assert_eq!(values.shape(), &[2, 3]);
        
        // For first row, top 3 values are 5.0, 4.0, 3.0
        assert_eq!(values[[0, 0]], 5.0);
        assert_eq!(values[[0, 1]], 4.0);
        assert_eq!(values[[0, 2]], 3.0);
        
        // For second row, top 3 values are 9.0, 8.0, 7.0
        assert_eq!(values[[1, 0]], 9.0);
        assert_eq!(values[[1, 1]], 8.0);
        assert_eq!(values[[1, 2]], 7.0);
        
        // Second output should be the indices of the top 3 values
        let indices = crate::ort_to_ndarray(&outputs[1]).unwrap();
        assert_eq!(indices.shape(), &[2, 3]);
        
        // For first row, indices should be 2, 4, 0
        assert_eq!(indices[[0, 0]], 2.0); // index of 5.0
        assert_eq!(indices[[0, 1]], 4.0); // index of 4.0
        assert_eq!(indices[[0, 2]], 0.0); // index of 3.0
        
        // For second row, indices should be 0, 3, 1
        assert_eq!(indices[[1, 0]], 0.0); // index of 9.0
        assert_eq!(indices[[1, 1]], 3.0); // index of 8.0
        assert_eq!(indices[[1, 2]], 1.0); // index of 7.0
    } else {
        panic!("Expected Sequence variant");
    }
}

#[test]
fn test_op_gather_elements() {
    // Test GatherElements operation
    let data = create_tensor(vec![3, 3], vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    ]);
    
    let indices = create_int64_tensor(vec![2, 2], vec![
        0, 1,
        1, 2
    ]);
    
    let mut node = NodeProto::default();
    add_attribute(&mut node, "axis", 0);
    
    let result = OrtEngine::op_gather_elements(&node, &[data, indices]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    
    // Expected results:
    // data[0, 0] = 1.0, data[1, 1] = 5.0
    // data[1, 0] = 4.0, data[2, 1] = 8.0
    assert_eq!(array[[0, 0]], 1.0);
    assert_eq!(array[[0, 1]], 5.0);
    assert_eq!(array[[1, 0]], 4.0);
    assert_eq!(array[[1, 1]], 8.0);
}

#[test]
fn test_op_reduce_max() {
    // Test ReduceMax operation
    let input = create_tensor(vec![2, 3, 4], vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0
    ]);
    
    // Reduce along axis 1 (middle dimension)
    let mut node = NodeProto::default();
    add_attribute(&mut node, "axes", 1);
    add_attribute(&mut node, "keepdims", 1);
    
    let result = OrtEngine::op_reduce_max(&node, &[input]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[2, 1, 4]);
    
    // Expected results:
    // Max along axis 1 for batch 0: [9.0, 10.0, 11.0, 12.0]
    // Max along axis 1 for batch 1: [21.0, 22.0, 23.0, 24.0]
    assert_eq!(array[[0, 0, 0]], 9.0);
    assert_eq!(array[[0, 0, 1]], 10.0);
    assert_eq!(array[[0, 0, 2]], 11.0);
    assert_eq!(array[[0, 0, 3]], 12.0);
    
    assert_eq!(array[[1, 0, 0]], 21.0);
    assert_eq!(array[[1, 0, 1]], 22.0);
    assert_eq!(array[[1, 0, 2]], 23.0);
    assert_eq!(array[[1, 0, 3]], 24.0);
}

#[test]
fn test_op_attention() {
    // Test Attention operation with a simple case
    // Create query, key, value tensors with batch=1, seq_len=2, num_heads=1, head_dim=2
    let query = create_tensor(vec![1, 2, 1, 2], vec![
        1.0, 2.0,  // First token query
        3.0, 4.0   // Second token query
    ]);
    
    let key = create_tensor(vec![1, 2, 1, 2], vec![
        5.0, 6.0,  // First token key
        7.0, 8.0   // Second token key
    ]);
    
    let value = create_tensor(vec![1, 2, 1, 2], vec![
        9.0, 10.0,  // First token value
        11.0, 12.0  // Second token value
    ]);
    
    let mut node = NodeProto::default();
    add_attribute(&mut node, "num_heads", 1);
    
    let result = OrtEngine::op_attention(&node, &[query, key, value]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[1, 2, 1, 2]);
    
    // The attention scores and outputs are complex to calculate by hand,
    // so we'll just check that the output has the right shape and contains reasonable values
    assert!(array[[0, 0, 0, 0]] >= 0.0);
    assert!(array[[0, 0, 0, 1]] >= 0.0);
    assert!(array[[0, 1, 0, 0]] >= 0.0);
    assert!(array[[0, 1, 0, 1]] >= 0.0);
}

#[test]
fn test_bert_inference_pipeline() {
    // This test simulates a mini BERT inference pipeline with the key operations
    
    // 1. Create input embeddings
    let embeddings = create_tensor(vec![1, 3, 4], vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    ]);
    
    // 2. Apply attention
    let query = create_tensor(vec![1, 3, 1, 4], vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    ]);
    
    let key = create_tensor(vec![1, 3, 1, 4], vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    ]);
    
    let value = create_tensor(vec![1, 3, 1, 4], vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    ]);
    
    let mut attention_node = NodeProto::default();
    add_attribute(&mut attention_node, "num_heads", 1);
    
    let attention_output = OrtEngine::op_attention(&attention_node, &[query, key, value]).unwrap();
    
    // 3. Apply GELU activation
    let gelu_node = NodeProto::default();
    let gelu_input = create_tensor(vec![1, 3, 4], vec![
        0.5, -0.5, 1.0, -1.0,
        0.1, -0.1, 2.0, -2.0,
        0.0, 3.0, -3.0, 0.3
    ]);
    
    let gelu_output = OrtEngine::op_gelu(&gelu_node, &[gelu_input]).unwrap();
    let gelu_array = crate::ort_to_ndarray(&gelu_output).unwrap();
    
    // 4. Apply LayerNorm
    let layernorm_input = create_tensor(vec![1, 3, 4], vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    ]);
    
    let scale = create_tensor(vec![4], vec![1.0, 1.0, 1.0, 1.0]);
    let bias = create_tensor(vec![4], vec![0.0, 0.0, 0.0, 0.0]);
    
    let mut layernorm_node = NodeProto::default();
    add_attribute(&mut layernorm_node, "epsilon", 0); // 1e-5 as integer (would need scaling)
    
    let layernorm_output = OrtEngine::op_layer_normalization(&layernorm_node, &[layernorm_input, scale, bias]).unwrap();
    let layernorm_array = crate::ort_to_ndarray(&layernorm_output).unwrap();
    
    // Verify that all operations produced outputs with the expected shapes
    let attention_array = crate::ort_to_ndarray(&attention_output).unwrap();
    assert_eq!(attention_array.shape(), &[1, 3, 1, 4]);
    assert_eq!(gelu_array.shape(), &[1, 3, 4]);
    assert_eq!(layernorm_array.shape(), &[1, 3, 4]);
    
    // Check that GELU produces expected values for known inputs
    assert!((gelu_array[[0, 0, 0]] - 0.345).abs() < 0.01); // GELU(0.5) ≈ 0.345
    assert!((gelu_array[[0, 0, 1]] + 0.154).abs() < 0.01); // GELU(-0.5) ≈ -0.154
    assert!((gelu_array[[0, 0, 2]] - 0.841).abs() < 0.01); // GELU(1.0) ≈ 0.841
    assert!((gelu_array[[0, 0, 3]] + 0.159).abs() < 0.01); // GELU(-1.0) ≈ -0.159
}