use std::sync::Arc;
use crate::{OrtEngine, OrtValue, DataType, Dimensions, OrtError, NodeProto, AttributeProto};

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

#[test]
fn test_op_add() {
    let a = create_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = create_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_add(&node, &[a, b]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    assert_eq!(array[[0, 0]], 6.0);  // 1 + 5
    assert_eq!(array[[0, 1]], 8.0);  // 2 + 6
    assert_eq!(array[[1, 0]], 10.0); // 3 + 7
    assert_eq!(array[[1, 1]], 12.0); // 4 + 8
}

#[test]
fn test_op_sub() {
    let a = create_tensor(vec![2, 2], vec![5.0, 7.0, 9.0, 11.0]);
    let b = create_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_sub(&node, &[a, b]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    assert_eq!(array[[0, 0]], 4.0);  // 5 - 1
    assert_eq!(array[[0, 1]], 5.0);  // 7 - 2
    assert_eq!(array[[1, 0]], 6.0);  // 9 - 3
    assert_eq!(array[[1, 1]], 7.0);  // 11 - 4
}

#[test]
fn test_op_mul() {
    let a = create_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = create_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_mul(&node, &[a, b]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    assert_eq!(array[[0, 0]], 5.0);   // 1 * 5
    assert_eq!(array[[0, 1]], 12.0);  // 2 * 6
    assert_eq!(array[[1, 0]], 21.0);  // 3 * 7
    assert_eq!(array[[1, 1]], 32.0);  // 4 * 8
}

#[test]
fn test_op_div() {
    let a = create_tensor(vec![2, 2], vec![10.0, 12.0, 14.0, 16.0]);
    let b = create_tensor(vec![2, 2], vec![2.0, 3.0, 7.0, 4.0]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_div(&node, &[a, b]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    assert_eq!(array[[0, 0]], 5.0);   // 10 / 2
    assert_eq!(array[[0, 1]], 4.0);   // 12 / 3
    assert_eq!(array[[1, 0]], 2.0);   // 14 / 7
    assert_eq!(array[[1, 1]], 4.0);   // 16 / 4
}

#[test]
fn test_op_matmul() {
    let a = create_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = create_tensor(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_matmul(&node, &[a, b]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    
    // [1, 2, 3] · [7, 8] = 1*7 + 2*9 + 3*11 = 58
    assert_eq!(array[[0, 0]], 58.0);
    
    // [1, 2, 3] · [8, 10] = 1*8 + 2*10 + 3*12 = 64
    assert_eq!(array[[0, 1]], 64.0);
    
    // [4, 5, 6] · [7, 8] = 4*7 + 5*9 + 6*11 = 139
    assert_eq!(array[[1, 0]], 139.0);
    
    // [4, 5, 6] · [8, 10] = 4*8 + 5*10 + 6*12 = 154
    assert_eq!(array[[1, 1]], 154.0);
}

#[test]
fn test_op_softmax() {
    let input = create_tensor(vec![1, 3], vec![1.0, 2.0, 3.0]);
    
    // Create node with axis attribute
    let mut node = NodeProto::default();
    add_attribute(&mut node, "axis", 1);
    
    let result = OrtEngine::op_softmax(&node, &[input]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    
    // Calculate expected softmax values
    let exp_sum = (1.0_f32).exp() + (2.0_f32).exp() + (3.0_f32).exp();
    let expected = vec![(1.0_f32).exp() / exp_sum, (2.0_f32).exp() / exp_sum, (3.0_f32).exp() / exp_sum];
    
    assert_eq!(array.shape(), &[1, 3]);
    assert!((array[[0, 0]] - expected[0]).abs() < 1e-6);
    assert!((array[[0, 1]] - expected[1]).abs() < 1e-6);
    assert!((array[[0, 2]] - expected[2]).abs() < 1e-6);
}

#[test]
fn test_op_gather() {
    let data = create_tensor(vec![3, 3], vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    ]);
    let indices = create_int64_tensor(vec![2], vec![0, 2]);
    
    let mut node = NodeProto::default();
    add_attribute(&mut node, "axis", 0);
    
    let result = OrtEngine::op_gather(&node, &[data, indices]).unwrap();
    
    let array = crate::ort_to_ndarray(&result).unwrap();
    assert_eq!(array.shape(), &[2, 3]);
    
    // First row (index 0)
    assert_eq!(array[[0, 0]], 1.0);
    assert_eq!(array[[0, 1]], 2.0);
    assert_eq!(array[[0, 2]], 3.0);
    
    // Third row (index 2)
    assert_eq!(array[[1, 0]], 7.0);
    assert_eq!(array[[1, 1]], 8.0);
    assert_eq!(array[[1, 2]], 9.0);
}

#[test]
fn test_error_handling() {
    // Test with incompatible shapes for Add
    let a = create_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = create_tensor(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_add(&node, &[a, b]);
    
    assert!(result.is_err());
    if let Err(OrtError::TypeMismatch(_)) = result {
        // Expected error type
    } else {
        panic!("Expected TypeMismatch error");
    }
}