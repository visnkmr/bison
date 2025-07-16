use std::sync::Arc;
use std::collections::HashMap;
use indexmap::IndexMap;
use crate::{OrtEngine, OrtValue, DataType, Dimensions, NodeProto, AttributeProto, MapKey, OrtResult};

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
fn test_sequence_construct() {
    let tensor1 = create_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let tensor2 = create_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_sequence_construct(&node, &[tensor1.clone(), tensor2.clone()]).unwrap();
    
    if let OrtValue::Sequence(seq) = result {
        assert_eq!(seq.len(), 2);
        assert_eq!(format!("{:?}", seq[0]), format!("{:?}", tensor1));
        assert_eq!(format!("{:?}", seq[1]), format!("{:?}", tensor2));
    } else {
        panic!("Expected Sequence variant");
    }
}

#[test]
fn test_sequence_empty() {
    let mut node = NodeProto::default();
    add_attribute(&mut node, "dtype", 1); // Float
    
    let result = OrtEngine::op_sequence_empty(&node, &[]).unwrap();
    
    if let OrtValue::Sequence(seq) = result {
        assert_eq!(seq.len(), 0);
    } else {
        panic!("Expected Sequence variant");
    }
}

#[test]
fn test_sequence_at() {
    let tensor1 = create_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let tensor2 = create_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let sequence = OrtValue::Sequence(vec![tensor1.clone(), tensor2.clone()]);
    
    // Test with positive index
    let position = create_int64_tensor(vec![1], vec![0]);
    let node = NodeProto::default();
    let result = OrtEngine::op_sequence_at(&node, &[sequence.clone(), position]).unwrap();
    
    assert_eq!(format!("{:?}", result), format!("{:?}", tensor1));
    
    // Test with negative index
    let position = create_int64_tensor(vec![1], vec![-1]);
    let result = OrtEngine::op_sequence_at(&node, &[sequence, position]).unwrap();
    
    assert_eq!(format!("{:?}", result), format!("{:?}", tensor2));
}

#[test]
fn test_sequence_erase() {
    let tensor1 = create_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let tensor2 = create_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let tensor3 = create_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);
    let sequence = OrtValue::Sequence(vec![tensor1.clone(), tensor2.clone(), tensor3.clone()]);
    
    // Erase middle element
    let position = create_int64_tensor(vec![1], vec![1]);
    let node = NodeProto::default();
    let result = OrtEngine::op_sequence_erase(&node, &[sequence, position]).unwrap();
    
    if let OrtValue::Sequence(seq) = result {
        assert_eq!(seq.len(), 2);
        assert_eq!(format!("{:?}", seq[0]), format!("{:?}", tensor1));
        assert_eq!(format!("{:?}", seq[1]), format!("{:?}", tensor3));
    } else {
        panic!("Expected Sequence variant");
    }
}

#[test]
fn test_sequence_insert() {
    let tensor1 = create_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let tensor2 = create_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let tensor3 = create_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);
    let sequence = OrtValue::Sequence(vec![tensor1.clone(), tensor3.clone()]);
    
    // Insert in the middle
    let position = create_int64_tensor(vec![1], vec![1]);
    let node = NodeProto::default();
    let result = OrtEngine::op_sequence_insert(&node, &[sequence, tensor2.clone(), position]).unwrap();
    
    if let OrtValue::Sequence(seq) = result {
        assert_eq!(seq.len(), 3);
        assert_eq!(format!("{:?}", seq[0]), format!("{:?}", tensor1));
        assert_eq!(format!("{:?}", seq[1]), format!("{:?}", tensor2));
        assert_eq!(format!("{:?}", seq[2]), format!("{:?}", tensor3));
    } else {
        panic!("Expected Sequence variant");
    }
}

#[test]
fn test_sequence_length() {
    let tensor1 = create_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let tensor2 = create_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let sequence = OrtValue::Sequence(vec![tensor1.clone(), tensor2.clone()]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_sequence_length(&node, &[sequence]).unwrap();
    
    if let OrtValue::Tensor { dtype, data, .. } = result {
        assert_eq!(dtype, DataType::Int64);
        let length = i64::from_le_bytes(data[..8].try_into().unwrap());
        assert_eq!(length, 2);
    } else {
        panic!("Expected Tensor variant");
    }
}

#[test]
fn test_map_from_tensor() {
    // Create keys tensor
    let keys = create_int64_tensor(vec![3], vec![1, 2, 3]);
    
    // Create values tensor
    let values = create_tensor(vec![3], vec![10.0, 20.0, 30.0]);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_map_from_tensor(&node, &[keys, values]).unwrap();
    
    if let OrtValue::Map(map) = result {
        assert_eq!(map.len(), 3);
        
        // Check keys and values
        let key1 = MapKey::Int64(1);
        let key2 = MapKey::Int64(2);
        let key3 = MapKey::Int64(3);
        
        assert!(map.contains_key(&key1));
        assert!(map.contains_key(&key2));
        assert!(map.contains_key(&key3));
        
        // Check a value
        if let OrtValue::Tensor { data, .. } = &map[&key1] {
            let value = f32::from_le_bytes(data[..4].try_into().unwrap());
            assert_eq!(value, 10.0);
        } else {
            panic!("Expected Tensor variant");
        }
    } else {
        panic!("Expected Map variant");
    }
}

#[test]
fn test_map_get() {
    // Create a map
    let mut map = IndexMap::new();
    map.insert(MapKey::Int64(1), create_tensor(vec![1], vec![10.0]));
    map.insert(MapKey::Int64(2), create_tensor(vec![1], vec![20.0]));
    map.insert(MapKey::Int64(3), create_tensor(vec![1], vec![30.0]));
    
    let map_value = OrtValue::Map(map);
    
    // Get an existing key
    let key = create_int64_tensor(vec![1], vec![2]);
    let node = NodeProto::default();
    let result = OrtEngine::op_map_get(&node, &[map_value.clone(), key]).unwrap();
    
    if let OrtValue::Tensor { data, .. } = result {
        let value = f32::from_le_bytes(data[..4].try_into().unwrap());
        assert_eq!(value, 20.0);
    } else {
        panic!("Expected Tensor variant");
    }
    
    // Get a non-existing key with default value
    let key = create_int64_tensor(vec![1], vec![4]);
    let default_value = create_tensor(vec![1], vec![40.0]);
    let result = OrtEngine::op_map_get(&node, &[map_value, key, default_value.clone()]).unwrap();
    
    assert_eq!(format!("{:?}", result), format!("{:?}", default_value));
}

#[test]
fn test_map_has_key() {
    // Create a map
    let mut map = IndexMap::new();
    map.insert(MapKey::Int64(1), create_tensor(vec![1], vec![10.0]));
    map.insert(MapKey::Int64(2), create_tensor(vec![1], vec![20.0]));
    
    let map_value = OrtValue::Map(map);
    
    // Test with existing key
    let key = create_int64_tensor(vec![1], vec![1]);
    let node = NodeProto::default();
    let result = OrtEngine::op_map_has_key(&node, &[map_value.clone(), key]).unwrap();
    
    if let OrtValue::Tensor { dtype, data, .. } = result {
        assert_eq!(dtype, DataType::Int64);
        let has_key = i64::from_le_bytes(data[..8].try_into().unwrap());
        assert_eq!(has_key, 1); // true
    } else {
        panic!("Expected Tensor variant");
    }
    
    // Test with non-existing key
    let key = create_int64_tensor(vec![1], vec![3]);
    let result = OrtEngine::op_map_has_key(&node, &[map_value, key]).unwrap();
    
    if let OrtValue::Tensor { dtype, data, .. } = result {
        assert_eq!(dtype, DataType::Int64);
        let has_key = i64::from_le_bytes(data[..8].try_into().unwrap());
        assert_eq!(has_key, 0); // false
    } else {
        panic!("Expected Tensor variant");
    }
}

#[test]
fn test_map_keys() {
    // Create a map
    let mut map = IndexMap::new();
    map.insert(MapKey::Int64(1), create_tensor(vec![1], vec![10.0]));
    map.insert(MapKey::Int64(2), create_tensor(vec![1], vec![20.0]));
    map.insert(MapKey::Int64(3), create_tensor(vec![1], vec![30.0]));
    
    let map_value = OrtValue::Map(map);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_map_keys(&node, &[map_value]).unwrap();
    
    if let OrtValue::Tensor { dtype, shape, data, .. } = result {
        assert_eq!(dtype, DataType::Int64);
        assert_eq!(shape.len(), 1);
        
        if let Dimensions::Fixed(size) = shape[0] {
            assert_eq!(size, 3);
            
            // Extract keys
            let keys: Vec<i64> = data.chunks(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            
            // Check keys (order may vary)
            assert!(keys.contains(&1));
            assert!(keys.contains(&2));
            assert!(keys.contains(&3));
        } else {
            panic!("Expected Fixed dimension");
        }
    } else {
        panic!("Expected Tensor variant");
    }
}

#[test]
fn test_map_to_tensor() {
    // Create a map
    let mut map = IndexMap::new();
    map.insert(MapKey::Int64(1), create_tensor(vec![1], vec![10.0]));
    map.insert(MapKey::Int64(2), create_tensor(vec![1], vec![20.0]));
    
    let map_value = OrtValue::Map(map);
    
    let node = NodeProto::default();
    let result = OrtEngine::op_map_to_tensor(&node, &[map_value]).unwrap();
    
    if let OrtValue::Sequence(seq) = result {
        assert_eq!(seq.len(), 2); // Keys and values tensors
        
        // Check keys tensor
        if let OrtValue::Tensor { dtype: key_dtype, shape: key_shape, data: key_data, .. } = &seq[0] {
            assert_eq!(*key_dtype, DataType::Int64);
            assert_eq!(key_shape.len(), 1);
            
            if let Dimensions::Fixed(size) = key_shape[0] {
                assert_eq!(size, 2);
                
                // Extract keys
                let keys: Vec<i64> = key_data.chunks(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                
                // Check keys (order may vary)
                assert!(keys.contains(&1));
                assert!(keys.contains(&2));
            } else {
                panic!("Expected Fixed dimension");
            }
        } else {
            panic!("Expected Tensor variant for keys");
        }
        
        // Check values tensor (simplified check)
        if let OrtValue::Tensor { .. } = &seq[1] {
            // Values tensor exists
        } else {
            panic!("Expected Tensor variant for values");
        }
    } else {
        panic!("Expected Sequence variant");
    }
}