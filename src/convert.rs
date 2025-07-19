use ndarray::{ArrayBase, ArrayD, Axis, Data, Dim, IxDyn, Slice};
// use onnxruntime::{OrtError, OrtResult, OrtValue, DataType, Dimensions};
use std::sync::Arc;
use std::collections::{HashSet, };
use indexmap::IndexMap;
// use crate::slice::ndarray_to_ort::SupportedType;
use crate::{DataType, Dimensions, OrtError, OrtResult, OrtValue,NodeProto};


// Enum to hold different array types
// pub enum ArrayDResult {
//     Float(ArrayD<f32>),
//     Int64(ArrayD<i64>),
//     Int32(ArrayD<i32>),
// }

// // Helper function to convert OrtValue to ndarray of appropriate type
// pub fn ort_to_ndarray(ort: &OrtValue) -> OrtResult<ArrayDResult> {
//     println!("{:?}", ort);
//     match ort {
//         OrtValue::Tensor { shape, dtype, data, .. } => {
//             // Check if shape contains symbolic dimensions
//             if shape.iter().any(|d| matches!(d, Dimensions::Symbolic(_))) {
//                 return Err(OrtError::InvalidTensorData("Cannot convert symbolic shape to ndarray".into()));
//             }
//             let concrete_shape: Vec<usize> = shape.iter().map(|d| match d {
//                 Dimensions::Fixed(n) => *n,
//                 Dimensions::Symbolic(_) => unreachable!(), // Handled above
//             }).collect();

//             match dtype {
//                 DataType::Float => {
//                     let float_data: Vec<f32> = data
//                         .chunks(4)
//                         .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
//                         .collect();
//                     ArrayD::from_shape_vec(IxDyn(&concrete_shape), float_data)
//                         .map(|arr| ArrayDResult::Float(arr))
//                         .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for float tensor".into()))
//                 }
//                 DataType::Int64 => {
//                     let int64_data: Vec<i64> = data
//                         .chunks(8)
//                         .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
//                         .collect();
//                     ArrayD::from_shape_vec(IxDyn(&concrete_shape), int64_data)
//                         .map(|arr| ArrayDResult::Int64(arr))
//                         .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for int64 tensor".into()))
//                 }
//                 DataType::Int32 => {
//                     let int32_data: Vec<i32> = data
//                         .chunks(4)
//                         .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
//                         .collect();
//                     ArrayD::from_shape_vec(IxDyn(&concrete_shape), int32_data)
//                         .map(|arr| ArrayDResult::Int32(arr))
//                         .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for int32 tensor".into()))
//                 }
//                 _ => Err(OrtError::TypeMismatch("Unsupported tensor type, expected Float, Int64, or Int32")),
//             }
//         }
//         _ => Err(OrtError::TypeMismatch("Expected tensor")),
//     }
// }

// Trait to define conversion behavior for supported types
// pub trait OrtConvertible: Sized {
//     const DATA_TYPE: DataType;
//     const BYTE_SIZE: usize;
//     fn from_le_bytes(chunk: &[u8]) -> Self;
// }

// impl OrtConvertible for f32 {
//     const DATA_TYPE: DataType = DataType::Float;
//     const BYTE_SIZE: usize = 4;
//     fn from_le_bytes(chunk: &[u8]) -> Self {
//         f32::from_le_bytes(chunk.try_into().unwrap())
//     }
// }

// impl OrtConvertible for i64 {
//     const DATA_TYPE: DataType = DataType::Int64;
//     const BYTE_SIZE: usize = 8;
//     fn from_le_bytes(chunk: &[u8]) -> Self {
//         i64::from_le_bytes(chunk.try_into().unwrap())
//     }
// }

// impl OrtConvertible for i32 {
//     const DATA_TYPE: DataType = DataType::Int32;
//     const BYTE_SIZE: usize = 4;
//     fn from_le_bytes(chunk: &[u8]) -> Self {
//         i32::from_le_bytes(chunk.try_into().unwrap())
//     }
// }

// // Helper function to convert OrtValue to ndarray
// pub fn ort_to_ndarray<T: OrtConvertible>(ort: &OrtValue) -> OrtResult<ArrayD<T>> {
//     println!("{:?}", ort);
//     match ort {
//         OrtValue::Tensor { shape, dtype, data, .. } => {
//             let u=dtype.clone();
//             // Check if dtype matches the expected type
//             if u != T::DATA_TYPE {
//                 return Err(OrtError::TypeMismatch("Mismatched data type"))
//             }

//             // Check if shape contains symbolic dimensions
//             if shape.iter().any(|d| matches!(d, Dimensions::Symbolic(_))) {
//                 return Err(OrtError::InvalidTensorData("Cannot convert symbolic shape to ndarray".into()));
//             }

//             // Convert shape to concrete dimensions
//             let concrete_shape: Vec<usize> = shape.iter().map(|d| match d {
//                 Dimensions::Fixed(n) => *n,
//                 Dimensions::Symbolic(_) => unreachable!(), // Handled above
//             }).collect();

//             // Convert raw data to Vec<T>
//             let converted_data: Vec<T> = data
//                 .chunks(T::BYTE_SIZE)
//                 .map(T::from_le_bytes)
//                 .collect();

//             // Create ArrayD from shape and data
//             ArrayD::from_shape_vec(IxDyn(&concrete_shape), converted_data)
//                 .map_err(|_| OrtError::InvalidTensorData("Shape mismatch".into()))
//         }
//         _ => Err(OrtError::TypeMismatch("Expected tensor")),
//     }
// }


// pub fn ndarray_to_ort(array: ArrayD<f32>, dtype: DataType) -> OrtValue {
//     let shape: Vec<Dimensions> = array.shape().iter().map(|&n| Dimensions::Fixed(n)).collect();
//     let data: Vec<u8> = array
//         .into_raw_vec()
//         .into_iter()
//         .flat_map(|x| x.to_le_bytes())
//         .collect();
//     OrtValue::Tensor {
//         shape,
//         dtype,
//         data: Arc::new(data),
//     }
// }

// Helper function to convert OrtValue to ndarray
pub fn ort_to_ndarray(ort: &OrtValue) -> OrtResult<ArrayD<f32>> {
    match ort {
        OrtValue::Tensor { shape, dtype: DataType::Float, data, .. } => {
            // Check if shape contains symbolic dimensions
            if shape.iter().any(|d| matches!(d, Dimensions::Symbolic(_))) {
                return Err(OrtError::InvalidTensorData("Cannot convert symbolic shape to ndarray".into()));
            }
            let concrete_shape: Vec<usize> = shape.iter().map(|d| match d {
                Dimensions::Fixed(n) => *n,
                Dimensions::Symbolic(_) => unreachable!(), // Handled above
            }).collect();
            let float_data: Vec<f32> = data
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            ArrayD::from_shape_vec(concrete_shape, float_data)
                .map_err(|_| OrtError::InvalidTensorData("Shape mismatch".into()))
        }
        _ => Err(OrtError::TypeMismatch("Expected float tensor")),
    }
}

pub fn ndarray_to_ort(array: ArrayD<f32>, dtype: DataType) -> OrtValue {
    let shape: Vec<Dimensions> = array.shape().iter().map(|&n| Dimensions::Fixed(n)).collect();
    let data: Vec<u8> = array
        .into_raw_vec()
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    OrtValue::Tensor {
        shape,
        dtype,
        data: Arc::new(data),
    }
}
