use ndarray::{ArrayBase, ArrayD, Axis, Data, Dim, IxDyn, Slice,ErrorKind};
use thiserror::Error;
use std::sync::Arc;
use std::collections::{HashSet, };
use indexmap::IndexMap;
use crate::{DataType, Dimensions, OrtError, OrtResult, OrtValue, NodeProto};
use std::fmt::Debug;
use std::ops::{Add, ControlFlow, Div, FromResidual, Mul, Sub, Try};




#[derive(Error, Debug)]
pub enum ArrayDResultError {
    #[error("Mismatched variants: cannot perform operation between {0} and {1}")]
    MismatchedVariants(String, String),
    // #[error("Shape mismatch: {0}")]
    // ShapeMismatch(#[from] NdArrayError),
    #[error("Division by zero")]
    DivisionByZero,
}

#[derive(Debug)]
pub enum ArrayDResult {
    Float(ArrayD<f32>),
    Int64(ArrayD<i64>),
    Int32(ArrayD<i32>),
}



macro_rules! impl_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for ArrayDResult {
            type Output = Result<ArrayDResult, ArrayDResultError>;

            fn $method(self, other: Self) -> Self::Output {
                match (self, other) {
                    (ArrayDResult::Float(a), ArrayDResult::Float(b)) => {
                        Ok(ArrayDResult::Float(a $op b))
                    }
                    (ArrayDResult::Int64(a), ArrayDResult::Int64(b)) => {
                        Ok(ArrayDResult::Int64(a $op b))
                    }
                    (ArrayDResult::Int32(a), ArrayDResult::Int32(b)) => {
                        Ok(ArrayDResult::Int32(a $op b))
                    }
                    (a, b) => {
                        let a_type = match a {
                            ArrayDResult::Float(_) => "Float",
                            ArrayDResult::Int64(_) => "Int64",
                            ArrayDResult::Int32(_) => "Int32",
                        };
                        let b_type = match b {
                            ArrayDResult::Float(_) => "Float",
                            ArrayDResult::Int64(_) => "Int64",
                            ArrayDResult::Int32(_) => "Int32",
                        };
                        Err(ArrayDResultError::MismatchedVariants(
                            a_type.to_string(),
                            b_type.to_string(),
                        ))
                    }
                }
            }
        }
    };
}

impl_op!(Add, add, +);
impl_op!(Sub, sub, -);
impl_op!(Mul, mul, *);

// Special handling for division to check for division by zero
impl Div for ArrayDResult {
    type Output = Result<ArrayDResult, ArrayDResultError>;

    fn div(self, other: Self) -> Self::Output {
        match (&self, &other) {
            (ArrayDResult::Float(_), ArrayDResult::Float(b)) => {
                if b.iter().any(|&x| x == 0.0) {
                    return Err(ArrayDResultError::DivisionByZero);
                }
            }
            (ArrayDResult::Int64(_), ArrayDResult::Int64(b)) => {
                if b.iter().any(|&x| x == 0) {
                    return Err(ArrayDResultError::DivisionByZero);
                }
            }
            (ArrayDResult::Int32(_), ArrayDResult::Int32(b)) => {
                if b.iter().any(|&x| x == 0) {
                    return Err(ArrayDResultError::DivisionByZero);
                }
            }
            _ => {}
        }

        match (self, other) {
            (ArrayDResult::Float(a), ArrayDResult::Float(b)) => {
                Ok(ArrayDResult::Float(a / b))
            }
            (ArrayDResult::Int64(a), ArrayDResult::Int64(b)) => {
                Ok(ArrayDResult::Int64(a / b))
            }
            (ArrayDResult::Int32(a), ArrayDResult::Int32(b)) => {
                Ok(ArrayDResult::Int32(a / b))
            }
            (a, b) => {
                let a_type = match a {
                    ArrayDResult::Float(_) => "Float",
                    ArrayDResult::Int64(_) => "Int64",
                    ArrayDResult::Int32(_) => "Int32",
                };
                let b_type = match b {
                    ArrayDResult::Float(_) => "Float",
                    ArrayDResult::Int64(_) => "Int64",
                    ArrayDResult::Int32(_) => "Int32",
                };
                Err(ArrayDResultError::MismatchedVariants(
                    a_type.to_string(),
                    b_type.to_string(),
                ))
            }
        }
    }
}

// Trait to define conversion behavior for supported types
// pub trait OrtConvertible: Sized + Clone + Copy + Debug + Send + Sync + 'static {
//     const DATA_TYPE: DataType;
//     const BYTE_SIZE: usize;
//     fn from_le_bytes(chunk: &[u8]) -> Self;
//     fn to_le_bytes(&self) -> Vec<u8>;
// }

// impl OrtConvertible for f32 {
//     const DATA_TYPE: DataType = DataType::Float;
//     const BYTE_SIZE: usize = 4;
//     fn from_le_bytes(chunk: &[u8]) -> Self {
//         f32::from_le_bytes(chunk.try_into().unwrap())
//     }
//     fn to_le_bytes(&self) -> Vec<u8> {
//         self.to_le_bytes().to_vec()
//     }
// }

// impl OrtConvertible for i64 {
//     const DATA_TYPE: DataType = DataType::Int64;
//     const BYTE_SIZE: usize = 8;
//     fn from_le_bytes(chunk: &[u8]) -> Self {
//         i64::from_le_bytes(chunk.try_into().unwrap())
//     }
//     fn to_le_bytes(&self) -> Vec<u8> {
//         self.to_le_bytes().to_vec()
//     }
// }

// impl OrtConvertible for i32 {
//     const DATA_TYPE: DataType = DataType::Int32;
//     const BYTE_SIZE: usize = 4;
//     fn from_le_bytes(chunk: &[u8]) -> Self {
//         i32::from_le_bytes(chunk.try_into().unwrap())
//     }
//     fn to_le_bytes(&self) -> Vec<u8> {
//         self.to_le_bytes().to_vec()
//     }
// }

// // Additional trait for numeric operations
// pub trait NumericOps: OrtConvertible + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> {
//     fn zero() -> Self;
//     fn one() -> Self;
//     fn from_f32(val: f32) -> Self;
//     fn to_f32(&self) -> f32;
//     fn pow(&self, exp: Self) -> Self;
//     fn sqrt(&self) -> Self;
//     fn exp(&self) -> Self;
//     fn tanh(&self) -> Self;
//     fn sin(&self) -> Self;
//     fn cos(&self) -> Self;
//     fn atan(&self) -> Self;
//     fn floor(&self) -> Self;
//     fn round(&self) -> Self;
//     fn clamp(&self, min: Self, max: Self) -> Self;
//     fn is_zero(&self) -> bool;
// }

// impl NumericOps for f32 {
//     fn zero() -> Self { 0.0 }
//     fn one() -> Self { 1.0 }
//     fn from_f32(val: f32) -> Self { val }
//     fn to_f32(&self) -> f32 { *self }
//     fn pow(&self, exp: Self) -> Self { self.powf(exp) }
//     fn sqrt(&self) -> Self { self.sqrt() }
//     fn exp(&self) -> Self { self.exp() }
//     fn tanh(&self) -> Self { self.tanh() }
//     fn sin(&self) -> Self { self.sin() }
//     fn cos(&self) -> Self { self.cos() }
//     fn atan(&self) -> Self { self.atan() }
//     fn floor(&self) -> Self { self.floor() }
//     fn round(&self) -> Self { self.round() }
//     fn clamp(&self, min: Self, max: Self) -> Self { self.clamp(min, max) }
//     fn is_zero(&self) -> bool { *self == 0.0 }
// }

// impl NumericOps for i64 {
//     fn zero() -> Self { 0 }
//     fn one() -> Self { 1 }
//     fn from_f32(val: f32) -> Self { val as i64 }
//     fn to_f32(&self) -> f32 { *self as f32 }
//     fn pow(&self, exp: Self) -> Self { (*self as f64).powf(exp as f64) as i64 }
//     fn sqrt(&self) -> Self { (*self as f64).sqrt() as i64 }
//     fn exp(&self) -> Self { (*self as f64).exp() as i64 }
//     fn tanh(&self) -> Self { (*self as f64).tanh() as i64 }
//     fn sin(&self) -> Self { (*self as f64).sin() as i64 }
//     fn cos(&self) -> Self { (*self as f64).cos() as i64 }
//     fn atan(&self) -> Self { (*self as f64).atan() as i64 }
//     fn floor(&self) -> Self { *self } // Integer already floored
//     fn round(&self) -> Self { *self } // Integer already rounded
//     fn clamp(&self, min: Self, max: Self) -> Self { 
//         if *self < min { min } else if *self > max { max } else { *self }
//     }
//     fn is_zero(&self) -> bool { *self == 0 }
// }

// impl NumericOps for i32 {
//     fn zero() -> Self { 0 }
//     fn one() -> Self { 1 }
//     fn from_f32(val: f32) -> Self { val as i32 }
//     fn to_f32(&self) -> f32 { *self as f32 }
//     fn pow(&self, exp: Self) -> Self { (*self as f64).powf(exp as f64) as i32 }
//     fn sqrt(&self) -> Self { (*self as f64).sqrt() as i32 }
//     fn exp(&self) -> Self { (*self as f64).exp() as i32 }
//     fn tanh(&self) -> Self { (*self as f64).tanh() as i32 }
//     fn sin(&self) -> Self { (*self as f64).sin() as i32 }
//     fn cos(&self) -> Self { (*self as f64).cos() as i32 }
//     fn atan(&self) -> Self { (*self as f64).atan() as i32 }
//     fn floor(&self) -> Self { *self } // Integer already floored
//     fn round(&self) -> Self { *self } // Integer already rounded
//     fn clamp(&self, min: Self, max: Self) -> Self { 
//         if *self < min { min } else if *self > max { max } else { *self }
//     }
//     fn is_zero(&self) -> bool { *self == 0 }
// }

// // Helper function to convert OrtValue to ndarray
// pub fn ort_to_ndarray<T: OrtConvertible>(ort: &OrtValue) -> OrtResult<ArrayD<T>> {
//     match ort {
//         OrtValue::Tensor { shape, dtype, data, .. } => {
//             let u = dtype.clone();
//             // Check if dtype matches the expected type
//             if u != T::DATA_TYPE {
//                 return Err(OrtError::TypeMismatch("Mismatched data type"));
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

// Generic function to convert ndarray to OrtValue
pub fn ndarray_to_ort(array: ArrayDResult, dtype: DataType) -> OrtValue {
    let (shape,data)=match(array){
        ArrayDResult::Float(array_base) => {
            (array_base.shape().iter().map(|&n| Dimensions::Fixed(n)).collect(),array_base.into_raw_vec()
            .into_iter()
            .flat_map(|x| x.to_le_bytes())
            .collect())
        },
        ArrayDResult::Int64(array_base) => {
            (array_base.shape().iter().map(|&n| Dimensions::Fixed(n)).collect(),array_base.into_raw_vec()
            .into_iter()
            .flat_map(|x| x.to_le_bytes())
            .collect())
        },
        ArrayDResult::Int32(array_base) => {
           ( array_base.shape().iter().map(|&n| Dimensions::Fixed(n)).collect(),array_base.into_raw_vec()
           .into_iter()
           .flat_map(|x| x.to_le_bytes())
           .collect())
        },
    };
    OrtValue::Tensor {
        shape,
        dtype,
        data: Arc::new(data),
    }
}

// Specialized version for f32 for backward compatibility
pub fn ndarray_to_ort_f32(array: ArrayD<f32>) -> OrtValue {
    ndarray_to_ort(ArrayDResult::Float(array), DataType::Float)
}

// Type conversion utilities
// pub fn convert_tensor<T: OrtConvertible, U: OrtConvertible>(tensor: &OrtValue) -> OrtResult<OrtValue> {
//     let array = ort_to_ndarray::<T>(tensor)?;
//     let converted: Vec<U> = array.iter().map(|&x| U::from(x.to_f32())).collect();
//     let shape = array.shape().to_vec();
//     let converted_array = ArrayD::from_shape_vec(IxDyn(&shape), converted)
//         .map_err(|_| OrtError::InvalidTensorData("Shape mismatch during conversion".into()))?;
    
//     Ok(ndarray_to_ort(converted_array, U::DATA_TYPE))
// }

// Enum to hold different array types

// Helper function to convert OrtValue to ndarray of appropriate type
pub fn ort_to_ndarray(ort: &OrtValue) -> OrtResult<ArrayDResult> {
    println!("{:?}", ort);
    match ort {
        OrtValue::Tensor { shape, dtype, data, .. } => {
            // Check if shape contains symbolic dimensions
            if shape.iter().any(|d| matches!(d, Dimensions::Symbolic(_))) {
                return Err(OrtError::InvalidTensorData("Cannot convert symbolic shape to ndarray".into()));
            }
            let concrete_shape: Vec<usize> = shape.iter().map(|d| match d {
                Dimensions::Fixed(n) => *n,
                Dimensions::Symbolic(_) => unreachable!(), // Handled above
            }).collect();

            match dtype {
                DataType::Float => {
                    let float_data: Vec<f32> = data
                        .chunks(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    ArrayD::from_shape_vec(IxDyn(&concrete_shape), float_data)
                        .map(|arr| ArrayDResult::Float(arr))
                        .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for float tensor".into()))
                }
                DataType::Int64 => {
                    let int64_data: Vec<i64> = data
                        .chunks(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    ArrayD::from_shape_vec(IxDyn(&concrete_shape), int64_data)
                        .map(|arr| ArrayDResult::Int64(arr))
                        .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for int64 tensor".into()))
                }
                DataType::Int32 => {
                    let int32_data: Vec<i32> = data
                        .chunks(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    ArrayD::from_shape_vec(IxDyn(&concrete_shape), int32_data)
                        .map(|arr| ArrayDResult::Int32(arr))
                        .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for int32 tensor".into()))
                }
                _ => Err(OrtError::TypeMismatch("Unsupported tensor type, expected Float, Int64, or Int32")),
            }
        }
        _ => Err(OrtError::TypeMismatch("Expected tensor")),
    }
}