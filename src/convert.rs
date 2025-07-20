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

pub fn pow_array(a: &ArrayDResult, b: &ArrayDResult) -> Result<ArrayDResult, OrtError> {
        fn is_scalar_like(arr: &ArrayDResult) -> bool {
        match arr {
            ArrayDResult::Float(arr) => arr.len() == 1,
            ArrayDResult::Int64(arr) => arr.len() == 1,
            ArrayDResult::Int32(arr) => arr.len() == 1,
        }
    }

    // Extract scalar value from b if scalar-like
    // fn get_scalar_value(b: &ArrayDResult) -> Result<f64, OrtError> {
        // let b_value: &f32=match b {
        //     ArrayDResult::Float(arr) if arr.len() == 1 => Ok(arr.first().unwrap().into()),
        //     ArrayDResult::Int64(arr) if arr.len() == 1 => Ok(arr.first().unwrap().into() as &f32),
        //     ArrayDResult::Int32(arr) if arr.len() == 1 => Ok(arr.first().unwrap().into() as &f32),
        //     _ => Err(()
        //         // OrtError::ShapeMismatch("Exponent must be a scalar-like value".into())
        //     ),
        // }.unwrap();
    // }

    // Ensure b is scalar-like
    // if !is_scalar_like(b) {
    //     return Err(OrtError::ShapeMismatch("Exponent must be a scalar-like value".into()));
    // }

    // Get the scalar value of b
    // let b_value = get_scalar_value(b)?;
    match (a, b) {
        (ArrayDResult::Float(_), ArrayDResult::Float(_)) => {
            let mut raiseto;
            let mut array;
            let mut raisetoval;
            if is_scalar_like(a){
                raiseto=a;
                array=b;
                
            }
            else if is_scalar_like(b){
                raiseto=b;
                array=a;
            }
            else{
                return Err(OrtError::TypeMismatch(("")))
            };
            if let ArrayDResult::Float(raisetowhat)=raiseto{
                raisetoval=raisetowhat.first().unwrap().clone();
            }
            else{
                return Err(OrtError::TypeMismatch(""))
            }

            if let ArrayDResult::Float(arrayinp)=array{
                let result = arrayinp.mapv(|x| x.powf(raisetoval));
                return Ok(ArrayDResult::Float(result))
            }
            else{
                return Err(OrtError::TypeMismatch(("")))
            }
        }
        (ArrayDResult::Int64(_), ArrayDResult::Int64(_)) => {
            let mut raiseto;
            let mut array;
            let mut raisetoval;
            if is_scalar_like(a){
                raiseto=a;
                array=b;
                
            }
            else if is_scalar_like(b){
                raiseto=b;
                array=a;
            }
            else{
                return Err(OrtError::TypeMismatch(("")))
            };
            if let ArrayDResult::Int64(raisetowhat)=raiseto{
                raisetoval=raisetowhat.first().unwrap().clone();
            }
            else{
                return Err(OrtError::TypeMismatch(("")))
            }

            if let ArrayDResult::Int64(arrayinp)=array{
                let result = arrayinp.mapv(|x| x.pow(raisetoval as u32));
                return Ok(ArrayDResult::Int64(result))
            }
            else{
                return Err(OrtError::TypeMismatch(("")))
            }
        }
        (ArrayDResult::Int32(_), ArrayDResult::Int32(_)) => {
            let mut raiseto;
            let mut array;
            let mut raisetoval;
            if is_scalar_like(a){
                raiseto=a;
                array=b;
                
            }
            else if is_scalar_like(b){
                raiseto=b;
                array=a;
            }
            else{
                return Err(OrtError::TypeMismatch(("")))
            };
            if let ArrayDResult::Int32(raisetowhat)=raiseto{
                raisetoval=raisetowhat.first().unwrap().clone();
            }
            else{
                return Err(OrtError::TypeMismatch(("")))
            }

            if let ArrayDResult::Int32(arrayinp)=array{
                let result = arrayinp.mapv(|x| x.pow(raisetoval as u32));
                return Ok(ArrayDResult::Int32(result))
            }
            else{
                return Err(OrtError::TypeMismatch(("")))
            }
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
            let retstr=format!("{}{}",a_type.to_string(),
                b_type.to_string());
                println!("{}",retstr);
            Err(OrtError::TypeMismatch(""))
        }
    }
}

pub fn sqrt_array(a: &ArrayDResult) -> Result<ArrayDResult, OrtError> {
    match (a) {
        ArrayDResult::Float(arrayinp) => {
                let result = arrayinp.mapv(|x| x.sqrt());
                return Ok(ArrayDResult::Float(result))
        }
        ArrayDResult::Int32(arrayinp) => {
                let result = arrayinp.mapv(|x| x.isqrt());
                return Ok(ArrayDResult::Int32(result))
        }
        ArrayDResult::Int64(arrayinp) => {
                let result = arrayinp.mapv(|x| x.isqrt());
                return Ok(ArrayDResult::Int64(result))
        }
        
        _ => {
            Err(OrtError::TypeMismatch(""))
        }
    }
}

// Function to compute a.pow(b) for ArrayDResult types
// pub fn pow_array(a: &ArrayDResult, b: &ArrayDResult) -> Result<ArrayDResult, OrtError> {
//     // Helper function to check if an ArrayDResult is scalar-like (0-dimensional or single element)
//     fn is_scalar_like(arr: &ArrayDResult) -> bool {
//         match arr {
//             ArrayDResult::Float(arr) => arr.len() == 1,
//             ArrayDResult::Int64(arr) => arr.len() == 1,
//             ArrayDResult::Int32(arr) => arr.len() == 1,
//         }
//     }

//     // Extract scalar value from b if scalar-like
//     // fn get_scalar_value(b: &ArrayDResult) -> Result<f64, OrtError> {
//         let b_value: &f32=match b {
//             ArrayDResult::Float(arr) if arr.len() == 1 => Ok(arr.first().unwrap().into()),
//             ArrayDResult::Int64(arr) if arr.len() == 1 => Ok(arr.first().unwrap().into() as &f32),
//             ArrayDResult::Int32(arr) if arr.len() == 1 => Ok(arr.first().unwrap().into() as &f32),
//             _ => Err(()
//                 // OrtError::ShapeMismatch("Exponent must be a scalar-like value".into())
//             ),
//         }.unwrap();
//     // }

//     // Ensure b is scalar-like
//     // if !is_scalar_like(b) {
//     //     return Err(OrtError::ShapeMismatch("Exponent must be a scalar-like value".into()));
//     // }

//     // Get the scalar value of b
//     // let b_value = get_scalar_value(b)?;

//     match (a, b) {
//         (ArrayDResult::Float(arr_a), _) => {
//             // For Float, we can use powf for fractional or integer exponents
//             let result = arr_a.mapv(|x| x.powf(b_value as f32));
//             Ok(ArrayDResult::Float(result))
//         }
//         (ArrayDResult::Int64(arr_a), ArrayDResult::Int64(_)) => {
//             // For Int64, ensure b is an integer
//             let b_int = b_value as i64;
//             if ((b_value - (b_int as f32)) as f64).abs() > f64::EPSILON {
//                 return Err(OrtError::TypeMismatch(
//                     "Int64 arrays require integer exponents".into(),
//                 ));
//             }
//             if b_int < 0 {
//                 return Err(OrtError::TypeMismatch(
//                     "Negative exponents not supported for Int64 arrays".into(),
//                 ));
//             }
//             let result = arr_a.mapv(|x| x.pow(b_int as u32));
//             Ok(ArrayDResult::Int64(result))
//         }
//         (ArrayDResult::Int32(arr_a), ArrayDResult::Int32(_) | ArrayDResult::Int64(_)) => {
//             // For Int32, ensure b is an integer
//             let b_int = b_value as i32;
//             if ((b_value - (b_int as f32)) as f64).abs() > f64::EPSILON {
//                 return Err(OrtError::TypeMismatch(
//                     "Int32 arrays require integer exponents".into(),
//                 ));
//             }
//             if b_int < 0 {
//                 return Err(OrtError::TypeMismatch(
//                     "Negative exponents not supported for Int32 arrays".into(),
//                 ));
//             }
//             let result = arr_a.mapv(|x| x.pow(b_int as u32));
//             Ok(ArrayDResult::Int32(result))
//         }
//         _ => Err(OrtError::TypeMismatch(
//             "Incompatible types for power operation".into(),
//         )),
//     }
// }

use num_traits::{Pow, Float}; // Needed for power and other float operations

// Implement power operation
impl ArrayDResult {
    // pub fn len(self)->usize{
        
    //     match self {
    //         ArrayDResult::Float(a) => a.len(),
    //         ArrayDResult::Int64(a) => a.len(),
    //         ArrayDResult::Int32(a) => a.len(),
    //     }
    // }
    // pub fn pow(self, exponent: f32) -> Result<Self, ArrayDResultError> {
    //     match self {
    //         ArrayDResult::Float(a) => Ok(ArrayDResult::Float(
    //             a.mapv(|x| x.powf(exponent))
    //         )),
    //         ArrayDResult::Int64(_) => Err(ArrayDResultError::MismatchedVariants(
    //             "Int64".to_string(),
    //             "pow requires Float".to_string(),
    //         )),
    //         ArrayDResult::Int32(_) => Err(ArrayDResultError::MismatchedVariants(
    //             "Int32".to_string(),
    //             "pow requires Float".to_string(),
    //         )),
    //     }
    // }

    // pub fn sqrt(self) -> Result<Self, ArrayDResultError> {
    //     match self {
    //         ArrayDResult::Float(a) => {
    //             if a.iter().any(|&x| x < 0.0) {
    //                 return Err(ArrayDResultError::MismatchedVariants(
    //                     "Float".to_string(),
    //                     "sqrt requires non-negative values".to_string(),
    //                 ));
    //             }
    //             Ok(ArrayDResult::Float(a.mapv(|x| x.sqrt())))
    //         }
    //         ArrayDResult::Int64(_) => Err(ArrayDResultError::MismatchedVariants(
    //             "Int64".to_string(),
    //             "sqrt requires Float".to_string(),
    //         )),
    //         ArrayDResult::Int32(_) => Err(ArrayDResultError::MismatchedVariants(
    //             "Int32".to_string(),
    //             "sqrt requires Float".to_string(),
    //         )),
    //     }
    // }

    // pub fn exp(self) -> Result<Self, ArrayDResultError> {
    //     match self {
    //         ArrayDResult::Float(a) => Ok(ArrayDResult::Float(
    //             a.mapv(|x| x.exp())
    //         )),
    //         ArrayDResult::Int64(_) => Err(ArrayDResultError::MismatchedVariants(
    //             "Int64".to_string(),
    //             "exp requires Float".to_string(),
    //         )),
    //         ArrayDResult::Int32(_) => Err(ArrayDResultError::MismatchedVariants(
    //             "Int32".to_string(),
    //             "exp requires Float".to_string(),
    //         )),
    //     }
    // }
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



#[cfg(test)]
mod tests {

    use super::*; // Import the parent module's items (pow_array, ArrayDResult, OrtError)
    use ndarray::{ArrayD, IxDyn};

    // Helper function to create ArrayDResult from a Vec and shape
    fn create_float_array(data: Vec<f32>, shape: &[usize]) -> ArrayDResult {
        ArrayDResult::Float(
            ArrayD::from_shape_vec(IxDyn(shape), data).expect("Failed to create float array"),
        )
    }

    fn create_int64_array(data: Vec<i64>, shape: &[usize]) -> ArrayDResult {
        ArrayDResult::Int64(
            ArrayD:: from_shape_vec(IxDyn(shape), data).expect("Failed to create int64 array"),
        )
    }

    fn create_int32_array(data: Vec<i32>, shape: &[usize]) -> ArrayDResult {
        ArrayDResult::Int32(
            ArrayD::from_shape_vec(IxDyn(shape), data).expect("Failed to create int32 array"),
        )
    }

    #[test]
    fn test_float_base_float_exponent() {
        let a = create_float_array(vec![2.0, 3.0, 4.0], &[3]); // Array [2.0, 3.0, 4.0]
        let b = create_float_array(vec![2.0], &[1]); // Scalar 2.0
        let result = pow_array(&a, &b).expect("Expected successful pow operation");
        
        if let ArrayDResult::Float(arr) = result {
            assert_eq!(arr, ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 9.0, 16.0]).unwrap());
        } else {
            panic!("Expected Float result");
        }
    }

    #[test]
    fn test_float_exponent_float_base() {
        let a = create_float_array(vec![2.0], &[1]); // Scalar 2.0
        let b = create_float_array(vec![2.0, 3.0, 4.0], &[3]); // Array [2.0, 3.0, 4.0]
        let result = pow_array(&a, &b).expect("Expected successful pow operation");
        println!("-----------------{:?}",result);
        if let ArrayDResult::Float(arr) = result {
            assert_eq!(arr, ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 9.0, 16.0]).unwrap());
        } else {
            panic!("Expected Float result");
        }
    }

    #[test]
    fn test_int64_base_int64_exponent() {
        let a = create_int64_array(vec![2, 3, 4], &[3]); // Array [2, 3, 4]
        let b = create_int64_array(vec![2], &[1]); // Scalar 2
        let result = pow_array(&a, &b).expect("Expected successful pow operation");
        if let ArrayDResult::Int64(arr) = result {
            assert_eq!(arr, ArrayD::from_shape_vec(IxDyn(&[3]), vec![4, 9, 16]).unwrap());
        } else {
            panic!("Expected Int64 result");
        }
    }

    #[test]
    fn test_int64_exponent_int64_base() {
        let a = create_int64_array(vec![2], &[1]); // Scalar 2
        let b = create_int64_array(vec![2, 3, 4], &[3]); // Array [2, 3, 4]
        let result = pow_array(&a, &b).expect("Expected successful pow operation");
        if let ArrayDResult::Int64(arr) = result {
            assert_eq!(arr, ArrayD::from_shape_vec(IxDyn(&[3]), vec![4, 9, 16]).unwrap());
        } else {
            panic!("Expected Int64 result");
        }
    }

    #[test]
    fn test_int32_base_int32_exponent() {
        let a = create_int32_array(vec![2, 3, 4], &[3]); // Array [2, 3, 4]
        let b = create_int32_array(vec![2], &[1]); // Scalar 2
        let result = pow_array(&a, &b).expect("Expected successful pow operation");
        if let ArrayDResult::Int32(arr) = result {
            assert_eq!(arr, ArrayD::from_shape_vec(IxDyn(&[3]), vec![4, 9, 16]).unwrap());
        } else {
            panic!("Expected Float result");
        }
    }

    #[test]
    fn test_int32_exponent_int32_base() {
        let a = create_int32_array(vec![2], &[1]); // Scalar 2
        let b = create_int32_array(vec![2, 3, 4], &[3]); // Array [2, 3, 4]
        let result = pow_array(&a, &b).expect("Expected successful pow operation");
        if let ArrayDResult::Int32(arr) = result {
            assert_eq!(arr, ArrayD::from_shape_vec(IxDyn(&[3]), vec![4, 9, 16]).unwrap());
        } else {
            panic!("Expected Float result");
        }
    }

    #[test]
    fn test_type_mismatch_float_int64() {
        let a = create_float_array(vec![2.0, 3.0], &[2]); // Float array
        let b = create_int64_array(vec![2], &[1]); // Int64 scalar
        let result = pow_array(&a, &b);
        assert!(matches!(result, Err(OrtError::TypeMismatch(_))));
    }

    #[test]
    fn test_type_mismatch_int64_float() {
        let a = create_int64_array(vec![2, 3], &[2]); // Int64 array
        let b = create_float_array(vec![2.0], &[1]); // Float scalar
        let result = pow_array(&a, &b);
        assert!(matches!(result, Err(OrtError::TypeMismatch(_))));
    }

    #[test]
    fn test_type_mismatch_int32_float() {
        let a = create_int32_array(vec![2, 3], &[2]); // Int32 array
        let b = create_float_array(vec![2.0], &[1]); // Float scalar
        let result = pow_array(&a, &b);
        assert!(matches!(result, Err(OrtError::TypeMismatch(_))));
    }

    #[test]
    fn test_type_mismatch_float_int32() {
        let a = create_float_array(vec![2.0, 3.0], &[2]); // Float array
        let b = create_int32_array(vec![2], &[1]); // Int32 scalar
        let result = pow_array(&a, &b);
        assert!(matches!(result, Err(OrtError::TypeMismatch(_))));
    }

    #[test]
    fn test_non_scalar_exponent() {
        let a = create_float_array(vec![2.0, 3.0], &[2]); // Float array
        let b = create_float_array(vec![2.0, 3.0], &[2]); // Non-scalar float array
        let result = pow_array(&a, &b);
        println!("----------{:?}",result);
        assert!(matches!(result, Err(OrtError::TypeMismatch(_))));
    }

    #[test]
    fn test_zero_exponent_float() {
        let a = create_float_array(vec![2.0, 3.0, 4.0], &[3]); // Array [2.0, 3.0, 4.0]
        let b = create_float_array(vec![0.0], &[1]); // Scalar 0.0
        let result = pow_array(&a, &b).expect("Expected successful pow operation");
        if let ArrayDResult::Float(arr) = result {
            assert_eq!(arr, ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 1.0, 1.0]).unwrap());
        } else {
            panic!("Expected Float result");
        }
    }

    #[test]
    fn test_zero_exponent_int64() {
        let a = create_int64_array(vec![2, 3, 4], &[3]); // Array [2, 3, 4]
        let b = create_int64_array(vec![0], &[1]); // Scalar 0
        let result = pow_array(&a, &b).expect("Expected successful pow operation");
        println!("-----------{:?}",result);

        if let ArrayDResult::Int64(arr) = result {
            assert_eq!(arr, ArrayD::from_shape_vec(IxDyn(&[3]), vec![1, 1, 1]).unwrap());
        } else {
            panic!("Expected Int64 result");
        }
    }

    #[test]
    fn test_zero_exponent_int32() {
        let a = create_int32_array(vec![2, 3, 4], &[3]); // Array [2, 3, 4]
        let b = create_int32_array(vec![0], &[1]); // Scalar 0
        let result = pow_array(&a, &b).expect("Expected successful pow operation");
        println!("-----------{:?}",result);
        if let ArrayDResult::Int32(arr) = result {
            assert_eq!(arr, ArrayD::from_shape_vec(IxDyn(&[3]), vec![1, 1, 1]).unwrap());
        } else {
            panic!("Expected Float result");
        }
    }
    
}