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
    Boolean(ArrayD<bool>),
}

pub fn pow_array(a: &ArrayDResult, b: &ArrayDResult) -> Result<ArrayDResult, OrtError> {
        fn is_scalar_like(arr: &ArrayDResult) -> bool {
        match arr {
            ArrayDResult::Float(arr) => arr.len() == 1,
            ArrayDResult::Int64(arr) => arr.len() == 1,
            ArrayDResult::Int32(arr) => arr.len() == 1,
            _=>false
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
                return Err(OrtError::TypeMismatch(("".to_string())))
            };
            if let ArrayDResult::Float(raisetowhat)=raiseto{
                raisetoval=raisetowhat.first().unwrap().clone();
            }
            else{
                return Err(OrtError::TypeMismatch("".to_string()))
            }

            if let ArrayDResult::Float(arrayinp)=array{
                let result = arrayinp.mapv(|x| x.powf(raisetoval));
                return Ok(ArrayDResult::Float(result))
            }
            else{
                return Err(OrtError::TypeMismatch(("".to_string())))
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
                return Err(OrtError::TypeMismatch(("").to_string()))
            };
            if let ArrayDResult::Int64(raisetowhat)=raiseto{
                raisetoval=raisetowhat.first().unwrap().clone();
            }
            else{
                return Err(OrtError::TypeMismatch(("".to_string())))
            }

            if let ArrayDResult::Int64(arrayinp)=array{
                let result = arrayinp.mapv(|x| x.pow(raisetoval as u32));
                return Ok(ArrayDResult::Int64(result))
            }
            else{
                return Err(OrtError::TypeMismatch(("".to_string())))
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
                return Err(OrtError::TypeMismatch(("".to_string())))
            };
            if let ArrayDResult::Int32(raisetowhat)=raiseto{
                raisetoval=raisetowhat.first().unwrap().clone();
            }
            else{
                return Err(OrtError::TypeMismatch(("".to_string())))
            }

            if let ArrayDResult::Int32(arrayinp)=array{
                let result = arrayinp.mapv(|x| x.pow(raisetoval as u32));
                return Ok(ArrayDResult::Int32(result))
            }
            else{
                return Err(OrtError::TypeMismatch(("".to_string())))
            }
        }
        (a, b) => {
            let a_type = match a {
                ArrayDResult::Float(_) => "Float",
                ArrayDResult::Int64(_) => "Int64",
                ArrayDResult::Int32(_) => "Int32",
                _=>"ERROR"
            };
            let b_type = match b {
                ArrayDResult::Float(_) => "Float",
                ArrayDResult::Int64(_) => "Int64",
                ArrayDResult::Int32(_) => "Int32",
                _=>"ERROR"
            };
            let retstr=format!("{}{}",a_type.to_string(),
                b_type.to_string());
                println!("{}",retstr);
            Err(OrtError::TypeMismatch("".to_string()))
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
            Err(OrtError::TypeMismatch("".to_string()))
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
                             _=>"ERROR"

                        };
                        let b_type = match b {
                            ArrayDResult::Float(_) => "Float",
                            ArrayDResult::Int64(_) => "Int64",
                            ArrayDResult::Int32(_) => "Int32",
                            _=>"ERROR"

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
                    _=>"ERROR"

                };
                let b_type = match b {
                    ArrayDResult::Float(_) => "Float",
                    ArrayDResult::Int64(_) => "Int64",
                    ArrayDResult::Int32(_) => "Int32",
                    _=>"ERROR"

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
        ArrayDResult::Boolean(array_base) => {
            (array_base.shape().iter().map(|&n| Dimensions::Fixed(n)).collect(), array_base.into_raw_vec()
                            .into_iter()
                            .map(|x| if x { 1u8 } else { 0u8 })
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
                DataType::Boolean=>{
                let bool_data: Vec<bool> = data
                                        .iter()
                                        .map(|&b| b != 0)
                                        .collect();
                    ArrayD::from_shape_vec(IxDyn(&concrete_shape), bool_data)
                        .map(|arr| ArrayDResult::Boolean(arr))
                        .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for boolean tensor".into()))
                    
                }
                _ => Err(OrtError::TypeMismatch("Unsupported tensor type, expected Float, Int64, or Int32".to_string())),
            }
        }
        _ => Err(OrtError::TypeMismatch("Expected tensor".to_string())),
    }
}



#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    use crate::{AttributeProto, OrtEngine, TensorProto};

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
    

    // Helper function to create an OrtValue boolean tensor from a Vec<bool>
    fn create_bool_tensor(data: Vec<bool>, shape: Vec<usize>) -> OrtValue {
        let bytes: Vec<u8> = data.into_iter().map(|b| b as u8).collect();
        OrtValue::Tensor {
            shape: shape.into_iter().map(Dimensions::Fixed).collect(),
            dtype: DataType::Boolean,
            data: Arc::new(bytes),
        }
    }

    #[test]
    fn test_op_and_valid_inputs() {
        // Test case 1: Two boolean tensors with same shape
        let input1 = create_bool_tensor(vec![true, false, true, true], vec![2, 2]);
        let input2 = create_bool_tensor(vec![true, true, false, true], vec![2, 2]);
        let inputs = vec![input1, input2];

        let result = OrtEngine::op_and(&NodeProto::default(), &inputs);
        let result2 = OrtEngine::op_and(&NodeProto::default(), &inputs);
        println!("{:?}",result);
        assert!(result.is_ok(), "Expected successful operation, got {:?}", result);

        // Verify the result (assuming ndarray_to_ort preserves the boolean array)
        let ora=ort_to_ndarray(&result2.unwrap()).unwrap();
        println!("{:?}",ora);

        if let Ok(OrtValue::Tensor { data, dtype, shape, .. }) = result {
            assert_eq!(dtype, DataType::Boolean);
            assert_eq!(shape, vec![Dimensions::Fixed(2), Dimensions::Fixed(2)]);
            let expected_data: Vec<u8> = vec![true, false, false, true].into_iter().map(|b| b as u8).collect();
            println!("{:?}",expected_data);
            assert_eq!(*data, expected_data);
        }
    }

    #[test]
    fn test_op_and_invalid_input_count() {
        // Test case 2: Too few inputs (0)
        let inputs: Vec<OrtValue> = vec![];
        let result = OrtEngine::op_and(&NodeProto::default(), &inputs);
        assert!(matches!(
            result,
            Err(OrtError::TypeMismatch(ref msg)) if msg == &"And requires exactly two boolean tensors"
        ));

        // Test case 3: Too many inputs (3)
        let input1 = create_bool_tensor(vec![true], vec![1]);
        let input2 = create_bool_tensor(vec![true], vec![1]);
        let input3 = create_bool_tensor(vec![true], vec![1]);
        let inputs = vec![input1, input2, input3];
        let result = OrtEngine::op_and(&NodeProto::default(), &inputs);
        assert!(matches!(
            result,
            Err(OrtError::TypeMismatch(ref msg)) if msg == &"And requires exactly two boolean tensors"
        ));
    }

    #[test]
    fn test_op_and_type_mismatch() {
        // Test case 4: First input is not boolean
        let input1 = OrtValue::Tensor {
            shape: vec![Dimensions::Fixed(1)],
            dtype: DataType::Float,
            data: Arc::new(vec![0u8; 4]), // Dummy float data
        };
        let input2 = create_bool_tensor(vec![true], vec![1]);
        let inputs = vec![input1, input2];
        let result = OrtEngine::op_and(&NodeProto::default(), &inputs);
        assert!(matches!(
            result,
            Err(OrtError::TypeMismatch(ref msg)) if msg == &"First input must be a boolean tensor"
        ));

        // Test case 5: Second input is not boolean
        let input1 = create_bool_tensor(vec![true], vec![1]);
        let input2 = OrtValue::Tensor {
            shape: vec![Dimensions::Fixed(1)],
            dtype: DataType::Int32,
            data: Arc::new(vec![0u8; 4]), // Dummy int32 data
        };
        let inputs = vec![input1, input2];
        let result = OrtEngine::op_and(&NodeProto::default(), &inputs);
        assert!(matches!(
            result,
            Err(OrtError::TypeMismatch(ref msg)) if msg == &"Second input must be a boolean tensor"
        ));
    }

    #[test]
    fn test_op_and_empty_tensors() {
        // Test case 6: Empty boolean tensors
        let input1 = create_bool_tensor(vec![], vec![0]);
        let input2 = create_bool_tensor(vec![], vec![0]);
        let inputs = vec![input1, input2];
        let result = OrtEngine::op_and(&NodeProto::default(), &inputs);
        assert!(result.is_ok(), "Expected successful operation for empty tensors, got {:?}", result);

        if let Ok(OrtValue::Tensor { data, dtype, shape, .. }) = result {
            assert_eq!(dtype, DataType::Boolean);
            assert_eq!(shape, vec![Dimensions::Fixed(0)]);
            assert!(data.is_empty());
        }
    }

    // Note: Broadcasting tests are commented out since the broadcasting code is commented in the function
    // #[test]
    // fn test_op_and_broadcasting() {
    //     // Test case 7: Broadcasting with compatible shapes
    //     let input1 = create_bool_tensor(vec![true, false], vec![2, 1]);
    //     let input2 = create_bool_tensor(vec![true], vec![1]);
    //     let inputs = vec![input1, input2];
    //     let result = op_and(&NodeProto {}, &inputs);
    //     assert!(result.is_ok(), "Expected successful broadcasting, got {:?}", result);
    //
    //     if let Ok(OrtValue::Tensor { data, dtype, shape, .. }) = result {
    //         assert_eq!(dtype, DataType::Boolean);
    //         assert_eq!(shape, vec![Dimensions::Fixed(2), Dimensions::Fixed(1)]);
    //         let expected_data: Vec<u8> = vec![true, false].into_iter().map(|b| b as u8).collect();
    //         assert_eq!(*data, expected_data);
    //     }
    // }


    #[test]
    fn test_cast_float_to_int64() {
        // Setup: NodeProto with to=7 (Int64), input tensor with Float data
        let node = NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Cast".to_string(),
            attributes: vec![AttributeProto {
                name: "to".to_string(),
                i: 7, // Cast to Int64
                ..Default::default()
            }],
            name: "cast_node".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let float_data = vec![1.5f32, 2.7f32, -3.2f32];
        let data: Vec<u8> = float_data
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let input = OrtValue::Tensor {
            shape: vec![Dimensions::Fixed(3)],
            dtype: DataType::Float,
            data: Arc::new(data),
        };

        // Execute
        let result = OrtEngine::op_cast(&node, &[input]).unwrap();

        // Verify
        match result {
            OrtValue::Tensor { shape, dtype, data } => {
                assert_eq!(shape, vec![Dimensions::Fixed(3)]);
                assert_eq!(dtype, DataType::Int64);
                let int_data: Vec<i64> = data
                    .chunks(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                assert_eq!(int_data, vec![1, 2, -3]); // 1.5 -> 1, 2.7 -> 2, -3.2 -> -3
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_cast_int64_to_float() {
        // Setup: NodeProto with to=1 (Float), input tensor with Int64 data
        let node = NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Cast".to_string(),
            attributes: vec![AttributeProto {
                name: "to".to_string(),
                i: 1, // Cast to Float
                ..Default::default()
            }],
            name: "cast_node".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let int_data = vec![1i64, -2i64, 3i64];
        let data: Vec<u8> = int_data
            .into_iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let input = OrtValue::Tensor {
            shape: vec![Dimensions::Fixed(3)],
            dtype: DataType::Int64,
            data: Arc::new(data),
        };

        // Execute
        let result = OrtEngine::op_cast(&node, &[input]).unwrap();

        // Verify
        match result {
            OrtValue::Tensor { shape, dtype, data } => {
                assert_eq!(shape, vec![Dimensions::Fixed(3)]);
                assert_eq!(dtype, DataType::Float);
                let float_data: Vec<f32> = data
                    .chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                assert_eq!(float_data, vec![1.0, -2.0, 3.0]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_cast_float_to_boolean() {
        // Setup: NodeProto with to=9 (Boolean), input tensor with Float data
        let node = NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Cast".to_string(),
            attributes: vec![AttributeProto {
                name: "to".to_string(),
                i: 9, // Cast to Boolean
                ..Default::default()
            }],
            name: "cast_node".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let float_data = vec![0.0f32, 1.5f32, -2.0f32];
        let data: Vec<u8> = float_data
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let input = OrtValue::Tensor {
            shape: vec![Dimensions::Fixed(3)],
            dtype: DataType::Float,
            data: Arc::new(data),
        };

        // Execute
        let result = OrtEngine::op_cast(&node, &[input]).unwrap();

        // Verify
        match result {
            OrtValue::Tensor { shape, dtype, data } => {
                assert_eq!(shape, vec![Dimensions::Fixed(3)]);
                assert_eq!(dtype, DataType::Boolean);
                let bool_data: Vec<bool> = data.iter().map(|&b| b != 0).collect();
                assert_eq!(bool_data, vec![false, true, true]); // 0.0 -> false, non-zero -> true
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_cast_boolean_to_float() {
        // Setup: NodeProto with to=1 (Float), input tensor with Boolean data
        let node = NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Cast".to_string(),
            attributes: vec![AttributeProto {
                name: "to".to_string(),
                i: 1, // Cast to Float
                ..Default::default()
            }],
            name: "cast_node".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let bool_data = vec![0u8, 1u8, 0u8]; // false, true, false
        let input = OrtValue::Tensor {
            shape: vec![Dimensions::Fixed(3)],
            dtype: DataType::Boolean,
            data: Arc::new(bool_data),
        };

        // Execute
        let result = OrtEngine::op_cast(&node, &[input]).unwrap();

        // Verify
        match result {
            OrtValue::Tensor { shape, dtype, data } => {
                assert_eq!(shape, vec![Dimensions::Fixed(3)]);
                assert_eq!(dtype, DataType::Float);
                let float_data: Vec<f32> = data
                    .chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                assert_eq!(float_data, vec![0.0, 1.0, 0.0]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_missing_to_attribute() {
        // Setup: NodeProto without 'to' attribute
        let node = NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Cast".to_string(),
            attributes: vec![], // No 'to' attribute
            name: "cast_node".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let input = OrtValue::Tensor {
            shape: vec![Dimensions::Fixed(1)],
            dtype: DataType::Float,
            data: Arc::new(vec![0.0f32.to_le_bytes().to_vec()].concat()),
        };

        // Execute
        let result = OrtEngine::op_cast(&node, &[input]);

        // Verify
        assert!(matches!(
            result,
            Err(OrtError::InvalidTensorData(msg)) if msg == "Cast requires 'to' attribute"
        ));
    }

    #[test]
    fn test_unsupported_target_type() {
        // Setup: NodeProto with unsupported 'to' value (e.g., 999)
        let node = NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Cast".to_string(),
            attributes: vec![AttributeProto {
                name: "to".to_string(),
                i: 999, // Unsupported type
                ..Default::default()
            }],
            name: "cast_node".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let input = OrtValue::Tensor {
            shape: vec![Dimensions::Fixed(1)],
            dtype: DataType::Float,
            data: Arc::new(vec![0.0f32.to_le_bytes().to_vec()].concat()),
        };

        // Execute
        let result = OrtEngine::op_cast(&node, &[input]);

        // Verify
        assert!(matches!(
            result,
            Err(OrtError::TypeMismatch(msg)) if msg == "Unsupported cast to type"
        ));
    }

    #[test]
    fn test_non_tensor_input() {
        // Setup: NodeProto with valid 'to' but non-tensor input
        let node = NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Cast".to_string(),
            attributes: vec![AttributeProto {
                name: "to".to_string(),
                i: 1, // Cast to Float
                ..Default::default()
            }],
            name: "cast_node".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let input = OrtValue::Sequence(vec![]); // Non-tensor input

        // Execute
        let result = OrtEngine::op_cast(&node, &[input]);

        // Verify
        assert!(matches!(
            result,
            Err(OrtError::TypeMismatch(msg)) if msg == "Input must be a tensor"
        ));
    }

    #[test]
    fn test_empty_input() {
        // Setup: NodeProto with valid 'to' but no inputs
        let node = NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Cast".to_string(),
            attributes: vec![AttributeProto {
                name: "to".to_string(),
                i: 1, // Cast to Float
                ..Default::default()
            }],
            name: "cast_node".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };

        // Execute
        let result = OrtEngine::op_cast(&node, &[]);

        // Verify
        assert!(matches!(
            result,
            Err(OrtError::TypeMismatch(msg)) if msg == "Cast requires one tensor"
        ));
    }
    
    #[test]
    fn test_op_clip_float() {
        let input_data = vec![-1.0f32, 0.0, 1.0, 5.0, 10.0];
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![5],
            DataType::Float,
        );
        // Create scalar tensors for min and max (shape should be [] not [1])
        let min_data = vec![0.0f32];
        let min = create_ort_tensor(
            min_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![],  // Empty shape for scalar
            DataType::Float,
        );
        let max_data = vec![3.0f32];
        let max = create_ort_tensor(
            max_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![],  // Empty shape for scalar
            DataType::Float,
        );

        let node = NodeProto ::default();
        let inputs = vec![input, min, max];
        let result = OrtEngine::op_clip(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();
        println!("{:?}",result_array);
        let expected = ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.0, 0.0, 1.0, 3.0, 3.0]).unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    #[test]
    fn test_op_clip_int32_no_min_max() {
        let input_data = vec![-1i32, 0, 1, 5, 10];
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![5],
            DataType::Int32,
        );

        let node = NodeProto::default();
        let inputs = vec![input];
        let result = OrtEngine::op_clip(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        let expected =
            ArrayD::from_shape_vec(IxDyn(&[5]), vec![-1i32, 0, 1, 5, 10]).unwrap();
        match result_array {
            ArrayDResult::Int32(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected int32 array"),
        }
    }

    // Test op_concat
    #[test]
    fn test_op_concat_float_axis_0() {
        let input1_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input1 = create_ort_tensor(
            input1_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2, 2],
            DataType::Float,
        );
        let input2_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let input2 = create_ort_tensor(
            input2_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2, 2],
            DataType::Float,
        );

        let node = NodeProto {
            attributes: vec![AttributeProto {name:"axis".to_string(),i:0,s:vec![],ints:vec![],t:None, f: 0.0, floats: vec![], g: None,strings:vec![] }],
            input: vec![],
            output: vec![],
            op_type: "".to_string(),
            name: "".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let inputs = vec![input1, input2];
        let result = OrtEngine::op_concat(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        let expected = ArrayD::from_shape_vec(
            IxDyn(&[4, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    #[test]
    fn test_op_concat_negative_axis() {
        let input1_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input1 = create_ort_tensor(
            input1_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2, 2],
            DataType::Float,
        );
        let input2_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let input2 = create_ort_tensor(
            input2_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2, 2],
            DataType::Float,
        );

        let node = NodeProto {
            attributes: vec![AttributeProto {name:"axis".to_string(),i:-1,s:vec![],ints:vec![],t:None, f: 0.0, floats: vec![], g: None,strings:vec![] }],
            input: vec![],
            output: vec![],
            op_type: "".to_string(),
            name: "".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let inputs = vec![input1, input2];
        let result = OrtEngine::op_concat(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        let expected = ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
        )
        .unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    // Test op_constant_of_shape
    #[test]
    fn test_op_constant_of_shape_default() {
        let shape_data = vec![2i64, 3];
        let shape_tensor = create_ort_tensor(
            shape_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2],
            DataType::Int64,
        );

        let node = NodeProto {
            attributes: vec![],
            input: vec![],
            output: vec![],
            op_type: "".to_string(),
            name: "".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let inputs = vec![shape_tensor];
        let result = OrtEngine::op_constant_of_shape(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        let expected =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0f32; 6]).unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    #[test]
    fn test_op_constant_of_shape_with_value() {
        let shape_data = vec![2i64, 2];
        let shape_tensor = create_ort_tensor(
            shape_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2],
            DataType::Int64,
        );
        let mut value_tensor = TensorProto::default();
        value_tensor.data_type = 1; // Float
        value_tensor.dims = vec![1]; // Shape [1]
        value_tensor.float_data = vec![5.0]; // Value 5.0
        value_tensor.raw_data = vec![0, 0, 160, 64]; // 5.0f32 in little-endian bytes
    

        let node = NodeProto {
            attributes: vec![AttributeProto {name:"value".to_string(),i:0,s:vec![],ints:vec![],t:Some(value_tensor), f: 0.0, floats: vec![], g: None,strings:vec![] }],
            input: vec![],
            output: vec![],
            op_type: "".to_string(),
            name: "".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let inputs = vec![shape_tensor];
        let result = OrtEngine::op_constant_of_shape(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        let expected =
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![5.0f32; 4]).unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    // Test op_conv
    #[test]
    fn test_op_conv_2d_valid_padding() {
        let input_data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // Batch 1
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, // Batch 2
        ];
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2, 1, 3, 3], // Batch=2, Channels=1, Height=3, Width=3
            DataType::Float,
        );
    
        let weight_data = vec![1.0f32, 0.0, 0.0, 1.0];
        let weight = create_ort_tensor(
            weight_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![1, 1, 2, 2], // OutChannels=1, InChannels=1, Kernel=2x2
            DataType::Float,
        );
    
        let node = NodeProto {
            attributes: vec![
                AttributeProto {
                    name: "auto_pad".to_string(),
                    i: 0,
                    s: "VALID".as_bytes().to_vec(),
                    ints: vec![],
                    t: None,
                    f: 0.0,
                    floats: vec![],
                    g: None,
                    strings:vec![]
                },
                AttributeProto {
                    name: "strides".to_string(),
                    i: 0,
                    s: vec![],
                    ints: vec![1, 1],
                    t: None,
                    f: 0.0,
                    floats: vec![],
                    g: None,
                    strings:vec![]
                },
                AttributeProto {
                    name: "dilations".to_string(),
                    i: 0,
                    s: vec![],
                    ints: vec![1, 1],
                    t: None,
                    f: 0.0,
                    floats: vec![],
                    g: None,
                    strings:vec![]
                },
            ],
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Conv".to_string(),
            name: "conv_node".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let inputs = vec![input, weight];
        let result = OrtEngine::op_conv(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();
    
        // Expected output shape: [2, 1, 2, 2] (VALID padding, stride=1)
        // For input [[1,2,3],[4,5,6],[7,8,9]], kernel [[1,0],[0,1]]:
        // output[0,0,0,0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
        // output[0,0,0,1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
        // output[0,0,1,0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
        // output[0,0,1,1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
        let expected = ArrayD::from_shape_vec(
            IxDyn(&[2, 1, 2, 2]),
            vec![6.0, 8.0, 12.0, 14.0, 24.0, 26.0, 30.0, 32.0],
        )
        .unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }
    
    // Mock helper functions for testing
    fn create_ort_tensor(
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: DataType,
    ) -> OrtValue {
        OrtValue::Tensor {
            shape: shape.into_iter().map(|d| Dimensions::Fixed(d)).collect(),
            dtype,
            data: Arc::new(data),
        }
    }
    #[test]
    fn test_op_conv_2d_same_padding() {
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![1, 1, 3, 3],
            DataType::Float,
        );

        let weight_data = vec![1.0f32, 0.0, 0.0, 1.0];
        let weight = create_ort_tensor(
            weight_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![1, 1, 2, 2],
            DataType::Float,
        );

        let node = NodeProto {
            attributes: vec![
                AttributeProto {name:"auto_pad".to_string(),i:0,s:"SAME_UPPER".as_bytes().to_vec(),ints:vec![],t:None, f: 0.0, floats: vec![], g: None, strings:vec![] },
                AttributeProto {name:"strides".to_string(),i:0,s:vec![],ints:vec![1,1],t:None, f: 0.0, floats: vec![], g: None, strings:vec![] },
                AttributeProto {name:"dilations".to_string(),i:0,s:vec![],ints:vec![1,1],t:None, f: 0.0, floats: vec![], g: None, strings:vec![] },
            ],
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Conv".to_string(),
            name: "conv_node".to_string(),
            domain: "".to_string(),
            subgraphs: HashMap::new(),
        };
        let inputs = vec![input, weight];
        let result = OrtEngine::op_conv(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        // Expected output shape: [1, 1, 3, 3] (SAME_UPPER padding)
        let expected = ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3, 3]),
            vec![6.0, 8.0, 3.0, 12.0, 14.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    #[test]
    fn test_op_floor() {
        // Test with float tensor
        let input_data = vec![1.7f32, -2.3, 3.0, -4.9, 5.5];
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![5],
            DataType::Float,
        );

        let node = NodeProto::default();
        let inputs = vec![input];
        let result = OrtEngine::op_floor(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        let expected = ArrayD::from_shape_vec(IxDyn(&[5]), vec![1.0f32, -3.0, 3.0, -5.0, 5.0]).unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    #[test]
    fn test_op_floor_non_float_input() {
        // Test with int32 tensor (should fail)
        let input_data = vec![1i32, 2, 3, 4, 5];
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![5],
            DataType::Int32,
        );

        let node = NodeProto::default();
        let inputs = vec![input];
        let result = OrtEngine::op_floor(&node, &inputs);
        
        assert!(matches!(
            result,
            Err(OrtError::TypeMismatch(msg)) if msg == "Floor only supports float tensors"
        ));
    }

    #[test]
    fn test_op_floor_empty_input() {
        // Test with empty input list
        let node = NodeProto::default();
        let inputs = vec![];
        let result = OrtEngine::op_floor(&node, &inputs);
        
        assert!(matches!(
            result,
            Err(OrtError::TypeMismatch(msg)) if msg == "Floor requires one tensor"
        ));
    }

    #[test]
    fn test_op_floor_non_tensor_input() {
        // Test with non-tensor input
        let input = OrtValue::Sequence(vec![]);
        let node = NodeProto::default();
        let inputs = vec![input];
        let result = OrtEngine::op_floor(&node, &inputs);
        
        assert!(matches!(
            result,
            Err(OrtError::TypeMismatch(msg)) if msg == "Input must be a tensor"
        ));
    }
    
#[test]
fn test_op_round_float() {
    // Test with float tensor
    let input_data = vec![1.4f32, 1.5, 1.6, 2.5, 3.5, -1.5, -2.5];
    let input = create_ort_tensor(
        input_data
            .iter()
            .flat_map(|x| x.to_le_bytes().to_vec())
            .collect(),
        vec![7],
        DataType::Float,
    );

    let node = NodeProto::default();
    let inputs = vec![input];
    let result = OrtEngine::op_round(&node, &inputs).unwrap();
    let result_array = ort_to_ndarray(&result).unwrap();

    let expected = ArrayD::from_shape_vec(IxDyn(&[7]), vec![1.0f32, 2.0, 2.0, 2.0, 4.0, -2.0, -2.0]).unwrap();
    match result_array {
        ArrayDResult::Float(arr) => assert_eq!(arr, expected),
        _ => panic!("Expected float array"),
    }
}

#[test]
fn test_op_round_non_float_input() {
    // Test with int32 tensor (should fail)
    let input_data = vec![1i32, 2, 3, 4, 5];
    let input = create_ort_tensor(
        input_data
            .iter()
            .flat_map(|x| x.to_le_bytes().to_vec())
            .collect(),
        vec![5],
        DataType::Int32,
    );

    let node = NodeProto::default();
    let inputs = vec![input];
    let result = OrtEngine::op_round(&node, &inputs);
    
    assert!(matches!(
        result,
        Err(OrtError::TypeMismatch(msg)) if msg == "Round only supports float tensors"
    ));
}

#[test]
fn test_op_round_empty_input() {
    // Test with empty input list
    let node = NodeProto::default();
    let inputs = vec![];
    let result = OrtEngine::op_round(&node, &inputs);
    
    assert!(matches!(
        result,
        Err(OrtError::TypeMismatch(msg)) if msg == "Round requires one tensor"
    ));
}

#[test]
fn test_op_round_non_tensor_input() {
    // Test with non-tensor input
    let input = OrtValue::Sequence(vec![]);
    let node = NodeProto::default();
    let inputs = vec![input];
    let result = OrtEngine::op_round(&node, &inputs);
    
    assert!(matches!(
        result,
        Err(OrtError::TypeMismatch(msg)) if msg == "Input must be a tensor"
    ));
}

#[test]
fn test_op_round_special_values() {
    // Test with NaN, infinity, and zero
    let input_data = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
    let input = create_ort_tensor(
        input_data
            .iter()
            .flat_map(|x| x.to_le_bytes().to_vec())
            .collect(),
        vec![5],
        DataType::Float,
    );

    let node = NodeProto::default();
    let inputs = vec![input];
    let result = OrtEngine::op_round(&node, &inputs).unwrap();
    let result_array = ort_to_ndarray(&result).unwrap();

    match result_array {
        ArrayDResult::Float(arr) => {
            assert!(arr[0].is_nan());
            assert!(arr[1].is_infinite() && arr[1].is_sign_positive());
            assert!(arr[2].is_infinite() && arr[2].is_sign_negative());
            assert_eq!(arr[3], 0.0);
            assert_eq!(arr[4], -0.0);
        },
        _ => panic!("Expected float array"),
    }
}

    // Tests for op_slice
    #[test]
    fn test_op_slice_basic_2d() {
        // Test basic 2D slicing
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2, 3],
            DataType::Float,
        );

        // starts = [0, 1], ends = [2, 3], axes = [0, 1], steps = [1, 1]
        let starts = create_ort_tensor(
            vec![0i64, 1].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![2],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![2i64, 3].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![2],
            DataType::Int64,
        );
        let axes = create_ort_tensor(
            vec![0i64, 1].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![2],
            DataType::Int64,
        );
        let steps = create_ort_tensor(
            vec![1i64, 1].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![2],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends, axes, steps];
        let result = OrtEngine::op_slice(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        // Expected: slice from [0:2, 1:3] = [[2.0, 3.0], [5.0, 6.0]]
        let expected = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0f32, 3.0, 5.0, 6.0]).unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    #[test]
    fn test_op_slice_negative_indices() {
        // Test slicing with negative indices
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2, 3],
            DataType::Float,
        );

        // starts = [-2], ends = [-1], axes = [0], steps = [1]
        let starts = create_ort_tensor(
            vec![-2i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![-1i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let axes = create_ort_tensor(
            vec![0i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends, axes];
        let result = OrtEngine::op_slice(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        // Expected: slice from [-2:-1, :] = first row only
        let expected = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0f32, 2.0, 3.0]).unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    #[test]
    fn test_op_slice_with_steps() {
        // Test slicing with step > 1
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 1x8 array
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![1, 8],
            DataType::Float,
        );

        // starts = [0], ends = [8], axes = [1], steps = [2]
        let starts = create_ort_tensor(
            vec![0i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![8i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let axes = create_ort_tensor(
            vec![1i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let steps = create_ort_tensor(
            vec![2i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends, axes, steps];
        let result = OrtEngine::op_slice(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        // Expected: every 2nd element = [1.0, 3.0, 5.0, 7.0]
        let expected = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0f32, 3.0, 5.0, 7.0]).unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    #[test]
fn test_op_slice_negative_steps() {
    // Test slicing with negative steps (reverse)
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 1x6 array
    let input = create_ort_tensor(
        input_data
            .iter()
            .flat_map(|x| x.to_le_bytes().to_vec())
            .collect(),
        vec![1, 6],
        DataType::Float,
    );

    // Slice parameters: start at index 5, end at index 0 (exclusive), step by -2 along axis 1
    let starts = create_ort_tensor(
        vec![5i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
        vec![1],
        DataType::Int64,
    );
    let ends = create_ort_tensor(
        vec![0i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
        vec![1],
        DataType::Int64,
    );
    let axes = create_ort_tensor(
        vec![1i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
        vec![1],
        DataType::Int64,
    );
    let steps = create_ort_tensor(
        vec![-2i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
        vec![1],
        DataType::Int64,
    );

    let node = NodeProto::default(); // Ensure this is configured for Slice op
    let inputs = vec![input, starts, ends, axes, steps];
    let result = OrtEngine::op_slice(&node, &inputs)
        .expect("Slice operation failed");
    let result_array = ort_to_ndarray(&result)
        .expect("Failed to convert ORT tensor to ndarray");

    // Expected: reverse every 2nd element from index 5 to 0 = [6.0, 4.0, 2.0]
    let expected = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![6.0f32, 4.0, 2.0])
        .expect("Failed to create expected array");
    match result_array {
        ArrayDResult::Float(arr) => assert_eq!(arr, expected),
        _ => panic!("Expected float array, got {:?}", result_array),
    }
}

    #[test]
    fn test_op_slice_negative_steps_int32() {
        // Test slicing with negative steps (reverse) for Int32
        let input_data = vec![1i32, 2, 3, 4, 5, 6]; // 1x6 array
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![1, 6],
            DataType::Int32,
        );

        // Slice parameters: start at index 5, end at index 0 (exclusive), step by -2 along axis 1
        let starts = create_ort_tensor(
            vec![5i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![0i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let axes = create_ort_tensor(
            vec![1i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let steps = create_ort_tensor(
            vec![-2i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends, axes, steps];
        let result = OrtEngine::op_slice(&node, &inputs)
            .expect("Slice operation failed");
        let result_array = ort_to_ndarray(&result)
            .expect("Failed to convert ORT tensor to ndarray");

        // Expected: reverse every 2nd element from index 5 to 0 = [6, 4, 2]
        let expected = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![6i32, 4, 2])
            .expect("Failed to create expected array");
        match result_array {
            ArrayDResult::Int32(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected int32 array, got {:?}", result_array),
        }
    }

    #[test]
    fn test_op_slice_negative_steps_int64() {
        // Test slicing with negative steps (reverse) for Int64
        let input_data = vec![1i64, 2, 3, 4, 5, 6]; // 1x6 array
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![1, 6],
            DataType::Int64,
        );

        // Slice parameters: start at index 5, end at index 0 (exclusive), step by -2 along axis 1
        let starts = create_ort_tensor(
            vec![5i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![0i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let axes = create_ort_tensor(
            vec![1i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let steps = create_ort_tensor(
            vec![-2i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends, axes, steps];
        let result = OrtEngine::op_slice(&node, &inputs)
            .expect("Slice operation failed");
        let result_array = ort_to_ndarray(&result)
            .expect("Failed to convert ORT tensor to ndarray");

        // Expected: reverse every 2nd element from index 5 to 0 = [6, 4, 2]
        let expected = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![6i64, 4, 2])
            .expect("Failed to create expected array");
        match result_array {
            ArrayDResult::Int64(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected int64 array, got {:?}", result_array),
        }
    }

    #[test]
    fn test_op_slice_negative_steps_boolean() {
        // Test slicing with negative steps (reverse) for Boolean
        let input_data = vec![true, false, true, false, true, false]; // 1x6 array
        let input = create_ort_tensor(
            input_data.iter().map(|&b| b as u8).collect(),
            vec![1, 6],
            DataType::Boolean,
        );

        // Slice parameters: start at index 5, end at index 0 (exclusive), step by -2 along axis 1
        let starts = create_ort_tensor(
            vec![5i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![0i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let axes = create_ort_tensor(
            vec![1i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let steps = create_ort_tensor(
            vec![-2i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends, axes, steps];
        let result = OrtEngine::op_slice(&node, &inputs)
            .expect("Slice operation failed");
        let result_array = ort_to_ndarray(&result)
            .expect("Failed to convert ORT tensor to ndarray");

        // Expected: reverse every 2nd element from index 5 to 0 = [false, false, false]
        let expected = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![false, false, false])
            .expect("Failed to create expected array");
        match result_array {
            ArrayDResult::Boolean(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected boolean array, got {:?}", result_array),
        }
    }

    #[test]
    fn test_op_slice_int32_data() {
        // Test slicing with int32 data
        let input_data = vec![1i32, 2, 3, 4, 5, 6]; // 2x3 matrix
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2, 3],
            DataType::Int32,
        );

        // starts = [0], ends = [2], axes = [1], steps = [1]
        let starts = create_ort_tensor(
            vec![0i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![2i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let axes = create_ort_tensor(
            vec![1i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends, axes];
        let result = OrtEngine::op_slice(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        // Expected: slice [:, 0:2] = [[1, 2], [4, 5]]
        let expected = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1i32, 2, 4, 5]).unwrap();
        match result_array {
            ArrayDResult::Int32(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected int32 array"),
        }
    }

    #[test]
    fn test_op_slice_boolean_data() {
        // Test slicing with boolean data
        let input_data = vec![true, false, true, false, true, false]; // 2x3 matrix
        let input = create_ort_tensor(
            input_data.iter().map(|&b| b as u8).collect(),
            vec![2, 3],
            DataType::Boolean,
        );

        // starts = [0], ends = [1], axes = [0], steps = [1]
        let starts = create_ort_tensor(
            vec![0i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![1i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let axes = create_ort_tensor(
            vec![0i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends, axes];
        let result = OrtEngine::op_slice(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        // Expected: slice [0:1, :] = first row only = [true, false, true]
        let expected = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![true, false, true]).unwrap();
        match result_array {
            ArrayDResult::Boolean(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected boolean array"),
        }
    }

    #[test]
    fn test_op_slice_default_axes() {
        // Test slicing without providing axes (should default to [0, 1, ...])
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2, 3],
            DataType::Float,
        );

        // starts = [0, 1], ends = [1, 3] (no axes provided)
        let starts = create_ort_tensor(
            vec![0i64, 1].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![2],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![1i64, 3].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![2],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends]; // No axes or steps
        let result = OrtEngine::op_slice(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        // Expected: slice [0:1, 1:3] = [[2.0, 3.0]]
        let expected = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![2.0f32, 3.0]).unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    #[test]
    fn test_op_slice_error_mismatched_lengths() {
        // Test error case: mismatched starts and ends lengths
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![4],
            DataType::Float,
        );

        let starts = create_ort_tensor(
            vec![0i64, 1].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![2],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![4i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends];
        let result = OrtEngine::op_slice(&node, &inputs);

        assert!(matches!(
            result,
            Err(OrtError::InvalidTensorData(msg)) if msg.contains("Starts and ends must have the same length")
        ));
    }

    #[test]
    fn test_op_slice_error_out_of_bounds_axis() {
        // Test error case: axis out of bounds
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![4],
            DataType::Float,
        );

        let starts = create_ort_tensor(
            vec![0i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![4i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let axes = create_ort_tensor(
            vec![5i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(), // Out of bounds
            vec![1],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends, axes];
        let result = OrtEngine::op_slice(&node, &inputs);

        assert!(matches!(
            result,
            Err(OrtError::InvalidTensorData(msg)) if msg.contains("out of bounds")
        ));
    }

    #[test]
    fn test_op_slice_empty_result() {
        // Test case that results in empty slice
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![4],
            DataType::Float,
        );

        // starts = [2], ends = [2] (empty range)
        let starts = create_ort_tensor(
            vec![2i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![2i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends];
        let result = OrtEngine::op_slice(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        // Expected: empty slice
        let expected = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    #[test]
    fn test_op_slice_int64_data() {
        // Test slicing with int64 data
        let input_data = vec![10i64, 20, 30, 40, 50, 60]; // 2x3 matrix
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2, 3],
            DataType::Int64,
        );

        // starts = [1], ends = [2], axes = [0]
        let starts = create_ort_tensor(
            vec![1i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![2i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );
        let axes = create_ort_tensor(
            vec![0i64].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![1],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends, axes];
        let result = OrtEngine::op_slice(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        // Expected: slice [1:2, :] = second row only = [40, 50, 60]
        let expected = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![40i64, 50, 60]).unwrap();
        match result_array {
            ArrayDResult::Int64(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected int64 array"),
        }
    }

    #[test]
    fn test_op_slice_3d_tensor() {
        // Test slicing with 3D tensor
        let input_data: Vec<f32> = (1..=24).map(|x| x as f32).collect(); // 2x3x4 tensor
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2, 3, 4],
            DataType::Float,
        );

        // starts = [0, 1, 0], ends = [1, 3, 2], axes = [0, 1, 2]
        let starts = create_ort_tensor(
            vec![0i64, 1, 0].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![3],
            DataType::Int64,
        );
        let ends = create_ort_tensor(
            vec![1i64, 3, 2].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![3],
            DataType::Int64,
        );
        let axes = create_ort_tensor(
            vec![0i64, 1, 2].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![3],
            DataType::Int64,
        );

        let node = NodeProto::default();
        let inputs = vec![input, starts, ends, axes];
        let result = OrtEngine::op_slice(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        // Expected: slice [0:1, 1:3, 0:2] = first batch, rows 1-2, columns 0-1
        // Original tensor: [[[1,2,3,4], [5,6,7,8], [9,10,11,12]], [[13,14,15,16], [17,18,19,20], [21,22,23,24]]]
        // Result should be: [[[5,6], [9,10]]]
        let expected = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![5.0f32, 6.0, 9.0, 10.0]).unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    #[test]
    fn test_op_where_float() {
        // Test basic where operation with float tensors
        // condition: [true, false, true]
        // x: [1.0, 2.0, 3.0]
        // y: [4.0, 5.0, 6.0]
        // expected: [1.0, 5.0, 3.0]
        
        let condition = create_ort_tensor(
            vec![1u8, 0, 1], // true, false, true
            vec![3],
            DataType::Boolean,
        );
        
        let x = create_ort_tensor(
            vec![1.0f32, 2.0, 3.0].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![3],
            DataType::Float,
        );
        
        let y = create_ort_tensor(
            vec![4.0f32, 5.0, 6.0].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![3],
            DataType::Float,
        );

        let node = NodeProto::default();
        let inputs = vec![condition, x, y];
        let result = OrtEngine::op_where(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        let expected = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0f32, 5.0, 3.0]).unwrap();
        match result_array {
            ArrayDResult::Float(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected float array"),
        }
    }

    #[test]
    fn test_op_where_int32() {
        // Test where operation with int32 tensors
        let condition = create_ort_tensor(
            vec![0u8, 1, 0], // false, true, false
            vec![3],
            DataType::Boolean,
        );
        
        let x = create_ort_tensor(
            vec![10i32, 20, 30].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![3],
            DataType::Int32,
        );
        
        let y = create_ort_tensor(
            vec![40i32, 50, 60].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![3],
            DataType::Int32,
        );

        let node = NodeProto::default();
        let inputs = vec![condition, x, y];
        let result = OrtEngine::op_where(&node, &inputs).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        let expected = ArrayD::from_shape_vec(IxDyn(&[3]), vec![40i32, 20, 60]).unwrap();
        match result_array {
            ArrayDResult::Int32(arr) => assert_eq!(arr, expected),
            _ => panic!("Expected int32 array"),
        }
    }

    // #[test]
    // fn test_op_where_broadcasting() {
    //     // Test where operation with broadcasting
    //     // condition: [true, false] (shape: [2])
    //     // x: [[1.0, 2.0], [3.0, 4.0]] (shape: [2, 2])
    //     // y: [10.0] (shape: [1])
    //     // expected: [[1.0, 2.0], [10.0, 10.0]]
        
    //     let condition = create_ort_tensor(
    //         vec![1u8, 0], // true, false
    //         vec![2],
    //         DataType::Boolean,
    //     );
        
    //     let x = create_ort_tensor(
    //         vec![1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
    //         vec![2, 2],
    //         DataType::Float,
    //     );
        
    //     let y = create_ort_tensor(
    //         vec![10.0f32].iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
    //         vec![1],
    //         DataType::Float,
    //     );

    //     let node = NodeProto::default();
    //     let inputs = vec![condition, x, y];
    //     let result = OrtEngine::op_where(&node, &inputs).unwrap();
    //     let result_array = ort_to_ndarray(&result).unwrap();

    //     let expected = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0f32, 2.0, 10.0, 10.0]).unwrap();
    //     match result_array {
    //         ArrayDResult::Float(arr) => assert_eq!(arr, expected),
    //         _ => panic!("Expected float array"),
    //     }
    // }

    #[test]
    fn test_op_softmax() {
        // Test LogSoftmax operation (note: op_softmax actually implements LogSoftmax)
        // Input: [1.0, 2.0, 3.0] along axis -1 (default)
        let input_data = vec![1.0f32, 2.0, 3.0];
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![3],
            DataType::Float,
        );

        // Create node with default axis (-1)
        let mut node = NodeProto::default();
        node.attributes = vec![AttributeProto {
            name: "axis".to_string(),
            i: -1,
            ..Default::default()
        }];

        let result = OrtEngine::op_softmax(&node, &[input]).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        // For LogSoftmax: log(softmax(x)) = x - log(sum(exp(x)))
        // For input [1, 2, 3]:
        // exp([1, 2, 3]) = [e, e^2, e^3] ≈ [2.718, 7.389, 20.086]
        // sum = 30.193
        // log(sum) ≈ 3.408
        // LogSoftmax = [1-3.408, 2-3.408, 3-3.408] = [-2.408, -1.408, -0.408]
        
        match result_array {
            ArrayDResult::Float(arr) => {
                assert_eq!(arr.shape(), &[3]);
                // Check that the values are approximately correct for LogSoftmax
                let values: Vec<f32> = arr.iter().cloned().collect();
                
                // The exact values depend on numerical precision, but we can check:
                // 1. The largest input should have the largest (least negative) output
                assert!(values[2] > values[1]);
                assert!(values[1] > values[0]);
                
                // 2. All values should be negative (since log(probability) <= 0)
                assert!(values.iter().all(|&x| x <= 0.0));
                
                // 3. The sum of exp(log_softmax) should be approximately 1
                let exp_sum: f32 = values.iter().map(|&x| x.exp()).sum();
                assert!((exp_sum - 1.0).abs() < 1e-6);
            }
            _ => panic!("Expected float array"),
        }
    }

    #[test]
    fn test_op_softmax_2d_axis_1() {
        // Test LogSoftmax on 2D tensor with axis=1
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // Shape: [2, 3]
        let input = create_ort_tensor(
            input_data
                .iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect(),
            vec![2, 3],
            DataType::Float,
        );

        // Create node with axis=1
        let mut node = NodeProto::default();
        node.attributes = vec![AttributeProto {
            name: "axis".to_string(),
            i: 1,
            ..Default::default()
        }];

        let result = OrtEngine::op_softmax(&node, &[input]).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();

        match result_array {
            ArrayDResult::Float(arr) => {
                assert_eq!(arr.shape(), &[2, 3]);
                
                // Check that LogSoftmax is applied along axis 1
                // For each row, the sum of exp(log_softmax) should be 1
                for row in 0..2 {
                    let row_values: Vec<f32> = (0..3).map(|col| arr[[row, col]]).collect();
                    let exp_sum: f32 = row_values.iter().map(|&x| x.exp()).sum();
                    assert!((exp_sum - 1.0).abs() < 1e-6);
                    
                    // All values should be negative
                    assert!(row_values.iter().all(|&x| x <= 0.0));
                }
            }
            _ => panic!("Expected float array"),
        }
    }
    #[test]
    fn test_op_softmax_negative_inputs() {
        let input_data = vec![-1.0f32, -2.0, -3.0];
        let input = create_ort_tensor(
            input_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
            vec![3],
            DataType::Float,
        );
        let mut node = NodeProto::default();
        node.attributes = vec![AttributeProto {
            name: "axis".to_string(),
            i: -1,
            ..Default::default()
        }];
        let result = OrtEngine::op_softmax(&node, &[input]).unwrap();
        let result_array = ort_to_ndarray(&result).unwrap();
        match result_array {
            ArrayDResult::Float(arr) => {
                assert_eq!(arr.shape(), &[3]);
                let values: Vec<f32> = arr.iter().cloned().collect();
                assert!(values[0] > values[1]);
                assert!(values[1] > values[2]);
                assert!(values.iter().all(|&x| x <= 0.0));
                let exp_sum: f32 = values.iter().map(|&x| x.exp()).sum();
                assert!((exp_sum - 1.0).abs() < 1e-6);
            }
            _ => panic!("Expected float array"),
        }
    }
    

    #[test]
fn test_op_softmax_large_inputs() {
    let input_data = vec![1000.0f32, 1001.0, 1002.0];
    let input = create_ort_tensor(
        input_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
        vec![3],
        DataType::Float,
    );
    let mut node = NodeProto::default();
    node.attributes = vec![AttributeProto {
        name: "axis".to_string(),
        i: -1,
        ..Default::default()
    }];
    let result = OrtEngine::op_softmax(&node, &[input]).unwrap();
    let result_array = ort_to_ndarray(&result).unwrap();
    match result_array {
        ArrayDResult::Float(arr) => {
            assert_eq!(arr.shape(), &[3]);
            let values: Vec<f32> = arr.iter().cloned().collect();
            assert!(values[2] > values[1]);
            assert!(values[1] > values[0]);
            assert!(values.iter().all(|&x| x <= 0.0));
            let exp_sum: f32 = values.iter().map(|&x| x.exp()).sum();
            assert!((exp_sum - 1.0).abs() < 1e-6);
            // Optional: Check approximate values
            let expected = vec![-2.408, -1.408, -0.408];
            for (v, e) in values.iter().zip(expected.iter()) {
                println!("{}",(v - e).abs());
                assert!((v - e).abs() < 1e-5);
            }
        }
        _ => panic!("Expected float array"),
    }
}

#[test]
fn test_op_softmax_single_element() {
    let input_data = vec![5.0f32];
    let input = create_ort_tensor(
        input_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
        vec![1],
        DataType::Float,
    );
    let mut node = NodeProto::default();
    node.attributes = vec![AttributeProto {
        name: "axis".to_string(),
        i: -1,
        ..Default::default()
    }];
    let result = OrtEngine::op_softmax(&node, &[input]).unwrap();
    let result_array = ort_to_ndarray(&result).unwrap();
    match result_array {
        ArrayDResult::Float(arr) => {
            assert_eq!(arr.shape(), &[1]);
            let values: Vec<f32> = arr.iter().cloned().collect();
            assert!((values[0] - 0.0).abs() < 1e-6);
        }
        _ => panic!("Expected float array"),
    }
}


#[test]
fn test_op_softmax_zero_inputs() {
    let input_data = vec![0.0f32, 0.0, 0.0];
    let input = create_ort_tensor(
        input_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
        vec![3],
        DataType::Float,
    );
    let mut node = NodeProto::default();
    node.attributes = vec![AttributeProto {
        name: "axis".to_string(),
        i: -1,
        ..Default::default()
    }];
    let result = OrtEngine::op_softmax(&node, &[input]).unwrap();
    let result_array = ort_to_ndarray(&result).unwrap();
    match result_array {
        ArrayDResult::Float(arr) => {
            assert_eq!(arr.shape(), &[3]);
            let values: Vec<f32> = arr.iter().cloned().collect();
            assert!((values[0] - (-1.0986123)).abs() < 1e-6);
            assert!(values.iter().all(|&x| (x - values[0]).abs() < 1e-6));
            let exp_sum: f32 = values.iter().map(|&x| x.exp()).sum();
            assert!((exp_sum - 1.0).abs() < 1e-6);
        }
        _ => panic!("Expected float array"),
    }
}

#[test]
fn test_op_softmax_multi_dim() {
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = create_ort_tensor(
        input_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
        vec![2, 3],
        DataType::Float,
    );
    let mut node = NodeProto::default();
    node.attributes = vec![AttributeProto {
        name: "axis".to_string(),
        i: 1,
        ..Default::default()
    }];
    let result = OrtEngine::op_softmax(&node, &[input]).unwrap();
    let result_array = ort_to_ndarray(&result).unwrap();
    match result_array {
        ArrayDResult::Float(arr) => {
            assert_eq!(arr.shape(), &[2, 3]);
            let values: Vec<f32> = arr.iter().cloned().collect();
            let expected = vec![-2.408, -1.408, -0.408, -2.408, -1.408, -0.408];
            for (v, e) in values.iter().zip(expected.iter()) {
                assert!((v - e).abs() < 1e-5);
            }
            for i in 0..2 {
                let row: Vec<f32> = values[i * 3..(i + 1) * 3].to_vec();
                let exp_sum: f32 = row.iter().map(|&x| x.exp()).sum();
                assert!((exp_sum - 1.0).abs() < 1e-6);
            }
        }
        _ => panic!("Expected float array"),
    }
}
#[test]
#[should_panic(expected = "Invalid axis")]
fn test_op_softmax_invalid_axis() {
    let input_data = vec![1.0f32, 2.0, 3.0];
    let input = create_ort_tensor(
        input_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
        vec![3],
        DataType::Float,
    );
    let mut node = NodeProto::default();
    node.attributes = vec![AttributeProto {
        name: "axis".to_string(),
        i: 10,
        ..Default::default()
    }];
    let _ = OrtEngine::op_softmax(&node, &[input]).unwrap();
}
#[test]
#[should_panic(expected = "Invalid data type")]
fn test_op_softmax_non_float_input() {
    let input_data = vec![1i32, 2, 3];
    let input = create_ort_tensor(
        input_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
        vec![3],
        DataType::Int32,
    );
    let mut node = NodeProto::default();
    node.attributes = vec![AttributeProto {
        name: "axis".to_string(),
        i: -1,
        ..Default::default()
    }];
    let _ = OrtEngine::op_softmax(&node, &[input]).unwrap();
}
#[test]
fn test_op_softmax_axis_0() {
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input = create_ort_tensor(
        input_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect(),
        vec![2, 2],
        DataType::Float,
    );
    let mut node = NodeProto::default();
    node.attributes = vec![AttributeProto {
        name: "axis".to_string(),
        i: 0,
        ..Default::default()
    }];
    let result = OrtEngine::op_softmax(&node, &[input]).unwrap();
    let result_array = ort_to_ndarray(&result).unwrap();
    match result_array {
        ArrayDResult::Float(arr) => {
            assert_eq!(arr.shape(), &[2, 2]);
            let values: Vec<f32> = arr.iter().cloned().collect();
            // For axis=0, apply LogSoftmax to columns: [1, 3] and [2, 4]
            let col1_exp_sum: f32 = [values[0], values[2]].iter().map(|&x| x.exp()).sum();
            let col2_exp_sum: f32 = [values[1], values[3]].iter().map(|&x| x.exp()).sum();
            assert!((col1_exp_sum - 1.0).abs() < 1e-6);
            assert!((col2_exp_sum - 1.0).abs() < 1e-6);
        }
        _ => panic!("Expected float array"),
    }
}

#[test]
fn test_op_stft_real_signal_default() {
    // Create a simple sine wave signal
    let signal_data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let signal_bytes: Vec<u8> = signal_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    // Create signal tensor with shape [1, 64, 1] (batch_size=1, signal_length=64, channels=1)
    let signal = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(64), Dimensions::Fixed(1)],
        dtype: DataType::Float,
        data: Arc::new(signal_bytes),
    };
    
    // Create frame_step tensor (scalar value 16)
    let frame_step_value = 16i64;
    let frame_step = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_step_value.to_le_bytes().to_vec()),
    };
    
    // Create node with default attributes
    let node = NodeProto::default();
    
    // Execute STFT operation
    let result = OrtEngine::op_stft(&node, &[signal, frame_step]).unwrap();
    
    // Verify result
    match result {
        OrtValue::Tensor { shape, dtype, .. } => {
            assert_eq!(dtype, DataType::Float);
            
            // For a frame_length of 64 (default) and frame_step of 16, we should have:
            // - 1 batch
            // - (64-64)/16 + 1 = 1 frame
            // - 64/2 + 1 = 33 frequency bins (onesided=true by default)
            // - 2 channels (real and imaginary parts)
            assert_eq!(shape, vec![Dimensions::Fixed(1), Dimensions::Fixed(1), Dimensions::Fixed(33), Dimensions::Fixed(2)]);
        },
        _ => panic!("Expected tensor output"),
    }
}

#[test]
fn test_op_stft_complex_signal() {
    // Create a complex signal (real and imaginary parts)
    let mut signal_data = Vec::new();
    for i in 0..32 {
        signal_data.push((i as f32 * 0.1).sin()); // Real part
        signal_data.push((i as f32 * 0.1).cos()); // Imaginary part
    }
    let signal_bytes: Vec<u8> = signal_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    // Create signal tensor with shape [1, 32, 2] (batch_size=1, signal_length=32, channels=2)
    let signal = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(32), Dimensions::Fixed(2)],
        dtype: DataType::Float,
        data: Arc::new(signal_bytes),
    };
    
    // Create frame_step tensor (scalar value 8)
    let frame_step_value = 8i64;
    let frame_step = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_step_value.to_le_bytes().to_vec()),
    };
    
    // Create node with onesided=false attribute
    let node = NodeProto {
        attributes: vec![AttributeProto {
            name: "onesided".to_string(),
            i: 0, // false
            ..Default::default()
        }],
        ..Default::default()
    };
    
    // Execute STFT operation
    let result = OrtEngine::op_stft(&node, &[signal, frame_step]).unwrap();
    
    // Verify result
    match result {
        OrtValue::Tensor { shape, dtype, .. } => {
            assert_eq!(dtype, DataType::Float);
            
            // For a frame_length of 32 (default) and frame_step of 8, we should have:
            // - 1 batch
            // - (32-32)/8 + 1 = 1 frame
            // - 32 frequency bins (onesided=false)
            // - 2 channels (real and imaginary parts)
            assert_eq!(shape, vec![Dimensions::Fixed(1), Dimensions::Fixed(1), Dimensions::Fixed(32), Dimensions::Fixed(2)]);
        },
        _ => panic!("Expected tensor output"),
    }
}

#[test]
fn test_op_stft_custom_window() {
    // Create a simple signal
    let signal_data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let signal_bytes: Vec<u8> = signal_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    // Create signal tensor with shape [1, 64, 1]
    let signal = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(64), Dimensions::Fixed(1)],
        dtype: DataType::Float,
        data: Arc::new(signal_bytes),
    };
    
    // Create frame_step tensor (scalar value 16)
    let frame_step_value = 16i64;
    let frame_step = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_step_value.to_le_bytes().to_vec()),
    };
    
    // Create Hann window of length 32
    let window_data: Vec<f32> = (0..32).map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / 31.0).cos())).collect();
    let window_bytes: Vec<u8> = window_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    let window = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(32)],
        dtype: DataType::Float,
        data: Arc::new(window_bytes),
    };
    
    // Create node with default attributes
    let node = NodeProto::default();
    
    // Create frame_length tensor (scalar value 32)
    let frame_length_value = 32i64;
    let frame_length = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_length_value.to_le_bytes().to_vec()),
    };

    // Execute STFT operation with custom window, explicitly passing None for frame_length
    let result = OrtEngine::op_stft(&node, &[signal, frame_step, window, frame_length]).unwrap();
    
    // Verify result
    match result {
        OrtValue::Tensor { shape, dtype, .. } => {
            assert_eq!(dtype, DataType::Float);
            
            // For a frame_length of 32 (from window) and frame_step of 16, we should have:
            // - 1 batch
            // - (64-32)/16 + 1 = 3 frames
            // - 32/2 + 1 = 17 frequency bins (onesided=true by default)
            // - 2 channels (real and imaginary parts)
            assert_eq!(shape, vec![Dimensions::Fixed(1), Dimensions::Fixed(3), Dimensions::Fixed(17), Dimensions::Fixed(2)]);
        },
        _ => panic!("Expected tensor output"),
    }
}

#[test]
fn test_op_stft_short_signal() {
    // Create a short signal
    let signal_data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).sin()).collect();
    let signal_bytes: Vec<u8> = signal_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    // Create signal tensor with shape [1, 16, 1]
    let signal = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(16), Dimensions::Fixed(1)],
        dtype: DataType::Float,
        data: Arc::new(signal_bytes),
    };
    
    // Create frame_step tensor (scalar value 8)
    let frame_step_value = 8i64;
    let frame_step = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_step_value.to_le_bytes().to_vec()),
    };
    
    // Create frame_length tensor (scalar value 32)
    let frame_length_value = 32i64;
    let frame_length = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_length_value.to_le_bytes().to_vec()),
    };
    
    // Create node with default attributes
    let node = NodeProto::default();
    
    // Create default window tensor of length 32 (ones, simulating no windowing)
    let window_data: Vec<f32> = vec![1.0f32; 32];
    let window_bytes: Vec<u8> = window_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    let window = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(32)],
        dtype: DataType::Float,
        data: Arc::new(window_bytes),
    };

    // Create frame_length tensor (scalar value 32)
    let frame_length_value = 32i64;
    let frame_length = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_length_value.to_le_bytes().to_vec()),
    };

    // Execute STFT operation with explicit frame_length
    let result = OrtEngine::op_stft(&node, &[signal, frame_step, window,frame_length]).unwrap();
    
    // Verify result
    match result {
        OrtValue::Tensor { shape, dtype, .. } => {
            assert_eq!(dtype, DataType::Float);
            
            // For a frame_length of 32 and frame_step of 8, we should have:
            // - 1 batch
            // - (16-32)/8 + 1 = -1 + 1 = 0 frames (but we should get at least 1 frame)
            // - 32/2 + 1 = 17 frequency bins (onesided=true by default)
            // - 2 channels (real and imaginary parts)
            assert_eq!(shape, vec![Dimensions::Fixed(1), Dimensions::Fixed(1), Dimensions::Fixed(17), Dimensions::Fixed(2)]);
        },
        _ => panic!("Expected tensor output"),
    }
}

#[test]
fn test_op_stft_invalid_signal_shape() {
    // Create a signal with invalid shape (missing channel dimension)
    let signal_data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let signal_bytes: Vec<u8> = signal_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    // Create signal tensor with shape [1, 64] (missing channel dimension)
    let signal = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(64)],
        dtype: DataType::Float,
        data: Arc::new(signal_bytes),
    };
    
    // Create frame_step tensor (scalar value 16)
    let frame_step_value = 16i64;
    let frame_step = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_step_value.to_le_bytes().to_vec()),
    };
    
    // Create node with default attributes
    let node = NodeProto::default();
    
    // Execute STFT operation
    let result = OrtEngine::op_stft(&node, &[signal, frame_step]);
    
    // Verify error
    assert!(matches!(result, Err(OrtError::InvalidTensorData(_))));
}

#[test]
fn test_op_stft_invalid_channels() {
    // Create a signal with invalid number of channels (3)
    let signal_data: Vec<f32> = (0..64*3).map(|i| (i as f32 * 0.1).sin()).collect();
    let signal_bytes: Vec<u8> = signal_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    // Create signal tensor with shape [1, 64, 3] (invalid number of channels)
    let signal = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(64), Dimensions::Fixed(3)],
        dtype: DataType::Float,
        data: Arc::new(signal_bytes),
    };
    
    // Create frame_step tensor (scalar value 16)
    let frame_step_value = 16i64;
    let frame_step = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_step_value.to_le_bytes().to_vec()),
    };
    
    // Create node with default attributes
    let node = NodeProto::default();
    
    // Execute STFT operation
    let result = OrtEngine::op_stft(&node, &[signal, frame_step]);
    
    // Verify error
    assert!(matches!(result, Err(OrtError::InvalidTensorData(_))));
}

#[test]
fn test_op_stft_complex_onesided() {
    // Create a complex signal
    let mut signal_data = Vec::new();
    for i in 0..32 {
        signal_data.push((i as f32 * 0.1).sin()); // Real part
        signal_data.push((i as f32 * 0.1).cos()); // Imaginary part
    }
    let signal_bytes: Vec<u8> = signal_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    // Create signal tensor with shape [1, 32, 2]
    let signal = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(32), Dimensions::Fixed(2)],
        dtype: DataType::Float,
        data: Arc::new(signal_bytes),
    };
    
    // Create frame_step tensor (scalar value 8)
    let frame_step_value = 8i64;
    let frame_step = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_step_value.to_le_bytes().to_vec()),
    };
    
    // Create node with onesided=true attribute (which is invalid for complex input)
    let node = NodeProto {
        attributes: vec![AttributeProto {
            name: "onesided".to_string(),
            i: 1, // true
            ..Default::default()
        }],
        ..Default::default()
    };
    
    // Execute STFT operation
    let result = OrtEngine::op_stft(&node, &[signal, frame_step]);
    
    // Verify error
    assert!(matches!(result, Err(OrtError::InvalidTensorData(_))));
}

#[test]
fn test_op_stft_invalid_frame_step() {
    // Create a simple signal
    let signal_data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let signal_bytes: Vec<u8> = signal_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    // Create signal tensor with shape [1, 64, 1]
    let signal = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(64), Dimensions::Fixed(1)],
        dtype: DataType::Float,
        data: Arc::new(signal_bytes),
    };
    
    // Create frame_step tensor with non-scalar shape
    let frame_step_data = vec![16i64, 32];
    let frame_step_bytes: Vec<u8> = frame_step_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    let frame_step = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(2)],
        dtype: DataType::Int64,
        data: Arc::new(frame_step_bytes),
    };
    
    // Create node with default attributes
    let node = NodeProto::default();
    
    // Execute STFT operation
    let result = OrtEngine::op_stft(&node, &[signal, frame_step]);
    
    // Verify error
    assert!(matches!(result, Err(OrtError::InvalidTensorData(_))));
}

#[test]
fn test_op_stft_mismatched_window() {
    // Create a simple signal
    let signal_data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let signal_bytes: Vec<u8> = signal_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    // Create signal tensor with shape [1, 64, 1]
    let signal = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(64), Dimensions::Fixed(1)],
        dtype: DataType::Float,
        data: Arc::new(signal_bytes),
    };
    
    // Create frame_step tensor (scalar value 16)
    let frame_step_value = 16i64;
    let frame_step = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_step_value.to_le_bytes().to_vec()),
    };
    
    // Create window of length 32
    let window_data: Vec<f32> = (0..32).map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / 31.0).cos())).collect();
    let window_bytes: Vec<u8> = window_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    // Create frame_length tensor (scalar value 64, which doesn't match window length)
    let frame_length_value = 64i64;
    let frame_length = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_length_value.to_le_bytes().to_vec()),
    };
    
    // Create node with default attributes
    let node = NodeProto::default();
    
    // Create default window tensor of length 32 (ones, simulating no windowing)
    let window_data: Vec<f32> = vec![1.0f32; 32];
    let window_bytes: Vec<u8> = window_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    let window = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(32)],
        dtype: DataType::Float,
        data: Arc::new(window_bytes),
    };

    // Execute STFT operation with mismatched window and frame_length
    let result = OrtEngine::op_stft(&node, &[signal, frame_step, window, frame_length]);
    
    // Verify error
    assert!(matches!(result, Err(OrtError::InvalidTensorData(_))));
}

#[test]
fn test_op_stft_nan_signal() {
    // Create a signal with NaN values
    let mut signal_data = Vec::new();
    for i in 0..64 {
        if i % 10 == 0 {
            signal_data.push(f32::NAN); // Add some NaN values
        } else {
            signal_data.push((i as f32 * 0.1).sin());
        }
    }
    let signal_bytes: Vec<u8> = signal_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    // Create signal tensor with shape [1, 64, 1]
    let signal = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(64), Dimensions::Fixed(1)],
        dtype: DataType::Float,
        data: Arc::new(signal_bytes),
    };
    
    // Create frame_step tensor (scalar value 16)
    let frame_step_value = 16i64;
    let frame_step = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_step_value.to_le_bytes().to_vec()),
    };
    
    // Create node with default attributes
    let node = NodeProto::default();
    
    // Execute STFT operation
    let result = OrtEngine::op_stft(&node, &[signal, frame_step]).unwrap();
    
    // Verify result (should complete without error)
    match result {
        OrtValue::Tensor { shape, dtype, .. } => {
            assert_eq!(dtype, DataType::Float);
            assert_eq!(shape, vec![Dimensions::Fixed(1), Dimensions::Fixed(1), Dimensions::Fixed(33), Dimensions::Fixed(2)]);
        },
        _ => panic!("Expected tensor output"),
    }
}

#[test]
fn test_op_stft_multiple_batches() {
    // Create a signal with multiple batches
    let mut signal_data = Vec::new();
    for b in 0..2 {
        for i in 0..32 {
            signal_data.push((i as f32 * 0.1 * (b + 1) as f32).sin());
        }
    }
    let signal_bytes: Vec<u8> = signal_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    // Create signal tensor with shape [2, 32, 1] (2 batches)
    let signal = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(2), Dimensions::Fixed(32), Dimensions::Fixed(1)],
        dtype: DataType::Float,
        data: Arc::new(signal_bytes),
    };
    
    // Create frame_step tensor (scalar value 8)
    let frame_step_value = 8i64;
    let frame_step = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_step_value.to_le_bytes().to_vec()),
    };
    
    // Create frame_length tensor (scalar value 16)
    let frame_length_value = 16i64;
    let frame_length = OrtValue::Tensor {
        shape: vec![],
        dtype: DataType::Int64,
        data: Arc::new(frame_length_value.to_le_bytes().to_vec()),
    };
    
    // Create node with default attributes
    let node = NodeProto::default();
    
    // Create default window tensor of length 32 (ones, simulating no windowing)
    let window_data: Vec<f32> = vec![1.0f32; 16];
    let window_bytes: Vec<u8> = window_data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
    
    let window = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(16)],
        dtype: DataType::Float,
        data: Arc::new(window_bytes),
    };

    // Execute STFT operation
    let result = OrtEngine::op_stft(&node, &[signal, frame_step, window, frame_length]).unwrap();
    
    // Verify result
    match result {
        OrtValue::Tensor { shape, dtype, .. } => {
            assert_eq!(dtype, DataType::Float);
            
            // For a frame_length of 16 and frame_step of 8, we should have:
            // - 2 batches
            // - (32-16)/8 + 1 = 3 frames
            // - 16/2 + 1 = 9 frequency bins (onesided=true by default)
            // - 2 channels (real and imaginary parts)
            assert_eq!(shape, vec![Dimensions::Fixed(2), Dimensions::Fixed(3), Dimensions::Fixed(9), Dimensions::Fixed(2)]);
        },
        _ => panic!("Expected tensor output"),
    }
}



}