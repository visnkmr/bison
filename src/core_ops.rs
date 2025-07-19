// core_ops.rs - Implementation of core operations
// This file contains implementations of functions declared in main.rs

use crate::{convert::{ndarray_to_ort, ort_to_ndarray}, *};

// Implementations of functions will go here

impl OrtEngine{
    // Helper function to determine output data type based on input types
    fn get_output_dtype(inputs: &[OrtValue]) -> DataType {
        // If there are no inputs, default to Float
        if inputs.is_empty() {
            return DataType::Float;
        }
        
        // For operations that should preserve input type, use the first tensor's type
        if let Some(OrtValue::Tensor { dtype, .. }) = inputs.first() {
            return *dtype;
        }
        
        // Default to Float if no tensor is found
        DataType::Float
    }
    
    // Helper function for operations that require same type inputs
    fn check_same_dtype(inputs: &[OrtValue]) -> OrtResult<DataType> {
        if inputs.len() < 2 {
            return Err(OrtError::InvalidTensorData("Operation requires at least two inputs".into()));
        }
        
        let first_dtype = match inputs.first() {
            Some(OrtValue::Tensor { dtype, .. }) => *dtype,
            _ => return Err(OrtError::TypeMismatch("First input must be a tensor")),
        };
        
        for input in inputs.iter().skip(1) {
            match input {
                OrtValue::Tensor { dtype, .. } if *dtype == first_dtype => {},
                OrtValue::Tensor { .. } => return Err(OrtError::TypeMismatch("All inputs must have the same data type")),
                _ => return Err(OrtError::TypeMismatch("All inputs must be tensors")),
            }
        }
        
        Ok(first_dtype)
    }

    
        // Arithmetic Operations
        pub fn op_add(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        // Check if both inputs have the same data type
        let dtype = match (inputs.get(0), inputs.get(1)) {
            (
                Some(OrtValue::Tensor { dtype: dtype1, .. }),
                Some(OrtValue::Tensor { dtype: dtype2, .. }),
            ) if dtype1 == dtype2 => dtype1,
            _ => {
                return Err(OrtError::TypeMismatch(
                    "Add requires two tensors with matching data types",
                ))
            }
        };

        // Handle different data types
        match dtype {
            DataType::Float => {
                let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
                let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
                let result = array1 + array2;
                Ok(ndarray_to_ort(result.unwrap(), DataType::Float))
            },
            DataType::Int64 => {
                let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
                let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
                let result = array1 + array2;
                Ok(ndarray_to_ort(result.unwrap(), DataType::Int64))
            },
            DataType::Int32 => {
                let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
                let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
                let result = array1 + array2;
                Ok(ndarray_to_ort(result.unwrap(), DataType::Int32))
            },
            _ => Err(OrtError::TypeMismatch("Unsupported data type for Add operation")),
        }
    }
    
    //     pub fn op_sub(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    //     // Check if both inputs have the same data type
    //     let dtype = match (inputs.get(0), inputs.get(1)) {
    //         (
    //             Some(OrtValue::Tensor { dtype: dtype1, .. }),
    //             Some(OrtValue::Tensor { dtype: dtype2, .. }),
    //         ) if dtype1 == dtype2 => dtype1,
    //         _ => {
    //             return Err(OrtError::TypeMismatch(
    //                 "Sub requires two tensors with matching data types",
    //             ))
    //         }
    //     };

    //     // Handle different data types
    //     match dtype {
    //         DataType::Float => {
    //             let array1 = ort_to_ndarray::<f32>(inputs.get(0).unwrap())?;
    //             let array2 = ort_to_ndarray::<f32>(inputs.get(1).unwrap())?;
    //             let result = array1 - array2;
    //             Ok(ndarray_to_ort(result, DataType::Float))
    //         },
    //         DataType::Int64 => {
    //             let array1 = ort_to_ndarray::<i64>(inputs.get(0).unwrap())?;
    //             let array2 = ort_to_ndarray::<i64>(inputs.get(1).unwrap())?;
    //             let result = array1 - array2;
    //             Ok(ndarray_to_ort(result, DataType::Int64))
    //         },
    //         DataType::Int32 => {
    //             let array1 = ort_to_ndarray::<i32>(inputs.get(0).unwrap())?;
    //             let array2 = ort_to_ndarray::<i32>(inputs.get(1).unwrap())?;
    //             let result = array1 - array2;
    //             Ok(ndarray_to_ort(result, DataType::Int32))
    //         },
    //         _ => Err(OrtError::TypeMismatch("Unsupported data type for Sub operation")),
    //     }
    // }
            
    
    
}