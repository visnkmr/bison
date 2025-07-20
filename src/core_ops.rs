// core_ops.rs - Implementation of core operations
// This file contains implementations of functions declared in main.rs

use crate::{convert::{ndarray_to_ort, ort_to_ndarray, pow_array, sqrt_array, ArrayDResult}, *};

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
            _ => return Err(OrtError::TypeMismatch(format!("First input must be a tensor"))),
        };
        
        for input in inputs.iter().skip(1) {
            match input {
                OrtValue::Tensor { dtype, .. } if *dtype == first_dtype => {},
                OrtValue::Tensor { .. } => return Err(OrtError::TypeMismatch(format!("All inputs must have the same data type"))),
                _ => return Err(OrtError::TypeMismatch(format!("All inputs must be tensors"))),
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
                    format!("Add requires two tensors with matching data types"),
                ))
            }
        };
        let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        let result = array1 + array2;
        Ok(ndarray_to_ort(result.unwrap(), *dtype))
        // Handle different data types
        // match dtype {
        //     DataType::Float => {
        //         let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        //         let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        //         let result = array1 + array2;
        //         Ok(ndarray_to_ort(result.unwrap(), DataType::Float))
        //     },
        //     DataType::Int64 => {
        //         let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        //         let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        //         let result = array1 + array2;
        //         Ok(ndarray_to_ort(result.unwrap(), DataType::Int64))
        //     },
        //     DataType::Int32 => {
        //         let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        //         let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        //         let result = array1 + array2;
        //         Ok(ndarray_to_ort(result.unwrap(), DataType::Int32))
        //     },
        //     _ => Err(OrtError::TypeMismatch("Unsupported data type for Add operation")),
        // }
    }

    
        pub fn op_sub(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        // Check if both inputs have the same data type
        let dtype = match (inputs.get(0), inputs.get(1)) {
            (
                Some(OrtValue::Tensor { dtype: dtype1, .. }),
                Some(OrtValue::Tensor { dtype: dtype2, .. }),
            ) if dtype1 == dtype2 => dtype1,
            _ => {
                return Err(OrtError::TypeMismatch(
                    "Sub requires two tensors with matching data types".to_string(),
                ))
            }
        };
        let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        let result = array1 - array2;
        Ok(ndarray_to_ort(result.unwrap(), *dtype))

        // Handle different data types
        // match dtype {
        //     DataType::Float => {
        //         let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        //         let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        //         let result = array1 - array2;
        //         Ok(ndarray_to_ort(result, DataType::Float))
        //     },
        //     DataType::Int64 => {
        //         let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        //         let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        //         let result = array1 - array2;
        //         Ok(ndarray_to_ort(result, DataType::Int64))
        //     },
        //     DataType::Int32 => {
        //         let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        //         let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        //         let result = array1 - array2;
        //         Ok(ndarray_to_ort(result, DataType::Int32))
        //     },
        //     _ => Err(OrtError::TypeMismatch("Unsupported data type for Sub operation")),
        // }
    }

    pub fn op_mul(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        // Check if both inputs have the same data type
        let dtype = match (inputs.get(0), inputs.get(1)) {
            (
                Some(OrtValue::Tensor { dtype: dtype1, .. }),
                Some(OrtValue::Tensor { dtype: dtype2, .. }),
            ) if dtype1 == dtype2 => dtype1,
            _ => {
                return Err(OrtError::TypeMismatch(
                    "Mul requires two tensors with matching data types".to_string(),
                ))
            }
        };

        let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        let result = array1 * array2;
        Ok(ndarray_to_ort(result.unwrap(), *dtype))
        // Handle different data types
        // match dtype {
        //     DataType::Float => {
        //         let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        //         let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        //         let result = array1 * array2;
        //         Ok(ndarray_to_ort(result, DataType::Float))
        //     },
        //     DataType::Int64 => {
        //         let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        //         let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        //         let result = array1 * array2;
        //         Ok(ndarray_to_ort(result, DataType::Int64))
        //     },
        //     DataType::Int32 => {
        //         let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        //         let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        //         let result = array1 * array2;
        //         Ok(ndarray_to_ort(result, DataType::Int32))
        //     },
        //     _ => Err(OrtError::TypeMismatch("Unsupported data type for Mul operation")),
        // }
    }
    
        pub fn op_div(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        // Check if both inputs have the same data type
        let dtype = match (inputs.get(0), inputs.get(1)) {
            (
                Some(OrtValue::Tensor { dtype: dtype1, .. }),
                Some(OrtValue::Tensor { dtype: dtype2, .. }),
            ) if dtype1 == dtype2 => dtype1,
            _ => {
                return Err(OrtError::TypeMismatch(
                    "Div requires two tensors with matching data types".to_string(),
                ))
            }
        };

        let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        let result = array1 / array2;
        Ok(ndarray_to_ort(result.unwrap(), *dtype))
    }
            
    pub fn op_pow(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let input1 = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Pow requires two tensors".to_string()))?;
            let input2 = inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Pow requires two tensors".to_string()))?;

            match (input1, input2) {
                (OrtValue::Tensor { dtype: dtype1, .. }, OrtValue::Tensor { dtype: dtype2, .. }) 
                if dtype1 == dtype2 => {
                   
            if let Ok(result)=pow_array(&ort_to_ndarray(input1)?, &ort_to_ndarray(input2)?){
                Ok(ndarray_to_ort(result, *dtype1))
            }
            else{
                Err(OrtError::TypeMismatch("Pow requires two tensors with matching data types".to_string()))
            }
                },
                _ => Err(OrtError::TypeMismatch("Pow requires two tensors with matching data types".to_string())),
            }
        
    }
    
        pub fn op_sqrt(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let input = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Sqrt requires one tensor".to_string()))?;
        match (input) {
            (OrtValue::Tensor { dtype: dtype1, .. })=>{
                if let Ok(result)=sqrt_array(&ort_to_ndarray(input)?){
                    Ok(ndarray_to_ort(result, *dtype1))
                }
                else{
                    Err(OrtError::TypeMismatch("Pow requires two tensors with matching data types".to_string()))
                }

            }
            _=>{
                Err(OrtError::TypeMismatch("Pow requires two tensors with matching data types".to_string()))
            }
        }
    }
    
        // Element-wise Operations
        pub fn op_exp(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let a = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Exp requires one tensor".to_string()))?;

            return match (ort_to_ndarray(a)?) {
                ArrayDResult::Float(arrayinp) => {
                        let result = arrayinp.mapv(|x| x.exp());
                        return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch("no float as expected of exp op".to_string()))
                }
            }
        }

        pub fn op_sigmoid(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Sigmoid requires one float tensor".to_string()))?)?;
            return match (array) {
                ArrayDResult::Float(arrayinp) => {
                        let result = arrayinp.mapv(|a| 1.0 / (1.0 + (-a).exp()));
                        return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch("sigmoid can only work with floats ".to_string()))
                }
            }
        }


        pub fn op_tanh(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Tanh requires one float tensor".to_string()))?)?;
            return match (array) {
                ArrayDResult::Float(arrayinp) => {
                        let result = arrayinp.mapv(|a|a.tanh());
                        return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch("can only work with floats".to_string()))
                }
            }
        }
    
        pub fn op_sin(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Sin requires one float tensor".to_string()))?)?;
            return match (array) {
                ArrayDResult::Float(arrayinp) => {
                        let result = arrayinp.mapv(|a|a.sin());
                        return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch("sin can only work with floats".to_string()))
                }
            }
        }
    
        pub fn op_cos(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Cos requires one float tensor".to_string()))?)?;
            return match (array) {
                ArrayDResult::Float(arrayinp) => {
                        let result = arrayinp.mapv(|a|a.cos());
                        return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch("".to_string()))
                }
            }
        }
    
        pub fn op_atan(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Atan requires one float tensor".to_string()))?)?;
            return match (array) {
                ArrayDResult::Float(arrayinp) => {
                        let result = arrayinp.mapv(|a| a.atan());
                        return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch("".to_string()))
                }
            }
        }
 
        pub fn op_and(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
           // Check that there are exactly two inputs
            if inputs.len() != 2 {
                return Err(OrtError::TypeMismatch("And requires exactly two boolean tensors".into()));
            }

            // Extract the first input and verify it's a boolean tensor
            let input1 = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Missing first input".into()))?;
            if !matches!(input1.dtype(), DataType::Boolean) {
                return Err(OrtError::TypeMismatch("First input must be a boolean tensor".into()));
            }

            // Extract the second input and verify it's a boolean tensor
            let input2 = inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Missing second input".into()))?;
            if !matches!(input2.dtype(), DataType::Boolean) {
                return Err(OrtError::TypeMismatch("Second input must be a boolean tensor".into()));
            }

            // Convert inputs to ndarrays
            let array1 = ort_to_ndarray(input1)?;
            let array2 = ort_to_ndarray(input2)?;
            let result: OrtResult<OrtValue>=match (array1) {
                ArrayDResult::Boolean(a1) => {
                    return match (array2) {
                        ArrayDResult::Boolean(a2) =>{
                            Ok(ndarray_to_ort(ArrayDResult::Boolean(ndarray::Zip::from(&a1).and(&a2).map_collect(|&a, &b| a && b)),DataType::Boolean))
                        },
                        _=> Err(OrtError::TypeMismatch("can only work with booleans".to_string()))
                    };
                        // let result = arrayinp.mapv(|x| x.exp());
                        // return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch("can only work with booleans".to_string()))
                }
            };
            result
            // Handle broadcasting
            // let result = match (array1.shape(), array2.shape()) {
            //     (shape1, shape2) if shape1 == shape2 => {
            //         // No broadcasting needed; perform element-wise AND
            //         ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| (a != 0) && (b != 0) as u8)
            //     }
            //     _ => {
            //         // Perform broadcasting
            //         let output_shape = broadcast_shapes(array1.shape(), array2.shape())?;
            //         let array1_b = array1.broadcast(output_shape.clone()).ok_or_else(|| {
            //             OrtError::TypeMismatch("Failed to broadcast first input".into())
            //         })?;
            //         let array2_b = array2.broadcast(output_shape.clone()).ok_or_else(|| {
            //             OrtError::TypeMismatch("Failed to broadcast second input".into())
            //         })?;
            //         ndarray::Zip::from(array1_b).and(array2_b).map_collect(|&a, &b| (a != 0) && (b != 0) as u8)
            //     }
            // };

            // Convert result back to OrtValue with DataType::Boolean
            // Ok(ndarray_to_ort(result, DataType::Boolean))
        }

        pub fn op_cast(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let to = node.attributes.iter().find(|a| a.name == "to")
                .map(|a| a.i)
                .ok_or_else(|| OrtError::InvalidTensorData("Cast requires 'to' attribute".into()))?;
            
            // Get optional attributes
            // let saturate = node.attributes.iter().find(|a| a.name == "saturate")
            //     .map(|a| a.i == 1)
            //     .unwrap_or(true);
            
            // let round_mode = node.attributes.iter().find(|a| a.name == "round_mode")
            //     .map(|a| a.s.clone())
            //     .unwrap_or_else(|| "up".to_string());
            
            let tensor = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Cast requires one tensor".to_string()))?;
            
            // Convert based on the 'to' attribute value
            match tensor {
                OrtValue::Tensor { shape, dtype, data } => {
                    // Map 'to' value to DataType
                    let target_dtype = match to {
                        1 => DataType::Float,
                        7 => DataType::Int64,
                        9 => DataType::Boolean,
                        // Add more mappings as needed
                        _ => return {
                            let tow=to.clone();
                            Err(OrtError::TypeMismatch(("Unsupported cast to type".to_string())))
                        },
                    };
                    
                    // Handle the conversion based on source and target types
                    match (dtype, target_dtype) {
                        (DataType::Float, DataType::Int64) => {
                            let float_data: Vec<f32> = data.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                            let int_data: Vec<u8> = float_data.iter().flat_map(|&f| (f as i64).to_le_bytes()).collect();
                    Ok(OrtValue::Tensor {
                        shape: shape.clone(),
                                dtype: DataType::Int64,
                                data: Arc::new(int_data),
                    })
                        },
                        (DataType::Int64, DataType::Float) => {
                            let int_data: Vec<i64> = data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
                            let float_data: Vec<u8> = int_data.iter().flat_map(|&i| (i as f32).to_le_bytes()).collect();
                            Ok(OrtValue::Tensor {
                                shape: shape.clone(),
                                dtype: DataType::Float,
                                data: Arc::new(float_data),
                            })
                        },
                        (DataType::Float, DataType::Boolean) => {
                            let float_data: Vec<f32> = data.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                            let bool_data: Vec<u8> = float_data.iter().map(|&f| if f == 0.0 { 0 } else { 1 }).collect();
                            Ok(OrtValue::Tensor {
                                shape: shape.clone(),
                                dtype: DataType::Boolean,
                                data: Arc::new(bool_data),
                            })
                        },
                        (DataType::Boolean, DataType::Float) => {
                            let bool_data: Vec<bool> = data.iter().map(|&b| b != 0).collect();
                            let float_data: Vec<u8> = bool_data.iter().flat_map(|&b| (if b { 1.0f32 } else { 0.0f32 }).to_le_bytes()).collect();
                            Ok(OrtValue::Tensor {
                                shape: shape.clone(),
                                dtype: DataType::Float,
                                data: Arc::new(float_data),
                            })
                        },
                        _ => Err(OrtError::TypeMismatch(("Unsupported cast".to_string()))),
            }
                },
                _ => Err(OrtError::TypeMismatch("Input must be a tensor".to_string())),
            }
        }

        pub fn op_clip(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
                    // Get the input tensor
                    let input = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Clip requires at least one tensor".to_string()))?;
                    let dtype = match input {
                        OrtValue::Tensor { dtype, .. } => *dtype,
                        _ => return Err(OrtError::TypeMismatch("Input must be a tensor".to_string())),
                    };

                    // Check if the data type is supported
                    if !is_numeric_dtype(dtype) {
                        return Err(OrtError::TypeMismatch(format!("Unsupported data type for Clip: {:?}", dtype)));
                    }

                    // Convert input to ndarray
                    let array = ort_to_ndarray(input)?;

                    // Get min and max values (if provided)
                    let min_value = if inputs.len() > 1  {
                        match ort_to_ndarray(&inputs[1])? {
                            ArrayDResult::Float(arr) if arr.len() == 1 => Some(arr[ndarray::IxDyn(&[])]),
                            ArrayDResult::Int32(arr) if arr.len() == 1 => Some(arr[ndarray::IxDyn(&[])] as f32),
                            ArrayDResult::Int64(arr) if arr.len() == 1 => Some(arr[ndarray::IxDyn(&[])] as f32),
                            _ => return Err(OrtError::InvalidTensorData("Min value must be a scalar".into())),
                        }
                    } else {
                        None
                    };

                    let max_value = if inputs.len() > 2  {
                        match ort_to_ndarray(&inputs[2])? {
                            ArrayDResult::Float(arr) if arr.len() == 1 => Some(arr[ndarray::IxDyn(&[])]),
                            ArrayDResult::Int32(arr) if arr.len() == 1 => Some(arr[ndarray::IxDyn(&[])] as f32),
                            ArrayDResult::Int64(arr) if arr.len() == 1 => Some(arr[ndarray::IxDyn(&[])] as f32),
                            _ => return Err(OrtError::InvalidTensorData("Max value must be a scalar".into())),
                        }
                    } else {
                        None
                    };

                    // Apply clipping based on data type
                    match array {
                        ArrayDResult::Float(arr) => {
                            let min = min_value.unwrap_or(f32::MIN);
                            let max = max_value.unwrap_or(f32::MAX);
                            let clipped = arr.mapv(|x| x.max(min).min(max));
                            Ok(ndarray_to_ort(ArrayDResult::Float(clipped), dtype))
                        },
                        ArrayDResult::Int32(arr) => {
                            let min = min_value.map(|v| v as i32).unwrap_or(i32::MIN);
                            let max = max_value.map(|v| v as i32).unwrap_or(i32::MAX);
                            let clipped = arr.mapv(|x| x.max(min).min(max));
                            Ok(ndarray_to_ort(ArrayDResult::Int32(clipped), dtype))
                        },
                        ArrayDResult::Int64(arr) => {
                            let min = min_value.map(|v| v as i64).unwrap_or(i64::MIN);
                            let max = max_value.map(|v| v as i64).unwrap_or(i64::MAX);
                            let clipped = arr.mapv(|x| x.max(min).min(max));
                            Ok(ndarray_to_ort(ArrayDResult::Int64(clipped), dtype))
                        },
                        _ => Err(OrtError::TypeMismatch(format!("Unsupported data type for Clip: {:?}", dtype))),
                    }
            
                }
    

                pub fn op_concat(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
// Get the axis attribute
let axis = node.attributes.iter()
    .find(|a| a.name == "axis")
    .map(|a| a.i)
    .ok_or_else(|| OrtError::InvalidTensorData("Concat requires 'axis' attribute".into()))?;

// Check if we have at least one input
if inputs.is_empty() {
    return Err(OrtError::InvalidTensorData("Concat requires at least one input".into()));
}

// Get the data type from the first input
let dtype = match inputs.first() {
    Some(OrtValue::Tensor { dtype, .. }) => *dtype,
    _ => return Err(OrtError::TypeMismatch("First input must be a tensor".into())),
};

// Check that all inputs have the same data type
for input in inputs.iter() {
    match input {
        OrtValue::Tensor { dtype: input_dtype, .. } if *input_dtype == dtype => {},
        OrtValue::Tensor { .. } => return Err(OrtError::TypeMismatch("All inputs must have the same data type".into())),
        _ => return Err(OrtError::TypeMismatch("All inputs must be tensors".into())),
    }
}

// Convert all inputs to ndarrays
let arrays: Vec<_> = inputs.iter()
    .map(|input| ort_to_ndarray(input))
    .collect::<OrtResult<Vec<_>>>()?;

// Get the rank of the first array to validate axis
let first_array_shape = match &arrays[0] {
    ArrayDResult::Float(arr) => arr.shape(),
    ArrayDResult::Int32(arr) => arr.shape(),
    ArrayDResult::Int64(arr) => arr.shape(),
    ArrayDResult::Boolean(arr) => arr.shape(),
    // Add other types as needed
};

let rank = first_array_shape.len();
let normalized_axis = if axis < 0 {
    (rank as i64 + axis) as usize
} else {
    axis as usize
};

if normalized_axis >= rank {
    return Err(OrtError::InvalidTensorData(format!(
        "Axis {} is out of bounds for array of rank {}",
        axis, rank
    )));
}

// Perform concatenation based on data type
match &arrays[0] {
    ArrayDResult::Float(_) => {
        let float_arrays: Vec<_> = arrays.into_iter()
            .map(|arr| match arr {
                ArrayDResult::Float(a) => Ok(a),
                _ => Err(OrtError::TypeMismatch("Expected float array".into())),
            })
            .collect::<OrtResult<_>>()?;
        
        let result = ndarray::concatenate(
            ndarray::Axis(normalized_axis),
            float_arrays.iter().map(|a| a.view()).collect::<Vec<_>>().as_slice(),
        ).map_err(|e| OrtError::InvalidTensorData(format!("Failed to concatenate arrays: {:?}", e)))?;
        
        Ok(ndarray_to_ort(ArrayDResult::Float(result), dtype))
    },
    ArrayDResult::Int32(_) => {
        let int_arrays: Vec<_> = arrays.into_iter()
            .map(|arr| match arr {
                ArrayDResult::Int32(a) => Ok(a),
                _ => Err(OrtError::TypeMismatch("Expected int32 array".into())),
            })
            .collect::<OrtResult<_>>()?;
        
        let result = ndarray::concatenate(
            ndarray::Axis(normalized_axis),
            int_arrays.iter().map(|a| a.view()).collect::<Vec<_>>().as_slice(),
        ).map_err(|e| OrtError::InvalidTensorData(format!("Failed to concatenate arrays: {:?}", e)))?;
        
        Ok(ndarray_to_ort(ArrayDResult::Int32(result), dtype))
    },
    ArrayDResult::Int64(_) => {
        let int_arrays: Vec<_> = arrays.into_iter()
            .map(|arr| match arr {
                ArrayDResult::Int64(a) => Ok(a),
                _ => Err(OrtError::TypeMismatch("Expected int64 array".into())),
            })
            .collect::<OrtResult<_>>()?;
        
        let result = ndarray::concatenate(
            ndarray::Axis(normalized_axis),
            int_arrays.iter().map(|a| a.view()).collect::<Vec<_>>().as_slice(),
        ).map_err(|e| OrtError::InvalidTensorData(format!("Failed to concatenate arrays: {:?}", e)))?;
        
        Ok(ndarray_to_ort(ArrayDResult::Int64(result), dtype))
    },
    ArrayDResult::Boolean(_) => {
        let bool_arrays: Vec<_> = arrays.into_iter()
            .map(|arr| match arr {
                ArrayDResult::Boolean(a) => Ok(a),
                _ => Err(OrtError::TypeMismatch("Expected boolean array".into())),
            })
            .collect::<OrtResult<_>>()?;
        
        let result = ndarray::concatenate(
            ndarray::Axis(normalized_axis),
            bool_arrays.iter().map(|a| a.view()).collect::<Vec<_>>().as_slice(),
        ).map_err(|e| OrtError::InvalidTensorData(format!("Failed to concatenate arrays: {:?}", e)))?;
        
        Ok(ndarray_to_ort(ArrayDResult::Boolean(result), dtype))
    },
    // Add other types as needed
}
                    
                }

        pub fn op_constant_of_shape(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
// Get the shape from the input tensor
let shape_tensor = inputs.get(0).ok_or_else(|| 
    OrtError::InvalidTensorData("ConstantOfShape requires one input tensor".into()))?;

// Verify the input is an int64 tensor
let shape_array = match ort_to_ndarray(shape_tensor)? {
    ArrayDResult::Int64(arr) => arr,
    _ => return Err(OrtError::TypeMismatch("ConstantOfShape input must be an int64 tensor".into())),
};

// Convert shape array to Vec<i64>
let shape: Vec<i64> = shape_array.iter().cloned().collect();

// Check that all dimensions are non-negative
if shape.iter().any(|&dim| dim < 0) {
    return Err(OrtError::InvalidTensorData("All dimensions must be non-negative".into()));
}

// Get the value attribute if provided
let value = node.attributes.iter()
    .find(|a| a.name == "value")
    .map(|a| &a.t);

// Determine the data type and value to use
let (dtype, value_data) = if let Some(value_tensor) = value {
    // Use the provided value tensor
let value_tensor = match value_tensor {
    Some(tensor) => tensor,
    None => return Err(OrtError::InvalidTensorData("Value tensor is missing".into())),
};
    let value_dtype = DataType::try_from(value_tensor.data_type).unwrap();
    let value_data = value_tensor.raw_data.clone();
    
    // Verify the value tensor has exactly one element
    if value_tensor.dims.iter().product::<i64>() != 1 {
        return Err(OrtError::InvalidTensorData("Value attribute must be a one-element tensor".into()));
    }
    
    (value_dtype, value_data)
} else {
    // Default to float32 with value 0
    let default_value = 0.0f32;
    let value_data = default_value.to_le_bytes().to_vec();
    (DataType::Float, value_data)
};

// Calculate the total number of elements
let total_elements: i64 = if shape.is_empty() {
    // Empty shape means a scalar output
    1
} else {
    shape.iter().product()
};

// Create the output data by repeating the value
let element_size = match dtype {
    DataType::Float => 4,
    DataType::Int64 => 8,
    DataType::Int32 => 4,
    DataType::Boolean => 1,
    // Add other types as needed
    _ => return Err(OrtError::TypeMismatch(format!("Unsupported data type: {:?}", dtype))),
};

let output_data = value_data.chunks(element_size)
    .next()
    .ok_or_else(|| OrtError::InvalidTensorData("Value data is empty".into()))?
    .to_vec()
    .repeat(total_elements as usize);

// Create the output tensor
Ok(OrtValue::Tensor {
    shape: shape.iter().map(|&d|Dimensions::Fixed(d as usize)).collect(),
    dtype,
    data: Arc::new(output_data),
})
            
        }


        pub fn op_conv(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
// Check if we have the required inputs
if inputs.len() < 2 {
    return Err(OrtError::InvalidTensorData("Conv requires at least input and weight tensors".into()));
}

// Get input tensor X and weight tensor W
let x = inputs.get(0).ok_or_else(|| OrtError::InvalidTensorData("Missing input tensor X".into()))?;
let w = inputs.get(1).ok_or_else(|| OrtError::InvalidTensorData("Missing weight tensor W".into()))?;

// Get optional bias tensor B
let b = inputs.get(2);

// Check data types
let dtype = match x {
    OrtValue::Tensor { dtype, .. } => *dtype,
    _ => return Err(OrtError::TypeMismatch("Input X must be a tensor".into())),
};

// Only support float for now
if dtype != DataType::Float {
    return Err(OrtError::TypeMismatch("Conv only supports float tensors currently".into()));
}

// Convert tensors to ndarrays
let x_array = match ort_to_ndarray(x)? {
    ArrayDResult::Float(arr) => arr,
    _ => return Err(OrtError::TypeMismatch("Input X must be a float tensor".into())),
};

let w_array = match ort_to_ndarray(w)? {
    ArrayDResult::Float(arr) => arr,
    _ => return Err(OrtError::TypeMismatch("Weight W must be a float tensor".into())),
};

let b_array = if let Some(b) = b {
    match ort_to_ndarray(b)? {
        ArrayDResult::Float(arr) => Some(arr),
        _ => return Err(OrtError::TypeMismatch("Bias B must be a float tensor".into())),
    }
} else {
    None
};

// Get attributes
let auto_pad = node.attributes.iter()
    .find(|a| a.name == "auto_pad")
    .map(|a| a.s.clone())
    // .unwrap();
    .unwrap_or_else(|| "NOTSET".as_bytes().to_vec());

let group = node.attributes.iter()
    .find(|a| a.name == "group")
    .map(|a| a.i as usize)
    .unwrap_or(1);

let dilations = node.attributes.iter()
    .find(|a| a.name == "dilations")
    .map(|a| a.ints.clone())
    .unwrap_or_else(|| vec![1, 1]);

let strides = node.attributes.iter()
    .find(|a| a.name == "strides")
    .map(|a| a.ints.clone())
    .unwrap_or_else(|| vec![1, 1]);

let kernel_shape = node.attributes.iter()
    .find(|a| a.name == "kernel_shape")
    .map(|a| a.ints.clone())
    .unwrap_or_else(|| {
        // Infer from weight tensor if not provided
        let w_shape = w_array.shape();
        if w_shape.len() >= 2 {
            w_shape[2..].iter().map(|&d| d as i64).collect()
        } else {
            vec![1, 1] // Default for 2D
        }
    });

let pads = if auto_pad == "NOTSET".as_bytes().to_vec() {
    node.attributes.iter()
        .find(|a| a.name == "pads")
        .map(|a| a.ints.clone())
        .unwrap_or_else(|| {
            // Default padding is 0 for all dimensions
            let spatial_dims = kernel_shape.len();
            vec![0; spatial_dims * 2]
        })
} else {
    // For SAME_UPPER, SAME_LOWER, VALID, we'll calculate padding later
    vec![]
};

// Extract dimensions
let x_shape = x_array.shape();
let w_shape = w_array.shape();

// Basic validation
if x_shape.len() < 3 || w_shape.len() < 3 {
    return Err(OrtError::InvalidTensorData("Input and weight tensors must have at least 3 dimensions".into()));
}

let batch_size = x_shape[0];
let input_channels = x_shape[1];
let output_channels = w_shape[0];

// Check that input channels match weight channels * group
if input_channels != w_shape[1] * group {
    return Err(OrtError::InvalidTensorData(
        format!("Input channels ({}) must equal weight channels ({}) * group ({})",
                input_channels, w_shape[1], group).into()));
}

// Check that output channels is divisible by group
if output_channels % group != 0 {
    return Err(OrtError::InvalidTensorData(
        format!("Output channels ({}) must be divisible by group ({})",
                output_channels, group).into()));
}

// For simplicity, we'll implement 2D convolution only
if x_shape.len() != 4 || w_shape.len() != 4 {
    return Err(OrtError::InvalidTensorData("This implementation only supports 2D convolution".into()));
}

let input_height = x_shape[2];
let input_width = x_shape[3];
let kernel_height = kernel_shape[0] as usize;
let kernel_width = kernel_shape[1] as usize;
let stride_h = strides[0] as usize;
let stride_w = strides[1] as usize;
let dilation_h = dilations[0] as usize;
let dilation_w = dilations[1] as usize;

// Calculate output dimensions and padding
let (output_height, output_width, pad_h_begin, pad_h_end, pad_w_begin, pad_w_end) = match String::from_utf8(auto_pad.clone()).unwrap().to_string().as_str() {
    "NOTSET" => {
        if pads.len() != 4 {
            return Err(OrtError::InvalidTensorData("For NOTSET auto_pad, pads must have 4 values".into()));
        }
        let pad_h_begin = pads[0] as usize;
        let pad_w_begin = pads[1] as usize;
        let pad_h_end = pads[2] as usize;
        let pad_w_end = pads[3] as usize;
        
        let output_height = (input_height + pad_h_begin + pad_h_end - (kernel_height - 1) * dilation_h - 1) / stride_h + 1;
        let output_width = (input_width + pad_w_begin + pad_w_end - (kernel_width - 1) * dilation_w - 1) / stride_w + 1;
        
        (output_height, output_width, pad_h_begin, pad_h_end, pad_w_begin, pad_w_end)
    },
    "VALID" => {
        let output_height = (input_height - (kernel_height - 1) * dilation_h - 1) / stride_h + 1;
        let output_width = (input_width - (kernel_width - 1) * dilation_w - 1) / stride_w + 1;
        (output_height, output_width, 0, 0, 0, 0)
    },
    "SAME_UPPER" | "SAME_LOWER" => {
        let output_height = (input_height + stride_h - 1) / stride_h;
        let output_width = (input_width + stride_w - 1) / stride_w;
        
        let pad_h_needed = (output_height - 1) * stride_h + (kernel_height - 1) * dilation_h + 1 - input_height;
        let pad_w_needed = (output_width - 1) * stride_w + (kernel_width - 1) * dilation_w + 1 - input_width;
        
        let (pad_h_begin, pad_h_end) = if String::from_utf8(auto_pad.clone()).unwrap().to_string().as_str() == "SAME_UPPER" {
            (pad_h_needed / 2, pad_h_needed - pad_h_needed / 2)
        } else {
            (pad_h_needed - pad_h_needed / 2, pad_h_needed / 2)
        };
        
        let (pad_w_begin, pad_w_end) = if String::from_utf8(auto_pad).unwrap().to_string().as_str() == "SAME_UPPER" {
            (pad_w_needed / 2, pad_w_needed - pad_w_needed / 2)
        } else {
            (pad_w_needed - pad_w_needed / 2, pad_w_needed / 2)
        };
        
        (output_height, output_width, pad_h_begin, pad_h_end, pad_w_begin, pad_w_end)
    },
    _ => return Err(OrtError::InvalidTensorData(format!("Unsupported auto_pad value: {:?}", auto_pad).into())),
};

// Create output array
let mut output = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[batch_size, output_channels, output_height, output_width]));

// Perform convolution
for n in 0..batch_size {
    for g in 0..group {
        let oc_per_group = output_channels / group;
        let ic_per_group = input_channels / group;
        
        for oc_within_group in 0..oc_per_group {
            let oc = g * oc_per_group + oc_within_group;
            
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut sum = 0.0;
                    
                    for ic_within_group in 0..ic_per_group {
                        let ic = g * ic_per_group + ic_within_group;
                        
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride_h + kh * dilation_h;
                                let iw = ow * stride_w + kw * dilation_w;
                                
                                // Apply padding
                                let ih_padded = ih as isize - pad_h_begin as isize;
                                let iw_padded = iw as isize - pad_w_begin as isize;
                                
                                // Check if the input position is valid
                                if ih_padded >= 0 && ih_padded < input_height as isize && 
                                   iw_padded >= 0 && iw_padded < input_width as isize {
                                    sum += x_array[[n, ic, ih_padded as usize, iw_padded as usize]] * 
                                           w_array[[oc, ic_within_group, kh, kw]];
                                }
                            }
                        }
                    }
                    
                    // Add bias if present
                    if let Some(ref b_arr) = b_array {
                        sum += b_arr[oc];
                    }
                    
                    output[[n, oc, oh, ow]] = sum;
                }
            }
        }
    }
}

// Convert result back to OrtValue
Ok(ndarray_to_ort(ArrayDResult::Float(output), dtype))
            
        }
    }

// Helper function to check if a data type is numeric
fn is_numeric_dtype(dtype: DataType) -> bool {
    matches!(
        dtype,
        // DataType::BFloat16
        //     | DataType::Double
            | DataType::Float
            // | DataType::Float16
            // | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            // | DataType::Int8
            // | DataType::UInt16
            // | DataType::UInt32
            // | DataType::UInt64
            // | DataType::UInt8
    )
}
    // Helper function to extract a scalar value from a tensor
// fn extract_scalar(tensor: &OrtValue, expected_dtype: DataType) -> OrtResult<ScalarValue> {
//     if !tensor.shape().is_empty() {
//         return Err(OrtError::InvalidInputShape(
//             "Clip min/max inputs must be scalar (empty shape)".into(),
//         ));
//     }
//     if *tensor.dtype() != expected_dtype {
//         return Err(OrtError::TypeMismatch(
//             format!("Clip min/max must match input type {:?}", expected_dtype),
//         ));
//     }
//     // Extract scalar value (implementation depends on OrtValue structure)
//     match *tensor.dtype() {
//         DataType::Float => Ok(ScalarValue::Float(tensor.get_scalar::<f32>()?)),
//         // DataType::Double => Ok(ScalarValue::Double(tensor.get_scalar::<f64>()?)),
//         DataType::Int32 => Ok(ScalarValue::Int32(tensor.get_scalar::<i32>()?)),
//         DataType::Int64 => Ok(ScalarValue::Int64(tensor.get_scalar::<i64>()?)),
//         // Add other types as needed
//         _ => Err(OrtError::TypeMismatch(
//             format!("Unsupported scalar type for Clip: {:?}", tensor.dtype()),
//         )),
//     }
// }

// Enum to hold scalar values of different types
// #[derive(Clone, Copy)]
// enum ScalarValue {
//     Float(f32),
//     Double(f64),
//     Int32(i32),
//     Int64(i64),
//     // Add other types as needed
// }

// impl ScalarValue {
//     fn as_f32(&self) -> Option<f32> {
//         match *self {
//             ScalarValue::Float(v) => Some(v),
//             _ => None,
//         }
//     }
//     fn as_f64(&self) -> Option<f64> {
//         match *self {
//             ScalarValue::Double(v) => Some(v),
//             _ => None,
//         }
//     }
//     fn as_i32(&self) -> Option<i32> {
//         match *self {
//             ScalarValue::Int32(v) => Some(v),
//             _ => None,
//         }
//     }
//     fn as_i64(&self) -> Option<i64> {
//         match *self {
//             ScalarValue::Int64(v) => Some(v),
//             _ => None,
//         }
//     }
// }