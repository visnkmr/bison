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
                    "Sub requires two tensors with matching data types",
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
                    "Mul requires two tensors with matching data types",
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
                    "Div requires two tensors with matching data types",
                ))
            }
        };

        let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        let result = array1 / array2;
        Ok(ndarray_to_ort(result.unwrap(), *dtype))
    }
            
    pub fn op_pow(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let input1 = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Pow requires two tensors"))?;
            let input2 = inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Pow requires two tensors"))?;

            match (input1, input2) {
                (OrtValue::Tensor { dtype: dtype1, .. }, OrtValue::Tensor { dtype: dtype2, .. }) 
                if dtype1 == dtype2 => {
                   
            if let Ok(result)=pow_array(&ort_to_ndarray(input1)?, &ort_to_ndarray(input2)?){
                Ok(ndarray_to_ort(result, *dtype1))
            }
            else{
                Err(OrtError::TypeMismatch("Pow requires two tensors with matching data types"))
            }
                },
                _ => Err(OrtError::TypeMismatch("Pow requires two tensors with matching data types")),
            }
        
    }
    
        pub fn op_sqrt(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let input = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Sqrt requires one tensor"))?;
        match (input) {
            (OrtValue::Tensor { dtype: dtype1, .. })=>{
                if let Ok(result)=sqrt_array(&ort_to_ndarray(input)?){
                    Ok(ndarray_to_ort(result, *dtype1))
                }
                else{
                    Err(OrtError::TypeMismatch("Pow requires two tensors with matching data types"))
                }

            }
            _=>{
                Err(OrtError::TypeMismatch("Pow requires two tensors with matching data types"))
            }
        }
    }
    
        // Element-wise Operations
        pub fn op_exp(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let a = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Exp requires one tensor"))?;

            return match (ort_to_ndarray(a)?) {
                ArrayDResult::Float(arrayinp) => {
                        let result = arrayinp.mapv(|x| x.exp());
                        return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch(""))
                }
            }
        }

        pub fn op_sigmoid(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Sigmoid requires one float tensor"))?)?;
            return match (array) {
                ArrayDResult::Float(arrayinp) => {
                        let result = arrayinp.mapv(|a| 1.0 / (1.0 + (-a).exp()));
                        return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch(""))
                }
            }
        }


        pub fn op_tanh(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Tanh requires one float tensor"))?)?;
            return match (array) {
                ArrayDResult::Float(arrayinp) => {
                        let result = arrayinp.mapv(|a|a.tanh());
                        return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch(""))
                }
            }
        }
    
        pub fn op_sin(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Sin requires one float tensor"))?)?;
            return match (array) {
                ArrayDResult::Float(arrayinp) => {
                        let result = arrayinp.mapv(|a|a.sin());
                        return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch(""))
                }
            }
        }
    
        pub fn op_cos(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Cos requires one float tensor"))?)?;
            return match (array) {
                ArrayDResult::Float(arrayinp) => {
                        let result = arrayinp.mapv(|a|a.cos());
                        return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch(""))
                }
            }
        }
    
        pub fn op_atan(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Atan requires one float tensor"))?)?;
            return match (array) {
                ArrayDResult::Float(arrayinp) => {
                        let result = arrayinp.mapv(|a| a.atan());
                        return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch(""))
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
                        _=> Err(OrtError::TypeMismatch(""))
                    };
                        // let result = arrayinp.mapv(|x| x.exp());
                        // return Ok(ndarray_to_ort(ArrayDResult::Float(result),DataType::Float))
                }
                
                _ => {
                    Err(OrtError::TypeMismatch(""))
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
            
            let tensor = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Cast requires one tensor"))?;
            
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
                            Err(OrtError::TypeMismatch(&("Unsupported cast to type")))
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
                        _ => Err(OrtError::TypeMismatch(&("Unsupported cast"))),
            }
                },
                _ => Err(OrtError::TypeMismatch("Input must be a tensor")),
            }
        }
}