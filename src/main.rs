// #![recursion_limit = "5684"]
#![feature(try_trait_v2)]
use std::sync::atomic::{AtomicUsize, Ordering};
mod core_ops;
// mod core_ops2;
pub use core_ops::*;
mod optimized_ops;
pub use optimized_ops::*;
mod lstm_optimized;
pub use lstm_optimized::*;
// mod debug_tensor_map;
// pub use debug_tensor_map::*;
// pub mod resumption_example;
#[cfg(test)]
mod tests {
    // pub mod tensor_ops_test;
    // pub mod parser_test;
    // pub mod onnx_model_test;
    // pub mod sequence_map_test;
    // pub mod bert_ops_test;
    // pub mod kokoro_ops_test;
    // pub mod resumption_test;
}

#[test]
fn printall(){
     print_model_info("./kokoro-v1.0.onnx");

}

// mod test_optimizations;

// Static counter for node indexing
// static NODE_COUNTER: AtomicUsize = AtomicUsize::new(0);
use anyhow::Result;
use indexmap::IndexMap;
use prost::Message;
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use ndarray_npy::NpzReader;
use std::{
    collections::{HashMap, HashSet},
    fmt,
    fs::File,
    io::{Read, Write},
    path::Path,
    sync::Arc,
};
use thiserror::Error;
use ndarray::{Array, Array2, Array3, Array4, ArrayD, Axis, Dimension, IxDyn, Zip};
use num_traits::{Float, ToBytes};

use crate::convert::ArrayDResult;

// Function to print tensor information in Python format
fn print_tensor_info(output_name: &str, tensor_value: &OrtValue) {
    match tensor_value {
        OrtValue::Tensor { shape, dtype, data } => {
            // Convert shape to display format
            let shape_str = shape.iter()
                .map(|dim| match dim {
                    Dimensions::Fixed(size) => size.to_string(),
                    Dimensions::Symbolic(name) => name.clone(),
                })
                .collect::<Vec<_>>()
                .join(", ");
            
            // Get data type string
            let dtype_str = match dtype {
                DataType::Float => "float32",
                DataType::Int64 => "int64", 
                DataType::Int32 => "int32",
                DataType::String => "string",
                DataType::Boolean => "bool",
            };
            
            // Get sample values (first 3 elements)
            let sample_values = get_sample_values(data, dtype,false);
            
            println!("Node '{}':", output_name);
            println!("  Shape: ({})", shape_str);
            println!("  Data type: {}", dtype_str);
            println!("  Sample values: {}", sample_values);
        }
        _ => {
            println!("Node '{}':", output_name);
            println!("  Type: Non-tensor value");
        }
    }
}

// Helper function to extract sample values from tensor data
fn get_sample_values(data: &Arc<Vec<u8>>, dtype: &DataType,fulloutput:bool) -> String {
    if data.is_empty() {
        return "Empty".to_string();
    }
    
    // match dtype {
    //     DataType::Float => {
    //         // let float_data = bytemuck::cast_slice::<u8, f32>(data);
    //         // let samples: Vec<String> = float_data.iter()
    //         //     .take(3)
    //         //     .map(|x| format!("{:?}", x))
    //         //     .collect();
    //         // format!("[{}]", samples.join(" "))
    //         format!("{:?}",data)
    //     }
    //     DataType::Int64 => {
    //         let int_data = bytemuck::cast_slice::<u8, i64>(data);
    //         let samples: Vec<String> = int_data.iter()
    //             .take(3)
    //             .map(|x| x.to_string())
    //             .collect();
    //         format!("[{}]", samples.join(" "))
    //     }
    //     DataType::Int32 => {
    //         let int_data = bytemuck::cast_slice::<u8, i32>(data);
    //         let samples: Vec<String> = int_data.iter()
    //             .take(3)
    //             .map(|x| x.to_string())
    //             .collect();
    //         format!("[{}]", samples.join(" "))
    //     }
    //     DataType::String => {
    //         "String data".to_string()
    //     }
    //     DataType::Boolean => {
    //         let samples: Vec<String> = data.iter()
    //             .take(3)
    //             .map(|&x| if x != 0 { "true" } else { "false" }.to_string())
    //             .collect();
    //         format!("[{}]", samples.join(" "))
    //     }
    // }
    
    match dtype {
        DataType::Float => {
            let float_data: Vec<f32> = data
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let takehowmany=if fulloutput{
                float_data.len()
            }
            else{
                3
            };
            let samples: Vec<String> = float_data.iter()
                .take(3)
                .map(|x| format!("{:?}", x))
                .collect();
            
                format!("[{}]", samples.join(" "))
        }
        DataType::Int64 => {
            let int_data: Vec<i64> = data
                .chunks(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
        let samples: Vec<String> = int_data.iter()
                .take(3)
                .map(|x| x.to_string())
                .collect();
            format!("[{}]", samples.join(" "))
        }
        DataType::Int32 => {
            let int_data: Vec<i32> = data
                .chunks(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let samples: Vec<String> = int_data.iter()
                        .take(3)
                        .map(|x| x.to_string())
                        .collect();
            format!("[{}]", samples.join(" "))
        }
        DataType::Boolean=>{
        let samples: Vec<String> = data.iter()
            .take(3)
            .map(|&x| if x != 0 { "true" } else { "false" }.to_string())
            .collect();
        format!("[{}]", samples.join(" "))
            
        }
        _ => {
            "Unknown data type".to_string()
        }
    }
}

// use crate::ndarray_to_ort::SupportedType;

// Custom serialization for Arc<Vec<u8>>
mod serde_arc_vec {
    use super::*;
    
    pub fn serialize<S>(data: &Arc<Vec<u8>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        data.as_ref().serialize(serializer)
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<Vec<u8>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec = Vec::<u8>::deserialize(deserializer)?;
        Ok(Arc::new(vec))
    }
}

// Simplified ONNX type definitions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    Float,
    Int64,
    Int32,
    String,
    Boolean
}

mod convert;
pub struct OrtEngine {
    model: ModelProto,
    node_registry: HashMap<String, fn(&NodeProto, &[OrtValue]) -> OrtResult<OrtValue>>,
    vendor_ops: HashMap<String, fn(&[u8], &[OrtValue]) -> OrtResult<OrtValue>>,
    shape_inference: ShapeInference, // Added
}


#[derive(Debug, Clone, Default)]
pub struct TensorProto {
    pub name: String,
    pub doc_string:String,
    pub data_type: i32,
    pub dims: Vec<i64>,
    pub float_data: Vec<f32>,
    pub int64_data: Vec<i64>,
    pub int32_data: Vec<i32>,
    pub string_data: Vec<Vec<u8>>,
    pub raw_data: Vec<u8>,
}



#[derive(Debug, Clone, Default)]
pub struct AttributeProto {
    pub name: String,        // Name of the attribute
    pub i: i64,              // Integer value
    pub f: f32,              // Float value
    pub s: Vec<u8>,          // String/bytes value
    pub t: Option<TensorProto>, // Tensor value
    pub ints: Vec<i64>,      // List of integers
    pub floats: Vec<f32>,    // List of floats
    pub g: Option<GraphProto>, // Subgraph value (for control flow operators)
    pub strings: Vec<Vec<u8>>
}


#[derive(Debug, Clone, Default)]
pub struct NodeProto {
    pub input: Vec<String>,
    pub output: Vec<String>,
    pub op_type: String,
    pub attributes: Vec<AttributeProto>,
    pub name: String, // Added for tag 3
    pub domain: String, // Added for tag 5
    pub subgraphs: HashMap<String, GraphProto>, // Added for subgraphs
}

#[derive(Debug, Clone, Default)]
pub struct GraphProto {
    pub node: Vec<NodeProto>,
    pub initializer: Vec<TensorProto>,
    pub output: Vec<ValueInfoProto>,
    pub name: String, // Added for tag 2
    pub doc_string: String, // Added for tag 5
    pub input: Vec<ValueInfoProto>, // Added for tag 12
    pub value_info: Vec<ValueInfoProto>, // Added for tag 13
}

#[derive(Debug, Clone, Default)]
pub struct ValueInfoProto {
    pub name: String,
    pub type_proto: Option<TypeProto>, // Added for tag 2
}
// Minimal TypeProto for tensor type information
#[derive(Debug, Clone, Default)]
pub struct TypeProto {
    pub tensor_type: Option<TensorTypeProto>,
}
#[derive(Debug, Clone, Default)]
pub struct TensorTypeProto {
    pub elem_type: i32, // Data type (e.g., 1 for float, 7 for int64)
    pub shape: Option<TensorShapeProto>,
}

#[derive(Debug, Clone, Default)]
pub struct TensorShapeProto {
    pub dim: Vec<TensorDimension>,
}

#[derive(Debug, Clone, Default)]
pub struct TensorDimension {
    pub dim_value: i64, // Specific dimension value (or 0 if symbolic)
}
#[derive(Debug, Clone, Default)]
pub struct OpSetImport {
    pub domain: String,
    pub version: i64,
}

#[derive(Debug, Clone, Default)]
pub struct ModelProto {
    pub ir_version: i64,
    pub opset_import: Vec<OpSetImport>,
    pub graph: Option<GraphProto>,
    pub producer_name: String, // Added for tag 2
    pub producer_version: String, // Added for tag 3
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    // pub processed_nodes: HashSet<String>,
    pub last_processed_index: usize,
    // pub tensor_map_keys: Vec<String>, // Store only keys for serialization
    pub incremental_tensors: HashMap<String, String>, // Maps tensor name to file path
    // pub node_outputs: HashMap<String, Vec<String>>, // Maps node name to its output tensor names
    pub tensor_to_node: HashMap<String, usize>, // Maps tensor name to the node index that produced it
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Dimensions {
    Fixed(usize),
    Symbolic(String),
}

#[derive(Clone, Serialize, Deserialize)]
pub enum OrtValue {
    Tensor {
        
        shape: Vec<Dimensions>,
        dtype: DataType,
        #[serde(with = "serde_arc_vec")]
        data: Arc<Vec<u8>>,
    },
    Sequence(Vec<OrtValue>),
    Map(IndexMap<MapKey, OrtValue>),
    Opaque(Vec<u8>),
}


impl DataType {
    fn try_from(value: i32) -> Result<Self, OrtError> {
        match value {
            1 => Ok(DataType::Float),
            6 => Ok(DataType::Int32),
            7 => Ok(DataType::Int64),
            8 => Ok(DataType::String),
            9 => Ok(DataType::Boolean),
            other => Err(OrtError::UnknownDataType(other)),
        }
    }
}

// Implement Message trait for OpSetImport
impl Message for OpSetImport {
    fn encode_raw<B: prost::bytes::BufMut>(&self, _buf: &mut B) {
        unimplemented!("Encoding not needed")
    }

    fn merge_field<B: prost::bytes::Buf>(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut B,
        ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
        match tag {
            1 => {
                //print domain
                prost::encoding::string::merge(wire_type, &mut self.domain, buf, ctx)?
            },
            2 => {
                //print version
                prost::encoding::int64::merge(wire_type, &mut self.version, buf, ctx)?
            },
            _ => {
                println!("skipping unknown tag from opsetimport");
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?
            },
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        unimplemented!("Encoded length not needed")
    }

    fn clear(&mut self) {
        *self = OpSetImport::default();
    }
}

// Implement Message trait for TensorProto
impl Message for TensorProto {
    fn encode_raw<B: prost::bytes::BufMut>(&self, buf: &mut B) {
        unimplemented!("Encoding not needed")
    }

    fn merge_field<B: prost::bytes::Buf>(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut B,
        ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
        match tag {
            1 => {
                // let mut dim = 0i64;
                prost::encoding::int64::merge_repeated(wire_type, &mut self.dims, buf, ctx)?;
                // self.dims.push(dim);
                //nowcommented  println!("TensorProto: Added dimension: {}", dim);
            }
            5 => {
                // let mut dim = 0i64;
                prost::encoding::int32::merge_repeated(wire_type, &mut self.int32_data, buf, ctx)?;
                // self.dims.push(dim);
                //nowcommented  println!("TensorProto: Added dimension: {}", dim);
            }
            2 => {
                prost::encoding::int32::merge(wire_type, &mut self.data_type, buf, ctx)?;
                //nowcommented  println!("TensorProto: Set data_type: {}", self.data_type);
            }
            8 => {
                prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?;
                //nowcommented  println!("TensorProto: Set name: {}", self.name);
            }
            9 => {
                prost::encoding::bytes::merge(wire_type, &mut self.raw_data, buf, ctx)?;
                //  println!("TensorProto: Added raw_data (length: {})", self.raw_data.len());//nowcommented 
            }
            6 => {
                // let mut string_data = Vec::new();
                prost::encoding::bytes::merge_repeated(wire_type, &mut self.string_data, buf, ctx)?;
                // self.string_data.push(string_data.clone());
                //nowcommented  println!("TensorProto: Added string_data: {:?}", String::from_utf8_lossy(&string_data));
            }
            4 => {
                // let mut float_data = 0f32;
                prost::encoding::float::merge_repeated(wire_type, &mut self.float_data, buf, ctx)?;
                // self.float_data.push(float_data);
                //nowcommented  println!("TensorProto: Added float_data: {}", float_data);
            }
           7 => {
                // let mut int64_data = 0i64;
                prost::encoding::int64::merge_repeated(wire_type, &mut self.int64_data, buf, ctx)?;
                // self.int64_data.push(int64_data);
                //nowcommented  println!("TensorProto: Added int64_data: {}", int64_data);
            }
            _ => {
                 println!("TensorProto: Skipping unknown tag: {}", tag);//nowcommented 
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        unimplemented!("Encoding not needed")
    }

    fn clear(&mut self) {
        *self = TensorProto::default();
    }
}

// Implement Message trait for ValueInfoProto
impl Message for ValueInfoProto {
    fn encode_raw<B: prost::bytes::BufMut>(&self, _buf: &mut B) {
        unimplemented!("Encoding not needed")
    }

    fn merge_field<B: prost::bytes::Buf>(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut B,
        ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
        match tag {
            1 => {
                prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?;
                //nowcommented  println!("ValueInfoProto: Set name: {}", self.name);
            }
            2 => {
                let mut type_proto = TypeProto::default();
                prost::encoding::message::merge(wire_type, &mut type_proto, buf, ctx)?;
                //nowcommented  println!("ValueInfoProto: Added type");
                self.type_proto = Some(type_proto);
            }
            _ => {
                  println!("ValueInfoProto: Skipping unknown tag: {}", tag);//nowcommented
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        unimplemented!("Encoded length not needed")
    }

    fn clear(&mut self) {
        *self = ValueInfoProto::default();
    }
}

impl Message for ModelProto {
    fn encode_raw<B: prost::bytes::BufMut>(&self, _buf: &mut B) {
        unimplemented!("Encoding not needed")
    }

    fn merge_field<B: prost::bytes::Buf>(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut B,
        ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
        match tag {
            1 => {
                let mut ir_version = 0i64;
                prost::encoding::int64::merge(wire_type, &mut ir_version, buf, ctx)?;
                self.ir_version = ir_version;
                //nowcommented  println!("ModelProto: Parsed ir_version: {}", ir_version);
            }
            2 => {
                prost::encoding::string::merge(wire_type, &mut self.producer_name, buf, ctx)?;
                //nowcommented  println!("ModelProto: Set producer_name: {}", self.producer_name);
            }
            3 => {
                prost::encoding::string::merge(wire_type, &mut self.producer_version, buf, ctx)?;
                //nowcommented  println!("ModelProto: Set producer_version: {}", self.producer_version);
            }
            7 => {
                let mut graph = GraphProto::default();
                prost::encoding::message::merge(wire_type, &mut graph, buf, ctx)?;
                self.graph = Some(graph);
                //nowcommented  println!("ModelProto: Parsed graph");
            }
            8 => {
                // let mut opset = OpSetImport::default();
                prost::encoding::message::merge_repeated(wire_type, &mut self.opset_import, buf, ctx)?;
                // self.opset_import.push(opset);
                //nowcommented  println!("ModelProto: Added opset_import (domain: {}, version: {})", self.opset_import.last().unwrap().domain, self.opset_import.last().unwrap().version);
            }
            _ => {
                  println!("ModelProto: Skipping unknown tag: {}", tag);//nowcommented
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        unimplemented!("Encoded length not needed")
    }

    fn clear(&mut self) {
        *self = ModelProto::default();
    }
}

impl Message for GraphProto {
    fn encode_raw<B: prost::bytes::BufMut>(&self, _buf: &mut B) {
        unimplemented!("Encoding not needed")
    }

    fn merge_field<B: prost::bytes::Buf>(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut B,
        ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
        match tag {
            1 => {
                // let node_index = NODE_COUNTER.fetch_add(1, Ordering::SeqCst);
                // let mut node = NodeProto::default();
                prost::encoding::message::merge_repeated(wire_type, &mut self.node, buf, ctx)?;
                // println!(
                //     "GraphProto: Parsed Node {} (OpType: {})",
                //     node_index,
                //     node.op_type
                // );
                // println!("  Inputs:");
                // for input in &node.input {
                //     println!("    - {}", input);
                // }
                // println!("  Outputs:");
                // for output in &node.output {
                //     println!("    - {}", output);
                // }
                // self.node.push(node);
            }
            2 => {
                prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?;
                //nowcommented  println!("GraphProto: Set name: {}", self.name);
            }
            10 => {
                let mut bytes = Vec::new();
                prost::encoding::bytes::merge(wire_type, &mut bytes, buf, ctx)?;
                // Attempt to convert bytes to UTF-8 string, fallback to empty string if invalid
                self.doc_string = String::from_utf8_lossy(&bytes).into_owned();
                // println!("GraphProto: Set doc_string: {:?}", bytes);
                // println!("GraphProto: Set doc_string: {}", self.doc_string);
                //nowcommented  println!("GraphProto: Set doc_string: {}", self.doc_string);
            }
            5 => {
                // let mut init = TensorProto::default();
                prost::encoding::message::merge_repeated(wire_type, &mut self.initializer, buf, ctx)?;
                //   println!("GraphProto: Added initializer (name: {})", init.name); //nowcommented
                // self.initializer.push(init);
            }
            12 => {
                // let mut output = ValueInfoProto::default();
                prost::encoding::message::merge_repeated(wire_type, &mut self.output, buf, ctx)?;
                //nowcommented  println!("GraphProto: Added output (name: {})", output.name);
                // self.output.push(output);
            }
            11 => {
                // let mut input = ValueInfoProto::default();
                prost::encoding::message::merge_repeated(wire_type, &mut self.input, buf, ctx)?;
                //nowcommented  println!("GraphProto: Added input (name: {})", input.name);
                // self.input.push(input);
            }
            13 => {
                // let mut value_info = ValueInfoProto::default();
                prost::encoding::message::merge_repeated(wire_type, &mut self.value_info, buf, ctx)?;
                //nowcommented  println!("GraphProto: Added value_info (name: {})", value_info.name);
                // self.value_info.push(value_info);
            }
            _ => {
                  println!("GraphProto: Skipping unknown tag: {}", tag);//nowcommented
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        unimplemented!("Encoded length not needed")
    }

    fn clear(&mut self) {
        *self = GraphProto::default();
        // NODE_COUNTER.store(0, Ordering::SeqCst);
    }
}

impl Message for NodeProto {
    fn encode_raw<B: prost::bytes::BufMut>(&self, _buf: &mut B) {
        unimplemented!("Encoding not needed")
    }

    fn merge_field<B: prost::bytes::Buf>(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut B,
        ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
        match tag {
            1 => {
                // let mut input = String::new();
                prost::encoding::string::merge_repeated(wire_type, &mut self.input, buf, ctx)?;
                // self.input.push(input.clone());
                //nowcommented  println!("NodeProto: Added input: {}", input);
            }
            2 => {
                // let mut output = String::new();
                prost::encoding::string::merge_repeated(wire_type, &mut self.output, buf, ctx)?;
                // self.output.push(output.clone());
                //nowcommented  println!("NodeProto: Added output: {}", output);
            }
            3 => {
                let mut bytes = Vec::new();
                prost::encoding::bytes::merge(wire_type, &mut bytes, buf, ctx)?;
                // Attempt to convert bytes to UTF-8 string, fallback to empty string if invalid
                self.name = String::from_utf8_lossy(&bytes).into_owned();
                // println!("GraphProto: Set doc_string: {:?}", bytes);
                // prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?;
                // println!("NodeProto: Set name: {}", self.name);
            }
            4 => {
                prost::encoding::string::merge(wire_type, &mut self.op_type, buf, ctx)?;
                //nowcommented  println!("NodeProto: Set op_type: {}", self.op_type);
            }
            7 => {
                let mut bytes = Vec::new();
                prost::encoding::bytes::merge(wire_type, &mut bytes, buf, ctx)?;
                // Attempt to convert bytes to UTF-8 string, fallback to empty string if invalid
                self.domain = String::from_utf8_lossy(&bytes).into_owned();
                // println!("GraphProto: Set doc_string: {:?}", bytes);
                // prost::encoding::string::merge(wire_type, &mut self.domain, buf, ctx)?;
                // println!("NodeProto: Set domain: {}", self.domain);
            }
            5 => {
                // let mut attr = AttributeProto::default();
                prost::encoding::message::merge_repeated(wire_type, &mut self.attributes, buf, ctx)?;
                //nowcommented  println!("NodeProto: Added attribute (name: {})", attr.name);
                // self.attributes.push(attr);
            }
            //  8 => {
            //     // Handle subgraph attributes (e.g., then_branch, else_branch, body)
            //     let mut subgraph = GraphProto::default();
            //     prost::encoding::message::merge(wire_type, &mut subgraph, buf, ctx)?;
            //     //nowcommented  println!("NodeProto: Added subgraph for attribute");
            //     self.subgraphs.insert(format!("subgraph_{}", self.subgraphs.len()), subgraph);
            // }
            _ => {
                 println!("NodeProto: Skipping unknown tag: {}", tag); //nowcommented
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        unimplemented!("Encoded length not needed")
    }

    fn clear(&mut self) {
        *self = NodeProto::default();
    }
}

impl Message for AttributeProto {
    fn encode_raw<B: prost::bytes::BufMut>(&self, _buf: &mut B) {
        unimplemented!("Encoding not needed")
    }

    fn merge_field<B: prost::bytes::Buf>(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut B,
        ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
        match tag {
            1 => {
                prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?;
                // println!("AttributeProto: Set name: {}", self.name);
            }
            3 => {
                prost::encoding::int64::merge(wire_type, &mut self.i, buf, ctx)?;
                //nowcommented  println!("AttributeProto: Set int: {}", self.i);
            }
            2 => {
                prost::encoding::float::merge(wire_type, &mut self.f, buf, ctx)?;
                //nowcommented  println!("AttributeProto: Set float: {}", self.f);
            }
            4 => {
                prost::encoding::bytes::merge(wire_type, &mut self.s, buf, ctx)?;
                //nowcommented  println!("AttributeProto: Set bytes: {:?}", String::from_utf8_lossy(&self.s));
            } 
            9 => {
                prost::encoding::bytes::merge_repeated(wire_type, &mut self.strings, buf, ctx)?;
                //nowcommented  println!("AttributeProto: Set bytes: {:?}", String::from_utf8_lossy(&self.s));
            }
            5 => {
                let mut tensor = TensorProto::default();
                prost::encoding::message::merge(wire_type, &mut tensor, buf, ctx)?;
                //nowcommented  println!("AttributeProto: Added tensor (name: {})", tensor.name);
                self.t = Some(tensor);
            }
            8 => {
                // let mut i = 0i64;
                prost::encoding::int64::merge_repeated(wire_type, &mut self.ints, buf, ctx)?;
                // self.ints.push(i);
                //nowcommented  println!("AttributeProto: Added int to ints: {}", i);
            }
            7 => {
                // let mut f = 0.0f32;
                prost::encoding::float::merge_repeated(wire_type, &mut self.floats, buf, ctx)?;
                // self.floats.push(f);
                //nowcommented  println!("AttributeProto: Added float to floats: {}", f);
            }
             6 => {
                let mut subgraph = GraphProto::default();
                prost::encoding::message::merge(wire_type, &mut subgraph, buf, ctx)?;
                //nowcommented  println!("AttributeProto: Added subgraph");
                self.g = Some(subgraph);
            }
            _ => {
                  println!("AttributeProto: Skipping unknown tag: {}", tag);//nowcommented
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        unimplemented!("Encoded length not needed")
    }

    fn clear(&mut self) {
        *self = AttributeProto::default();
    }
}

impl Message for TypeProto {
    fn encode_raw<B: prost::bytes::BufMut>(&self, _buf: &mut B) {
        unimplemented!("Encoding not needed")
    }

    fn merge_field<B: prost::bytes::Buf>(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut B,
        ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
        //nowcommented  println!("Parsing TypeProto (tag: {})", tag);
        match tag {
            1 => {
                let mut tensor_type = TensorTypeProto::default();
                prost::encoding::message::merge(wire_type, &mut tensor_type, buf, ctx)?;
                //nowcommented  println!("TypeProto: Added tensor_type");
                self.tensor_type = Some(tensor_type);
            }
            _ => {
                  println!("TypeProto: Skipping unknown tag: {}", tag);//nowcommented
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        unimplemented!("Encoded length not needed")
    }

    fn clear(&mut self) {
        *self = TypeProto::default();
    }
}

impl Message for TensorTypeProto {
    fn encode_raw<B: prost::bytes::BufMut>(&self, _buf: &mut B) {
        unimplemented!("Encoding not needed")
    }

    fn merge_field<B: prost::bytes::Buf>(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut B,
        ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
        //nowcommented  println!("Parsing TensorTypeProto (tag: {})", tag);
        match tag {
            1 => {
                prost::encoding::int32::merge(wire_type, &mut self.elem_type, buf, ctx)?;
                //nowcommented  println!("TensorTypeProto: Set elem_type: {}", self.elem_type);
            }
            2 => {
                let mut shape = TensorShapeProto::default();
                prost::encoding::message::merge(wire_type, &mut shape, buf, ctx)?;
                //nowcommented  println!("TensorTypeProto: Added shape");
                self.shape = Some(shape);
            }
            _ => {
                  println!("TensorTypeProto: Skipping unknown tag: {}", tag);//nowcommented
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        unimplemented!("Encoded length not needed")
    }

    fn clear(&mut self) {
        *self = TensorTypeProto::default();
    }
}

impl Message for TensorShapeProto {
    fn encode_raw<B: prost::bytes::BufMut>(&self, _buf: &mut B) {
        unimplemented!("Encoding not needed")
    }

    fn merge_field<B: prost::bytes::Buf>(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut B,
        ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
        //nowcommented  println!("Parsing TensorShapeProto (tag: {})", tag);
        match tag {
            1 => {
                // let mut dim = TensorDimension::default();
                prost::encoding::message::merge_repeated(wire_type, &mut self.dim, buf, ctx)?;
                //nowcommented  println!("TensorShapeProto: Added dimension (dim_value: {})", dim.dim_value);
                // self.dim.push(dim);
            }
            _ => {
                  println!("TensorShapeProto: Skipping unknown tag: {}", tag);//nowcommented
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        unimplemented!("Encoded length not needed")
    }

    fn clear(&mut self) {
        *self = TensorShapeProto::default();
    }
}

impl Message for TensorDimension {
    fn encode_raw<B: prost::bytes::BufMut>(&self, _buf: &mut B) {
        unimplemented!("Encoding not needed")
    }

    fn merge_field<B: prost::bytes::Buf>(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut B,
        ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
        //nowcommented  println!("Parsing Dimension (tag: {})", tag);
        match tag {
            1 => {
                prost::encoding::int64::merge(wire_type, &mut self.dim_value, buf, ctx)?;
                //nowcommented  println!("Dimension: Set dim_value: {}", self.dim_value);
            }
            _ => {
                  println!("Dimension: Skipping unknown tag: {}", tag);//nowcommented
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        unimplemented!("Encoded length not needed")
    }

    fn clear(&mut self) {
        *self = TensorDimension::default();
    }
}
#[derive(Error, Debug)]
pub enum OrtError {
    #[error("Protobuf error: {0}")]
    Prost(#[from] prost::DecodeError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Type mismatch: {0}")]
    TypeMismatch(String),
    #[error("Index error: {0}")]
    IndexError(&'static str),
    #[error("Invalid model")]
    InvalidModel,
    #[error("Invalid input count")]
    InvalidInputCount,
    #[error("invalid input shape")]
    InvalidInputShape(String),
    #[error("Data cannot be accessed")]
    DataAccessError(String),
    #[error("Missing1 input: {0}")]
    MissingInput(String),
    #[error("Unsupported op: {0}")]
    UnsupportedOp(String),
    #[error("Missing output: {0}")]
    MissingOutput(String),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unknown data type: {0}")]
    UnknownDataType(i32),
    #[error("Invalid tensor data: {0}")]
    InvalidTensorData(String),
}

pub type OrtResult<T> = Result<T, OrtError>;



impl ExecutionState {
    pub fn new() -> Self {
        Self {
            // processed_nodes: HashSet::new(),
            last_processed_index: 0,
            // tensor_map_keys: Vec::new(),
            incremental_tensors: HashMap::new(),
            // node_outputs: HashMap::new(),
            tensor_to_node: HashMap::new(),
        }
    }
    
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let state: ExecutionState = serde_json::from_str(&json)?;
        Ok(state)
    }
    
    // Save only new tensors incrementally with node output tracking
    pub fn save_new_tensors(&mut self, new_tensors: &HashMap<String, OrtValue>, node_index: usize, node_name: &str, output_names: &[String]) -> Result<(), Box<dyn std::error::Error>> {
        if new_tensors.is_empty() {
            return Ok(());
        }
        
        let tensor_file = format!("./tensor_store/tensors_node_{}.bin", node_index);
        println!("Saving {} new tensors to {}", new_tensors.len(), tensor_file);
        
        match bincode::serialize(new_tensors) {
            Ok(serialized) => {
                std::fs::write(&tensor_file, &serialized)?;
                
                // Update the incremental tensors map and tracking
                for tensor_name in new_tensors.keys() {
                    self.incremental_tensors.insert(tensor_name.clone(), tensor_file.clone());
                    self.tensor_to_node.insert(tensor_name.clone(), node_index);
                }
                
                // Track node outputs
                // self.node_outputs.insert(node_name.to_string(), output_names.to_vec());
                
                println!("✓ Saved {} tensors ({} bytes)", new_tensors.len(), serialized.len());
                Ok(())
            }
            Err(e) => {
                println!("✗ Tensor serialization failed: {}", e);
                Err(Box::new(e))
            }
        }
    }
    
    // Load specific tensors on demand
    pub fn load_tensors_for_inputs(&self, input_names: &[String]) -> Result<HashMap<String, OrtValue>, Box<dyn std::error::Error>> {
        let mut loaded_tensors = HashMap::new();
        let mut files_to_load = HashSet::new();
        
        // Determine which files need to be loaded
        for input_name in input_names {
            if let Some(file_path) = self.incremental_tensors.get(input_name) {
                files_to_load.insert(file_path.clone());
            }
        }
        
        // Load each required file
        for file_path in files_to_load {
            match std::fs::read(&file_path) {
                Ok(data) => {
                    match bincode::deserialize::<HashMap<String, OrtValue>>(&data) {
                        Ok(file_tensors) => {
                            // Only extract the tensors we need from this file
                            for input_name in input_names {
                                if let Some(tensor) = file_tensors.get(input_name) {
                                    loaded_tensors.insert(input_name.clone(), tensor.clone());
                                }
                            }
                            println!("✓ Loaded required tensors from {}", file_path);
                        }
                        Err(e) => {
                            println!("✗ Failed to deserialize {}: {}", file_path, e);
                        }
                    }
                }
                Err(e) => {
                    println!("✗ Failed to read {}: {}", file_path, e);
                }
            }
        }
        
        println!("Loaded {} tensors on demand", loaded_tensors.len());
        Ok(loaded_tensors)
    }
    
    // Get information about which node produced a tensor
    pub fn get_tensor_producer(&self, tensor_name: &str) -> Option<usize> {
        self.tensor_to_node.get(tensor_name).copied()
    }
    
    // Get all outputs produced by a specific node
    // pub fn get_node_outputs(&self, node_name: &str) -> Option<&Vec<String>> {
    //     // self.node_outputs.get(node_name)
    // }
    
    // Get statistics about the execution state
    pub fn get_stats(&self) -> (usize) {
        (
            // self.processed_nodes.len(),
            self.incremental_tensors.len()
            // self.node_outputs.len()
        )
    }
    
    // Load all tensors from incremental files
    pub fn load_incremental_tensors(&self) -> Result<HashMap<String, OrtValue>, Box<dyn std::error::Error>> {
        let mut tensor_map = HashMap::new();
        let mut loaded_files = HashSet::new();
        
        for (tensor_name, file_path) in &self.incremental_tensors {
            if !loaded_files.contains(file_path) {
                match std::fs::read(file_path) {
                    Ok(data) => {
                        match bincode::deserialize::<HashMap<String, OrtValue>>(&data) {
                            Ok(file_tensors) => {
                                tensor_map.extend(file_tensors);
                                loaded_files.insert(file_path.clone());
                                println!("✓ Loaded tensors from {}", file_path);
                            }
                            Err(e) => {
                                println!("✗ Failed to deserialize {}: {}", file_path, e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("✗ Failed to read {}: {}", file_path, e);
                    }
                }
            }
        }
        
        println!("Loaded {} tensors from {} files", tensor_map.len(), loaded_files.len());
        Ok(tensor_map)
    }
    
    // Legacy methods for backward compatibility
    pub fn save_tensor_map(&self, tensor_map: &HashMap<String, OrtValue>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Only save if tensor map is small (for initial state)
        if tensor_map.len() < 100 {
            println!("Saving small tensor map with {} entries to {}", tensor_map.len(), path);
            match bincode::serialize(tensor_map) {
                Ok(serialized) => {
                    std::fs::write(path, &serialized)?;
                    println!("✓ Saved tensor map ({} bytes)", serialized.len());
                    Ok(())
                }
                Err(e) => {
                    println!("✗ Tensor map serialization failed: {}", e);
                    Err(Box::new(e))
                }
            }
        } else {
            println!("Skipping large tensor map save ({} entries) - using incremental storage", tensor_map.len());
            Ok(())
        }
    }
    
    pub fn load_tensor_map(&self, path: &str) -> Result<HashMap<String, OrtValue>, Box<dyn std::error::Error>> {
        match std::fs::read(path) {
            Ok(data) => {
                let tensor_map: HashMap<String, OrtValue> = bincode::deserialize(&data)?;
                Ok(tensor_map)
            }
            Err(_) => {
                // If legacy file doesn't exist, try incremental loading
                self.load_incremental_tensors()
            }
        }
    }
}

impl fmt::Debug for OrtValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrtValue::Tensor { shape, dtype, data } =>{ 
            //     let concrete_shape: Vec<usize> = shape.iter().map(|d| match d {
            //         Dimensions::Fixed(n) => *n,
            //         Dimensions::Symbolic(_) => unreachable!(), // Handled above
            //     }).collect();
            // let result = match dtype {
            //     DataType::Float => {
            //         let float_data: Vec<f32> = data
            //             .chunks(4)
            //             .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            //             .collect();
            //         ArrayD::from_shape_vec(IxDyn(&concrete_shape), float_data)
            //             .map(|arr| ArrayDResult::Float(arr))
            //             .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for float tensor".into()))
            //     }
            //     DataType::Int64 => {
            //         let int64_data: Vec<i64> = data
            //             .chunks(8)
            //             .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            //             .collect();
            //         ArrayD::from_shape_vec(IxDyn(&concrete_shape), int64_data)
            //             .map(|arr| ArrayDResult::Int64(arr))
            //             .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for int64 tensor".into()))
            //     }
            //     DataType::Int32 => {
            //         let int32_data: Vec<i32> = data
            //             .chunks(4)
            //             .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            //             .collect();
            //         ArrayD::from_shape_vec(IxDyn(&concrete_shape), int32_data)
            //             .map(|arr| ArrayDResult::Int32(arr))
            //             .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for int32 tensor".into()))
            //     }
            //     DataType::Boolean=>{
            //     let bool_data: Vec<bool> = data
            //                             .iter()
            //                             .map(|&b| b != 0)
            //                             .collect();
            //         ArrayD::from_shape_vec(IxDyn(&concrete_shape), bool_data)
            //             .map(|arr| ArrayDResult::Boolean(arr))
            //             .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for boolean tensor".into()))
                    
            //     }
                
            //     _ => Err(OrtError::TypeMismatch("Unsupported tensor type, expected Float, Int64, or Int32".to_string())),
            // };
            // println!("--->>>>=========={:?}",result.unwrap());
            f
                .debug_struct("Tensor")
                // .field("data", data)
                // .field("anything", &result.unwrap())
                .field("shape", shape)
                .field("dtype", dtype)
                .finish()},
            OrtValue::Sequence(seq) => f.debug_tuple("Sequence").field(seq).finish(),
            OrtValue::Map(map) => f.debug_tuple("Map").field(map).finish(),
            OrtValue::Opaque(o) => f.debug_tuple("Opaque").field(o).finish(),
        }
    }
}
impl fmt::Display for OrtValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrtValue::Tensor { shape, dtype, data } => {
                write!(f, "Tensor {{ shape: {:?}, dtype: {:?}", shape, dtype).unwrap();
                // Summarize data based on dtype
                let preview_len = 5; // Show up to 5 elements
                match dtype {
                    DataType::Float => {
                                        let float_data: Vec<f32> = data
                                            .chunks_exact(4)
                                            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                                            .collect();
                                        let preview = float_data.iter().take(preview_len).collect::<Vec<_>>();
                                        write!(f, ", data: [{} elements, first {:?}]", float_data.len(), preview)?;
                                    }
                    DataType::Int64 => {
                                        let int_data: Vec<i64> = data
                                            .chunks_exact(8)
                                            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                                            .collect();
                                        let preview = int_data.iter().take(preview_len).collect::<Vec<_>>();
                                        write!(f, ", data: [{} elements, first {:?}]", int_data.len(), preview)?;
                                    }
                    DataType::Int32 => {
                                        let int_data: Vec<i32> = data
                                            .chunks_exact(4)
                                            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                                            .collect();
                                        let preview = int_data.iter().take(preview_len).collect::<Vec<_>>();
                                        write!(f, ", data: [{} elements, first {:?}]", int_data.len(), preview)?;
                                    }
                    DataType::String => todo!(),
                    DataType::Boolean => {
                                    let bool_data: Vec<bool> = data
                                        .iter()
                                        .map(|&byte| byte != 0)
                                        .collect();
                                    let preview = bool_data.iter().take(preview_len).collect::<Vec<_>>();
                                    write!(f, ", data: [{} elements, first {:?}]", bool_data.len(), preview)?;
                        
                    },
                }
                write!(f, " }}")
            }
            OrtValue::Sequence(seq) => {
                write!(f, "Sequence(len={}) [", seq.len())?;
                for (i, item) in seq.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            OrtValue::Map(map) => {
                write!(f, "Map(len={}) {{", map.len())?;
                for (i, (key, value)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}: {}", key, value)?;
                }
                write!(f, "}}")
            }
            OrtValue::Opaque(data) => {
                let preview = data.iter().take(5).collect::<Vec<_>>();
                write!(f, "Opaque(len={}) [first {:?}]", data.len(), preview)
            }
        }
    }
}


impl OrtValue {
    pub fn dtype(&self)->&DataType{
        match self {
            OrtValue::Tensor { dtype, .. } => dtype,
            _ => panic!("Shape only available for Tensor variant"),
        }
    }
    pub fn shape(&self) -> &Vec<Dimensions> {
        match self {
            OrtValue::Tensor { shape, .. } => shape,
            _ => panic!("Shape only available for Tensor variant"),
        }
    }
    // pub fn print(&self){
    //     println!("{:?}",self.Te)
    // }
    
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MapKey {
    String(String),
    Int64(i64),
}
#[derive(Debug, Clone, Default)]
struct ShapeInference {
    shapes: HashMap<String, Vec<Dimensions>>,
    value_info: HashMap<String, ValueInfoProto>,
    graph: GraphProto, // Added

}

impl ShapeInference {
    fn new(graph: &GraphProto) -> Self {
        let mut value_info = HashMap::new();
        for vi in graph.input.iter().chain(graph.output.iter()).chain(graph.value_info.iter()) {
            value_info.insert(vi.name.clone(), vi.clone());
        }
        ShapeInference {
            shapes: HashMap::new(),
            value_info,
            graph: graph.clone(), // Store a clone of the graph
        }
    }

    fn infer_shapes(&mut self, inputs: &HashMap<String, OrtValue>) -> OrtResult<()> {
        // Initialize shapes from inputs
        for (name, value) in inputs {
            self.shapes.insert(name.clone(), value.shape().clone());
        }

        // Iterate through nodes and infer output shapes
        for node in &self.graph.node {
            let input_shapes: Vec<Vec<Dimensions>> = node.input.iter()
                .map(|name| self.shapes.get(name)
                    .cloned()
                    .ok_or_else(|| OrtError::MissingInput("from shape".to_string()+&name.clone())))
                .collect::<OrtResult<Vec<_>>>()?;

            let output_shapes = self.infer_node_shapes(node, &input_shapes)?;
            for (output, shape) in node.output.iter().zip(output_shapes.iter()) {
                self.shapes.insert(output.clone(), shape.clone());
            }
        }
        Ok(())
    }

   fn infer_node_shapes(&self, node: &NodeProto, input_shapes: &[Vec<Dimensions>]) -> OrtResult<Vec<Vec<Dimensions>>> {
    match node.op_type.as_str() {
        "Identity" => {
            // Identity op returns the same shape as input
            if input_shapes.is_empty() {
                return Err(OrtError::MissingInput("Identity requires an input".to_string()));
            }
            Ok(vec![input_shapes[0].clone()])
        },
        "Shape" => {
            // Shape op returns the shape of the input tensor as a 1D tensor
            // Output shape: [rank of input]
            let rank = input_shapes[0].len();
            Ok(vec![vec![Dimensions::Fixed(rank)]])
        },
        "Add" | "Sub" | "Mul" | "Div" => {
            // Element-wise operations require matching shapes
            if input_shapes[0] != input_shapes[1] {
                return Err(OrtError::TypeMismatch(format!(
                     "Input shapes must match for {}: got {:?} and {:?}", 
                    node.op_type, input_shapes[0], input_shapes[1]
                )));
            }
            Ok(vec![input_shapes[0].clone()])
        },
        // Sequence operators
        "SequenceAt" => {
            // Output shape depends on the sequence element shape, which we don't know at inference time
            // Return a placeholder shape
            Ok(vec![vec![Dimensions::Symbolic("unknown".to_string())]])
        },
        "SequenceConstruct" => {
            // Output is a sequence, not a tensor, so no shape to infer
            Ok(vec![vec![]])
        },
        "SequenceEmpty" => {
            // Output is an empty sequence, not a tensor, so no shape to infer
            Ok(vec![vec![]])
        },
        "SequenceErase" => {
            // Output is a sequence, not a tensor, so no shape to infer
            Ok(vec![vec![]])
        },
        "SequenceInsert" => {
            // Output is a sequence, not a tensor, so no shape to infer
            Ok(vec![vec![]])
        },
        "SequenceLength" => {
            // Output is a scalar tensor
            Ok(vec![vec![Dimensions::Fixed(1)]])
        },
        // Map operators
        "MapFromTensor" => {
            // Output is a map, not a tensor, so no shape to infer
            Ok(vec![vec![]])
        },
        "MapToTensor" => {
            // Output is a sequence of two tensors (keys and values)
            // We don't know their shapes at inference time
            Ok(vec![vec![]])
        },
        "MapGet" => {
            // Output shape depends on the map value shape, which we don't know at inference time
            // Return a placeholder shape
            Ok(vec![vec![Dimensions::Symbolic("unknown".to_string())]])
        },
        "MapHasKey" => {
            // Output is a scalar tensor
            Ok(vec![vec![Dimensions::Fixed(1)]])
        },
        "MapKeys" => {
            // Output shape depends on the number of keys in the map, which we don't know at inference time
            // Return a placeholder shape
            Ok(vec![vec![Dimensions::Symbolic("unknown".to_string())]])
        },
        "MatMul" => {
            // Matrix multiplication: [..., m, k] @ [..., k, n] -> [..., m, n]
            let shape1 = &input_shapes[0];
            let shape2 = &input_shapes[1];
            let ndim1 = shape1.len();
            let ndim2 = shape2.len();
            if ndim1 < 1 || ndim2 < 1 {
                return Err(OrtError::InvalidTensorData(format!(
                    "MatMul requires at least 1D tensors for {}", node.name
                )));
            }
            let m = shape1[ndim1 - 2].clone(); // Rows of first matrix
            let n = shape2[ndim2 - 1].clone(); // Columns of second matrix
            // Check compatibility: shape1[..., k] and shape2[..., k]
            if ndim1 > 1 && ndim2 > 1 && shape1[ndim1 - 1] != shape2[ndim2 - 2] {
                return Err(OrtError::TypeMismatch(format!( "MatMul inner dimensions mismatch: {:?} vs {:?}", 
                     shape1[ndim1 - 1], shape2[ndim2 - 2]
                )));
            }
            let mut out_shape = shape1[..ndim1 - 2].to_vec();
            out_shape.push(m);
            out_shape.push(n);
            Ok(vec![out_shape])
        }
        "Gather" => {
            // Gather: Select elements from data tensor using indices
            // Inputs: data [d0, d1, ..., dn], indices [i0, i1, ..., im]
            // Attribute: axis (default 0)
            // Output shape: [d0, ..., d(axis-1), i0, i1, ..., im, d(axis+1), ..., dn]
            let data_shape = &input_shapes[0];
            let indices_shape = &input_shapes[1];
            let axis = node.attributes.iter()
                .find(|attr| attr.name == "axis")
                .and_then(|attr| Some(attr.i))
                .unwrap_or(0) as usize;
            if axis >= data_shape.len() {
                return Err(OrtError::InvalidTensorData(format!(
                    "Gather axis {} out of bounds for data shape {:?}", axis, data_shape
                )));
            }
            let mut output_shape = Vec::new();
            // Copy dimensions before axis
            for i in 0..axis {
                output_shape.push(data_shape[i].clone());
            }
            // Append indices shape
            for dim in indices_shape {
                output_shape.push(dim.clone());
            }
            // Copy dimensions after axis
            for i in (axis + 1)..data_shape.len() {
                output_shape.push(data_shape[i].clone());
            }
            Ok(vec![output_shape])
        }
        "Softmax" => {
            // Softmax: Normalizes along the specified axis (default: last dimension)
            // Output shape is the same as input shape
            let axis = node.attributes.iter()
                .find(|attr| attr.name == "axis")
                .and_then(|attr| Some(attr.i))
                .unwrap_or(-1) as i64;
            let input_shape = &input_shapes[0];
            let axis = if axis >= 0 { axis as usize } else { input_shape.len() - (-axis as usize) };
            if axis >= input_shape.len() {
                return Err(OrtError::InvalidTensorData(format!(
                    "Softmax axis {} out of bounds for shape {:?}", axis, input_shape
                )));
            }
            Ok(vec![input_shape.clone()])
        }
        "LayerNormalization" => {
            // LayerNormalization: Normalizes over the last dimension
            // Inputs: X, scale, bias (scale and bias may have shape matching last dimension)
            // Output shape is the same as input X shape
            let input_shape = &input_shapes[0];
            // Optional: Validate scale and bias shapes if needed
            Ok(vec![input_shape.clone()])
        }
        "Relu" => {
            // Relu: Element-wise activation, preserves input shape
            Ok(vec![input_shapes[0].clone()])
        }
        "Transpose" => {
            // Transpose: Permutes dimensions according to perm attribute
            let input_shape = &input_shapes[0];
            let perm = node.attributes.iter()
                .find(|attr| attr.name == "perm")
                .and_then(|attr| Some(attr.ints.clone()))
                .ok_or_else(|| OrtError::InvalidTensorData(format!(
                    "Transpose node {} missing perm attribute", node.name
                )))?;
            if perm.len() != input_shape.len() {
                return Err(OrtError::InvalidTensorData(format!(
                    "Transpose perm length {} does not match input shape {:?}", 
                    perm.len(), input_shape
                )));
            }
            let mut output_shape = vec![Dimensions::Fixed(0); input_shape.len()];
            for (i, &p) in perm.iter().enumerate() {
                output_shape[i] = input_shape[p as usize].clone();
            }
            Ok(vec![output_shape])
        }
        "Reshape" => {
            // Reshape: Reshapes input tensor according to shape input
            // Inputs: data, shape (shape is a 1D tensor of integers)
            let data_shape = &input_shapes[0];
            let shape_tensor = &input_shapes[1];
            // For simplicity, assume shape input is provided at inference time
            // In a full implementation, you may need to evaluate the shape tensor
            // Here, we assume the shape tensor’s values are known or symbolic
            let mut output_shape = shape_tensor.clone();
            // Handle -1 (infer dimension) or 0 (copy from input)
            let mut product = 1;
            let mut inferred_dim = None;
            for (i, dim) in output_shape.iter_mut().enumerate() {
                if let Dimensions::Fixed(d) = dim {
                    if *d == 0 {
                        *dim = data_shape[i].clone();
                    } else if *d == usize::MAX {
                        if inferred_dim.is_some() {
                            return Err(OrtError::InvalidTensorData(
                                "Reshape allows only one -1 dimension".into()
                            ));
                        }
                        inferred_dim = Some(i);
                    }
                    if let Dimensions::Fixed(d) = dim {
                        product *= *d as i64;
                    }
                }
            }
            if let Some(inferred_idx) = inferred_dim {
                let input_product: i64 = data_shape.iter()
                    .filter_map(|d| if let Dimensions::Fixed(d) = d { Some(*d as i64) } else { None })
                    .product();
                if product == 0 {
                    return Err(OrtError::InvalidTensorData(
                        "Reshape product of fixed dimensions is zero".into()
                    ));
                }
                output_shape[inferred_idx] = Dimensions::Fixed((input_product / product) as usize);
            }
            Ok(vec![output_shape])
        }
        "Cast" => {
            // Cast: Changes data type but preserves shape
            Ok(vec![input_shapes[0].clone()])
        }
        "Concat" => {
            // Concat: Concatenates tensors along a specified axis
            let axis = node.attributes.iter()
                .find(|attr| attr.name == "axis")
                .and_then(|attr| Some(attr.i))
                .ok_or_else(|| OrtError::InvalidTensorData(format!(
                    "Concat node {} missing axis attribute", node.name
                )))? as usize;
            let mut output_shape = input_shapes[0].clone();
            let mut concat_dim = 0;
            for shape in input_shapes {
                if shape.len() != output_shape.len() {
                    // return Err(OrtError::TypeMismatch(&(""
                    return Err(OrtError::TypeMismatch(format!(
                        "Concat input shapes must have same rank: {:?}", input_shapes
                    )));
                }
                for (i, dim) in shape.iter().enumerate() {
                    if i == axis {
                        if let Dimensions::Fixed(d) = dim {
                            concat_dim += d;
                        } else {
                            // Handle symbolic dimensions by keeping symbolic
                            concat_dim = 0; // Reset to indicate symbolic
                            break;
                        }
                    } else if shape[i] != output_shape[i] {
                        // return Err(OrtError::TypeMismatch(&(""
                        return Err(OrtError::TypeMismatch(format!(
                            "Concat non-axis dimensions must match: {:?}", input_shapes
                        )));
                    }
                }
            }
            if concat_dim > 0 {
                output_shape[axis] = Dimensions::Fixed(concat_dim);
            }
            Ok(vec![output_shape])
        }
        // Add more operators as needed
        _ => Err(OrtError::UnsupportedOp(format!(
            "Shape inference not implemented for {} in node {}", 
            node.op_type, node.name
        ))),
    }
}
}



impl OrtEngine {
    pub fn new<P: AsRef<Path>>(path: P) -> OrtResult<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let model = ModelProto::decode(&*buffer)?;
        let shape_inference = ShapeInference::new(model.graph.as_ref().ok_or(OrtError::InvalidModel)?);

        let mut engine = Self {
            model,
            node_registry: HashMap::new(),
            vendor_ops: HashMap::new(),
            shape_inference,
        };
        engine.register_core_ops();
        Ok(engine)
    }

    fn register_core_ops(&mut self) {
        self.node_registry.insert("Add".into(), Self::op_add);
        self.node_registry.insert("Sub".into(), Self::op_sub);
        self.node_registry.insert("Mul".into(), Self::op_mul);
        self.node_registry.insert("Div".into(), Self::op_div);
        self.node_registry.insert("CumSum".into(), Self::op_cumsum);
        self.node_registry.insert("Range".into(), Self::op_range);
        self.node_registry.insert("Shape".into(), Self::op_shape);
        self.node_registry.insert("MatMul".into(), Self::op_matmul);
        self.node_registry.insert("Less".into(), Self::op_less);
        self.node_registry.insert("Squeeze".into(), Self::op_squeeze);
        // self.node_registry.insert("STFT".into(), Self::op_stft);
        self.node_registry.insert("Slice".into(), Self::op_slice);
        self.node_registry.insert("Exp".into(), Self::op_exp);
        self.node_registry.insert("NonZero".into(), Self::op_nonzero);
        self.node_registry.insert("Tanh".into(), Self::op_tanh);
        self.node_registry.insert("LeakyRelu".into(), Self::op_leaky_relu);
        self.node_registry.insert("Greater".into(), Self::op_greater);
        self.node_registry.insert("Sigmoid".into(), Self::op_sigmoid);
        self.node_registry.insert("ReduceMean".into(), Self::op_reduce_mean);
        self.node_registry.insert("Atan".into(), Self::op_atan);
        self.node_registry.insert("Pow".into(), Self::op_pow);
        self.node_registry.insert("Gather".into(), Self::op_gather);
        self.node_registry.insert("Softmax".into(), Self::op_softmax);
        self.node_registry.insert("Unsqueeze".into(), Self::op_unsqueeze);
        self.node_registry.insert("Round".into(), Self::op_round);
        self.node_registry.insert("And".into(), Self::op_and);
        // self.node_registry.insert("ConvTranspose".into(), Self::op_conv_transpose);
        self.node_registry.insert("Pad".into(), Self::op_pad);
        self.node_registry.insert("Reshape".into(), Self::op_reshape);
        self.node_registry.insert("ScatterND".into(), Self::op_scatter_nd);
        self.node_registry.insert("Where".into(), Self::op_where);
        self.node_registry.insert("Sin".into(), Self::op_sin);
        // self.node_registry.insert("LSTM".into(), Self::op_lstm);
        self.node_registry.insert("ReduceSum".into(), Self::op_reduce_sum);
        self.node_registry.insert("Clip".into(), Self::op_clip);
        self.node_registry.insert("Resize".into(), Self::op_resize);
        self.node_registry.insert("Floor".into(), Self::op_floor);
        self.node_registry.insert("Cos".into(), Self::op_cos);
        self.node_registry.insert("Concat".into(), Self::op_concat);
        self.node_registry.insert("Cast".into(), Self::op_cast);
        self.node_registry.insert("Transpose".into(), Self::op_transpose);
        self.node_registry.insert("Equal".into(), Self::op_equal);
        self.node_registry.insert("ConstantOfShape".into(), Self::op_constant_of_shape);
        self.node_registry.insert("GreaterOrEqual".into(), Self::op_greater_or_equal);
        self.node_registry.insert("Sqrt".into(), Self::op_sqrt);
        self.node_registry.insert("Expand".into(), Self::op_expand);
        // self.node_registry.insert("Conv".into(), Self::op_conv);
        self.node_registry.insert("LayerNormalization".into(), Self::op_layer_normalization);
        self.node_registry.insert("Gemm".into(), Self::op_gemm);
        
        // // BERT-specific operators
        // self.node_registry.insert("Erf".into(), Self::op_erf);
        // self.node_registry.insert("Gelu".into(), Self::op_gelu);
        // self.node_registry.insert("Split".into(), Self::op_split);
        // self.node_registry.insert("Dropout".into(), Self::op_dropout);
        // self.node_registry.insert("Einsum".into(), Self::op_einsum);
        // self.node_registry.insert("TopK".into(), Self::op_topk);
        // self.node_registry.insert("GatherElements".into(), Self::op_gather_elements);
        // self.node_registry.insert("GatherND".into(), Self::op_gather_nd);
        // self.node_registry.insert("ReduceMax".into(), Self::op_reduce_max);
        // self.node_registry.insert("Attention".into(), Self::op_attention);
        
        // // Kokoro-specific operators
        // self.node_registry.insert("Embedding".into(), Self::op_embedding);
        // self.node_registry.insert("ConstantOfShapeInt64".into(), Self::op_constant_of_shape_int64);
        // self.node_registry.insert("LayerNormalizationWithEpsilon".into(), Self::op_layer_normalization_with_epsilon);
        self.node_registry.insert("Expand".into(), Self::op_expand);
        // self.node_registry.insert("PositionEmbeddings".into(), Self::op_position_embeddings);
        // self.node_registry.insert("TokenTypeEmbeddings".into(), Self::op_token_type_embeddings);
        // self.node_registry.insert("BertAttention".into(), Self::op_bert_attention);
        // self.node_registry.insert("BertIntermediate".into(), Self::op_bert_intermediate);
        // self.node_registry.insert("BertOutput".into(), Self::op_bert_output);
        // self.node_registry.insert("BertPooler".into(), Self::op_bert_pooler);
        
        // // Sequence operators
        // self.node_registry.insert("SequenceAt".into(), Self::op_sequence_at);
        // self.node_registry.insert("SequenceConstruct".into(), Self::op_sequence_construct);
        // self.node_registry.insert("SequenceEmpty".into(), Self::op_sequence_empty);
        // self.node_registry.insert("SequenceErase".into(), Self::op_sequence_erase);
        // self.node_registry.insert("SequenceInsert".into(), Self::op_sequence_insert);
        // self.node_registry.insert("SequenceLength".into(), Self::op_sequence_length);
        
        // // Map operators
        // self.node_registry.insert("MapFromTensor".into(), Self::op_map_from_tensor);
        // self.node_registry.insert("MapToTensor".into(), Self::op_map_to_tensor);
        // self.node_registry.insert("MapGet".into(), Self::op_map_get);
        // self.node_registry.insert("MapHasKey".into(), Self::op_map_has_key);
        // self.node_registry.insert("MapKeys".into(), Self::op_map_keys);
        
        // Control flow operators (improved)
        
        
        // Other operators
        self.node_registry.insert("CumSum".into(), Self::op_cumsum);
        self.node_registry.insert("NonZero".into(), Self::op_nonzero);
        self.node_registry.insert("ScatterND".into(), Self::op_scatter_nd);
        // Use optimized versions for better performance
        self.node_registry.insert("Conv".into(), Self::op_conv_optimized);
        self.node_registry.insert("ConvTranspose".into(), Self::op_conv_transpose_optimized);
        self.node_registry.insert("LSTM".into(), Self::op_lstm);
        // self.node_registry.insert("Resize".into(), Self::op_resize_optimized);
        self.node_registry.insert("STFT".into(), Self::op_stft_optimized);
        
        // Keep original versions available with different names for fallback
        // self.node_registry.insert("Conv_Original".into(), Self::op_conv);
        // self.node_registry.insert("ConvTranspose_Original".into(), Self::op_conv_transpose);
        // self.node_registry.insert("LSTM_Original".into(), Self::op_lstm);
        // self.node_registry.insert("Resize_Original".into(), Self::op_resize);
        // self.node_registry.insert("STFT_Original".into(), Self::op_stft);
    }
    pub fn infer(&self, inputs: HashMap<String, OrtValue>) -> OrtResult<HashMap<String, OrtValue>> {
        self.infer_with_resumption(inputs, None)
    }
    
    pub fn resume_inference(&self, inputs: HashMap<String, OrtValue>) -> OrtResult<HashMap<String, OrtValue>> {
        self.infer_with_resumption(inputs, Some("execution_state2.json"))
    }
    
    pub fn resume_inference_from_file(&self, inputs: HashMap<String, OrtValue>, state_file: &str) -> OrtResult<HashMap<String, OrtValue>> {
        self.infer_with_resumption(inputs, Some(state_file))
    }
    
    pub fn clear_execution_state() -> Result<(), std::io::Error> {
        let _ = std::fs::remove_file("execution_state2.json");
        let _ = std::fs::remove_file("tensor_map2.bin");
        
        // // Clean up incremental tensor files
        // if let Ok(state) = ExecutionState::load_from_file("execution_state.json") {
        //     Self::cleanup_incremental_tensors(&state);
        // }
        
        Ok(())
    }
    
    fn cleanup_incremental_tensors(execution_state: &ExecutionState) {
        let mut cleaned_files = HashSet::new();
        for (_, file_path) in &execution_state.incremental_tensors {
            if !cleaned_files.contains(file_path) {
                if let Err(e) = std::fs::remove_file(file_path) {
                    eprintln!("Warning: Failed to clean up incremental tensor file {}: {}", file_path, e);
                } else {
                    println!("Cleaned up incremental tensor file: {}", file_path);
                }
                cleaned_files.insert(file_path.clone());
            }
        }
    }
    
    pub fn get_execution_progress(state_file: Option<&str>) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let state_file_path = state_file.unwrap_or("execution_state2.json");
        let state = ExecutionState::load_from_file(state_file_path)?;
        // We need to get total nodes from somewhere - this is a limitation
        // In practice, you'd pass the total or calculate it differently
        Ok((state.last_processed_index, 0)) // 0 as placeholder for total
    }
    
    pub fn get_total_nodes(&self) -> usize {
        self.model.graph.as_ref().map(|g| g.node.len()).unwrap_or(0)
    }
    
    pub fn get_execution_progress_with_total(&self, state_file: Option<&str>) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let state_file_path = state_file.unwrap_or("execution_state2.json");
        let state = ExecutionState::load_from_file(state_file_path)?;
        let total_nodes = self.get_total_nodes();
        Ok((state.last_processed_index, total_nodes))
    }
    
    pub fn infer_with_resumption(&self, inputs: HashMap<String, OrtValue>, state_file: Option<&str>) -> OrtResult<HashMap<String, OrtValue>> {
        // println!("starting inference1");
        let graph = self.model.graph.as_ref().ok_or(OrtError::InvalidModel)?;
        // println!("starting inference3");

        // Load or create execution state
        let state_file_path = state_file.unwrap_or("execution_state2.json");
        // let tensor_map_path = "tensor_map2.bin";
        let mut execution_state = if let Ok(state) = ExecutionState::load_from_file(state_file_path) {
            println!("Resuming from node index: {}", state.last_processed_index);
            state
        } else {
            println!("Starting fresh execution");
            ExecutionState::new()
        };

        // let mut shape_inference = ShapeInference::new(graph);
        // shape_inference.infer_shapes(&inputs)?;
    
        // Initialize tensor map with only initializers and inputs (no lazy loading of intermediate tensors yet)
        let mut tensor_map: HashMap<String, OrtValue> = HashMap::new();
        
        // Load all initializers into tensor_map (these are always needed)
        println!("Loading initializers...");
        for tensor in &graph.initializer {
            if !tensor.name.is_empty() {
                match self.parse_tensor(tensor) {
                    Ok(parsed_tensor) => {
                        tensor_map.insert(tensor.name.clone(), parsed_tensor);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to parse initializer {}: {:?}", tensor.name, e);
                    }
                }
            }
        }
        
        // Add user-provided inputs, overriding initializers if provided
        for (name, value) in inputs {
            tensor_map.insert(name, value);
        }
        
        println!("Initialized tensor map with {} entries (initializers + inputs)", tensor_map.len());
            
        // } else {
            // If resuming, only add user inputs that aren't already in tensor_map
            // for (name, value) in inputs {
            //     if !tensor_map.contains_key(&name) {
            //         tensor_map.insert(name, value);
            //     }
            // }
        // }
        // let mut i=0;
        println!("total no of nodes is {}, resuming from index: {}", graph.node.len(), execution_state.last_processed_index);
        // Process each node
        // let mut execution_times = Vec::new();
        let started_at = std::time::Instant::now();
        let mut last_save_time = std::time::Instant::now();
        let save_interval = std::time::Duration::from_secs(30); // Save every 30 seconds
        let mut execution_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("./execution_times2.csv")
            .map_err(|e| {
                eprintln!("Warning: Failed to open execution times file: {}", e);
                e
            })
            .ok();
            
        for (node_index, node) in graph.node.iter().enumerate() {
            // Skip already processed nodes
            if node_index < execution_state.last_processed_index {
                // println!("Skipping already processed node {}: {} ({})", node_index, node.name, node.op_type);
                continue;
            }
            let start_time = std::time::Instant::now();
            
            if node.output.is_empty() {
                return Err(OrtError::InvalidModel);
            }
            
            // Lazy load required tensors for this node's inputs
            let required_inputs: Vec<String> = node.input.iter()
                .filter(|name| !name.is_empty())
                .cloned()
                .collect();
            
            // Load any missing tensors from disk
            let missing_inputs: Vec<String> = required_inputs.iter()
                .filter(|name| !tensor_map.contains_key(*name))
                .cloned()
                .collect();
            
            if !missing_inputs.is_empty() {
                println!("Loading {} missing tensors for node {}: {:?}", missing_inputs.len(), node.name, missing_inputs);
                match execution_state.load_tensors_for_inputs(&missing_inputs) {
                    Ok(loaded_tensors) => {
                        for (name, tensor) in loaded_tensors {
                            tensor_map.insert(name, tensor);
                        }
                        println!("✓ Loaded {} tensors on demand", missing_inputs.len());
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load some tensors: {}", e);
                    }
                }
            }
            
            // Collect inputs for the node, allowing for optional inputs
            let node_inputs_p = node.input.clone();
            println!("Processing node {}: {:?}==========================={:?}", node_index, node,node_inputs_p);
            let jio=node;
            // println!("{:?}------------====",jio.output);
            let node_inputs = node_inputs_p
                .iter().enumerate().map(|(index, name)| {
                    if name.is_empty() {
                        // Check if this is at the end of the input list
                        let is_last = index == node.input.len() - 1;
                        let is_followed_by_empty = node.input.iter().skip(index + 1).all(|n| n.is_empty());
                        
                        if is_last || is_followed_by_empty {
                            // Ignore empty-named inputs at the end
                            Ok(None)
                        } else {
                            Ok(Some(OrtValue::Tensor {
                                shape: vec![], // Scalar tensor for empty input
                                dtype: DataType::Float,
                                data: Arc::new(vec![]), // Empty data
                            }))
                        }
                    } else {
                        // Handle required inputs
                        tensor_map
                            .get(name)
                            .cloned()
                            .ok_or_else(|| OrtError::MissingInput(format!("Required input missing for node {}: {}", node.name, name)))
                            .map(Some)
                    }
                })
                .collect::<OrtResult<Vec<Option<OrtValue>>>>()?;
        
            // Filter out None values (optional inputs)
            let node_inputs: Vec<OrtValue> = node_inputs.into_iter().flatten().collect();
    
            let output = if let Some(op) = self.node_registry.get(&node.op_type) {
                // if node.op_type == "Gather"{
                    // println!("{:?}---{:?}",node,node_inputs);
                // }
                op(node, &node_inputs)?
            // } else if node.op_type == "If" {
            //     self.op_if(node, &node_inputs)?
            // } else if node.op_type == "Loop" {
            //     self.op_loop(node, &node_inputs)?
            // } else if node.op_type == "Scan" {
            //     self.op_scan(node, &node_inputs)?
            } else {
                return Err(OrtError::UnsupportedOp(node.op_type.clone()));
            };
            // Track tensor map size before processing
            let tensor_count_before = tensor_map.len();
            
            // Collect new tensors produced by this node
            let mut new_tensors = HashMap::new();
            
            // Handle multiple outputs (if node.output.len() > 1)
            if node.output.len() == 1 {
                new_tensors.insert(node.output[0].clone(), output.clone());
                tensor_map.insert(node.output[0].clone(), output);
                execution_state.tensor_to_node.insert(node.output[0].clone(), node_index);
            } else {
                // If the operator returns multiple outputs, assume output is a Vec<OrtValue>
                if let OrtValue::Sequence(outputs) = output {
                    if outputs.len() != node.output.len() {
                        return Err(OrtError::InvalidModel);
                    }
                    for (out_name, out_value) in node.output.iter().zip(outputs.into_iter()) {
                        new_tensors.insert(out_name.clone(), out_value.clone());
                        tensor_map.insert(out_name.clone(), out_value);
                        execution_state.tensor_to_node.insert(out_name.clone(), node_index);
                    }
                } else {
                    return Err(OrtError::TypeMismatch(format!("Expected multiple outputs but got single value")));
                }
            }
            
            let elapsed = start_time.elapsed();
            let tensor_count_after = tensor_map.len();
            let new_tensor_count = tensor_count_after - tensor_count_before;
            
            println!("Node '{}' (op: {}) took: {:?}, produced {} new tensors, total: {} tensors, running for: {:?}",  node.name, node.op_type, elapsed, new_tensor_count, tensor_count_after, started_at.elapsed());
            
            
            // Print node output information in Python format
            for output_name in &node.output {
                if let Some(tensor_value) = tensor_map.get(output_name) {
                    print_tensor_info(output_name, tensor_value);
                }
            }
            
            
            
            // Update execution state
            // execution_state.processed_nodes.insert(node.name.clone());
            execution_state.last_processed_index = node_index + 1;
            // execution_state.tensor_map_keys = tensor_map.keys().cloned().collect();
            
            // Save only new tensors incrementally with output tracking
            // if let Err(e) = execution_state.save_new_tensors(&new_tensors, node_index, &node.name, &node.output) {
            //     eprintln!("ERROR: Failed to save new tensors: {}", e);
            // }
            
            // Save execution state (lightweight)
            // if let Err(e) = execution_state.save_to_file(state_file_path) {
            //     eprintln!("Warning: Failed to save execution state: {}", e);
            // }
            
            // Append execution time data immediately to file
            let execution_line = format!("{},{},{:?}\n", node.name, node.op_type, elapsed);
            if let Some(ref mut file) = execution_file {
                use std::io::Write;
                if let Err(e) = file.write_all(execution_line.as_bytes()) {
                    eprintln!("Warning: Failed to write execution time: {}", e);
        }
                if let Err(e) = file.flush() {
                    eprintln!("Warning: Failed to flush execution times: {}", e);
                }
            }
            
            // Periodically ensure data is synced to disk
            if last_save_time.elapsed() >= save_interval {
                if let Some(ref mut file) = execution_file {
                    use std::io::Write;
                    if let Err(e) = file.flush() {
                        eprintln!("Warning: Failed to flush execution times: {}", e);
                    }
                }
                last_save_time = std::time::Instant::now();
            }
        }
        
        // Final save of execution times
        // if let Err(e) = std::fs::write("execution_times.csv", execution_times.join("\n")) {
        //     eprintln!("Warning: Failed to save final execution times: {}", e);
        // }
        // Clean up state files on successful completion
        if let Err(e) = std::fs::remove_file(state_file_path) {
            eprintln!("Warning: Failed to clean up state file: {}", e);
        }
        // if let Err(e) = std::fs::remove_file(tensor_map_path) {
        //     eprintln!("Warning: Failed to clean up tensor map file: {}", e);
        // }
        
        // Clean up incremental tensor files
        Self::cleanup_incremental_tensors(&execution_state);
        
        // Collect graph outputs
        graph.output.iter()
            .map(|output| tensor_map.get(&output.name)
                .cloned()
                .ok_or_else(|| OrtError::MissingOutput(output.name.clone()))
                .map(|v| (output.name.clone(), v)))
            .collect()
    }
    
    fn parse_tensor(&self, proto: &TensorProto) -> OrtResult<OrtValue> {
    let shape: Vec<Dimensions> = proto.dims.iter().map(|&d| {
        if d >= 0 {
            Dimensions::Fixed(d as usize)
        } else {
            // Use dimension name or a default symbolic identifier
            Dimensions::Symbolic(format!("dim_{}", d))
        }
    }).collect();
    let total_elements = shape.iter().filter_map(|d| match d {
        Dimensions::Fixed(n) => Some(*n),
        Dimensions::Symbolic(_) => None,
    }).product::<usize>();

    match DataType::try_from(proto.data_type)? {
        DataType::Float => {
                        let data = parse_float_data(proto, total_elements)?;
                        if data.len() != total_elements * 4 {
                            return Err(OrtError::InvalidTensorData(
                                "Float tensor data length mismatch".into(),
                            ));
                        }
                        Ok(OrtValue::Tensor {
                            shape,
                            dtype: DataType::Float,
                            data: Arc::new(data),
                        })
            }
        DataType::Int64 => {
                let data = parse_int64_data(proto, total_elements)?;
                if data.len() != total_elements * 8 {
                    return Err(OrtError::InvalidTensorData(
                        "Int64 tensor data length mismatch".into(),
                    ));
                }
                Ok(OrtValue::Tensor {
                    shape,
                    dtype: DataType::Int64,
                    data: Arc::new(data),
                })
            }
        DataType::Int32 => {
                let data = parse_int32_data(proto, total_elements)?;
                if data.len() != total_elements * 4 {
                    return Err(OrtError::InvalidTensorData(
                        "Int32 tensor data length mismatch".into(),
                    ));
                }
                Ok(OrtValue::Tensor {
                    shape,
                    dtype: DataType::Int32,
                    data: Arc::new(data),
                })
            }
        DataType::String => {
                let strings = proto
                    .string_data
                    .iter()
                    .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                    .collect::<Vec<_>>();
                if strings.len() != total_elements {
                    return Err(OrtError::InvalidTensorData(
                        "String tensor data length mismatch".into(),
                    ));
                }
                Ok(OrtValue::Tensor {
                    shape,
                    dtype: DataType::String,
                    data: Arc::new(strings.join("").into_bytes()),
                })
            }
        DataType::Boolean => {
            let data = if !proto.raw_data.is_empty() {
                if proto.raw_data.len() != total_elements {
                    return Err(OrtError::InvalidTensorData(
                        format!("Boolean raw_data length mismatch: expected {}, got {}", total_elements, proto.raw_data.len()),
                    ));
                }
                proto.raw_data.clone()
            } else {
                if proto.int32_data.len() != total_elements {
                    return Err(OrtError::InvalidTensorData(
                        format!("Boolean int32_data length mismatch: expected {}, got {}", total_elements, proto.int32_data.len()),
                    ));
                }
                let mut bytes = Vec::with_capacity(total_elements);
                for &b in &proto.int32_data {
                    if b != 0 && b != 1 {
                        return Err(OrtError::InvalidTensorData(
                            "Boolean int32_data contains invalid values".into(),
                        ));
                    }
                    bytes.push(if b == 0 { 0 } else { 1 });
                }
                bytes
            };
            Ok(OrtValue::Tensor {
                shape,
                dtype: DataType::Boolean,
                data: Arc::new(data),
            })
            
        },
    }
}


pub fn execute_subgraph(&self,graph: &GraphProto, inputs: HashMap<String, OrtValue>) -> OrtResult<HashMap<String, OrtValue>> {
    let mut tensor_map: HashMap<String, OrtValue> = HashMap::new();
    for tensor in &graph.initializer {
        tensor_map.insert(tensor.name.clone(), Self::parse_tensor(&self,tensor)?);
    }
    tensor_map.extend(inputs);

    for node in &graph.node {
        if node.output.is_empty() {
            return Err(OrtError::InvalidModel);
        }
        let node_inputs = node.input.iter()
            .map(|name| tensor_map.get(name).cloned().ok_or_else(|| OrtError::MissingInput(name.clone())))
            .collect::<OrtResult<Vec<_>>>()?;

        let output = if let Some(op) = self.node_registry.get(&node.op_type) {
            op(node, &node_inputs)?
        } else {
            return Err(OrtError::UnsupportedOp(node.op_type.clone()));
        };
        tensor_map.insert(node.output[0].clone(), output);
    }

    let mut outputs = HashMap::new();
    for output in &graph.output {
        if let Some(value) = tensor_map.get(&output.name) {
            outputs.insert(output.name.clone(), value.clone());
        } else {
            return Err(OrtError::MissingOutput(output.name.clone()));
        }
    }
    Ok(outputs)
}




}

fn parse_float_data(proto: &TensorProto, count: usize) -> OrtResult<Vec<u8>> {
    if !proto.raw_data.is_empty() {
        Ok(proto.raw_data.clone())
    } else {
        let mut bytes = Vec::with_capacity(count * 4);
        for f in &proto.float_data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        Ok(bytes)
    }
}

fn parse_int64_data(proto: &TensorProto, count: usize) -> OrtResult<Vec<u8>> {
    if !proto.raw_data.is_empty() {
        Ok(proto.raw_data.clone())
    } else {
        let mut bytes = Vec::with_capacity(count * 8);
        for i in &proto.int64_data {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        Ok(bytes)
    }

}
fn parse_int32_data(proto: &TensorProto, count: usize) -> OrtResult<Vec<u8>> {
    if !proto.raw_data.is_empty() {
        Ok(proto.raw_data.clone())
    } else {
        let mut bytes = Vec::with_capacity(count * 4);
        for i in &proto.int32_data {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        Ok(bytes)
    }
}
#[test]
fn justprint(){
    print_model_info("./kokoro-v1.0.onnx").expect("Failed to print model info");
}
pub fn print_model_info<P: AsRef<Path>>(path: P) -> OrtResult<()> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let model = ModelProto::decode(&*buffer)?;
    let graph = model.graph.as_ref().ok_or(OrtError::InvalidModel)?;
    println!("\nFirst appearance of each unique op_type:");
    let mut seen_op_types = HashSet::new();
    for (i, node) in graph.node.iter().enumerate() {
        if seen_op_types.insert(node.op_type.clone()) {
            println!("  first occurrence of '{}' is at {}", node.op_type,i);
            println!("{:?}",node)
        }
    }
    return Ok(());
    
    println!("Opcode Versions:");
    for opset in &model.opset_import {
        println!(
            "  Domain: {}, Version: {}",
            opset.domain,
            opset.version
        );
    }

    
    let mut core_ops = HashSet::new();
    let mut vendor_ops = HashSet::new();
    println!("\nModel Initializers:");

    // for initializers in &graph.initializer {
    //     println!("{:?}",initializers)
    // }
    // Print model inputs
    println!("\nModel Inputs:");
    for input in &graph.input {
        let input_name = &input.name;
        let type_info = input.type_proto.as_ref().map_or("Unknown".to_string(), |tp| {
            if let Some(tensor_type) = &tp.tensor_type {
                let dtype = DataType::try_from(tensor_type.elem_type)
                    .map_or(format!("Unknown({})", tensor_type.elem_type), |dt| format!("{:?}", dt));
                let shape = tensor_type.shape.as_ref().map_or("Unknown".to_string(), |shape_proto| {
                    let dims: Vec<String> = shape_proto.dim.iter().map(|dim| {
                        if dim.dim_value >= 0 {
                            dim.dim_value.to_string()
                        } else {
                            "Symbolic".to_string()
                        }
                    }).collect();
                    format!("[{}]", dims.join(", "))
                });
                format!("Type: {}, Shape: {}", dtype, shape)
            } else {
                "Unknown".to_string()
            }
        });
        println!("  - {}: {}", input_name, type_info);
    }

    println!("\nNodes with Input and Output Names:");
    for (i, node) in graph.node.iter().enumerate() {
        if node.op_type.contains('.') {
            vendor_ops.insert(node.op_type.clone());
        } else {
            core_ops.insert(node.op_type.clone());
        }

        // Print node information
        println!(
            "Node {} (OpType: {}):",
            i,
            node.op_type
        );
        println!("  Inputs:");
        for input in &node.input {
            println!("    - {}", input);
        }
        println!("  Outputs:");
        for output in &node.output {
            println!("    - {}", output);
        }
    }

    println!("\nCore Operations:");
    for op in core_ops {
        println!("  {}", op);
    }

    println!("\nVendor Operations:");
    for op in vendor_ops {
        println!("  {}", op);
    }

    Ok(())
}

fn main() -> Result<()> {
    //  print_model_info("./kokoro-v1.0.onnx")?;
    // let engine = OrtEngine::new("./corrected_add_model_i16.onnx")?;
    // return Ok(());
    let engine = OrtEngine::new("./kokoro-v1.0-simplified.onnx")?;
    // let graph=engine.model.graph.unwrap();
    // println!("{}",graph.initializer.len());
    // for i in graph.initializer{
    //     println!("{:?}",i.name);
    // }
let mut npz = NpzReader::new(File::open("./voices-v1.0.bin").unwrap()).unwrap();
    let mut voices = HashMap::new();

    for voice in npz.names().unwrap() {
        let voice_data: Result<Array3<f32>, _> = npz.by_name(&voice);
        if let Ok(voice_data) = voice_data {
            voices.insert(voice, voice_data);
        }
    }

    let sorted_voices = {
        let mut voice_names = voices.keys().cloned().collect::<Vec<_>>();
        voice_names.sort();
        voice_names
    };

    println!("Loaded {} voices: {:?}", voices.len(), sorted_voices);
    // Example input data
    let tokens: Vec<Vec<i64>> = vec![vec![0, 50, 156, 43, 102, 4, 0]]; // [1, 7]
    let speed: f32 = 1.0;

    // Create tokens tensor
    let batch_size = tokens.len();
    let sequence_length = tokens[0].len();
    let tokens_flat: Vec<i64> = tokens.clone().into_iter().flatten().collect();
    let tokens_tensor = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(batch_size), Dimensions::Fixed(sequence_length)],
        dtype: DataType::Int64,
        data: Arc::new(
            tokens_flat
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        ),
    };

    // Create speed tensor
    let speed_tensor = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1)],
        dtype: DataType::Float,
        data: Arc::new(speed.to_le_bytes().to_vec()),
    };

    // Parse style string and blend styles
    let style_str = "af_sarah.4+af_nicole.6";
    let styles: Vec<&str> = style_str.split('+').collect();

    let mut style_names = Vec::new();
    let mut style_portions = Vec::new();

    // Parse style names and portions
    for style in styles {
        if let Some((name, portion)) = style.split_once('.') {
            if let Ok(portion) = portion.parse::<f32>() {
                style_names.push(name);
                style_portions.push(portion * 0.1); // Scale portion to 0.0-1.0 range
            }
        }
    }
    println!("Using styles: {:?}, portions: {:?}", style_names, style_portions);

    // Initialize blended_style as a 1x256 tensor
    let mut blended_style = vec![vec![0.0; 256]; 1];

    // Blend styles from the voices map
    for (name, portion) in style_names.iter().zip(style_portions.iter()) {
        if let Some(voice_data) = voices.get(*name) {
            // Get the style vector for the first token position
            let style_slice = voice_data.slice(ndarray::s![0, 0, ..]);
            for j in 0..256 {
                blended_style[0][j] += style_slice[j] * portion;
            }
        } else {
            println!("Warning: style {} not found in voices", name);
        }
    }

    // Convert blended_style to raw bytes for OrtValue::Tensor
    let flat_data: Vec<f32> = vec![
        -0.16746138, 0.106833816, -0.17197946, -0.17930198, -0.4060307, 0.11337316, -0.05904325, -0.13578473, -0.343865, -0.0030500141, -0.058186237, -0.18617716, 0.3655906, 0.1500281, 0.0323276, -0.2660883, -0.021834578, -0.18887411, 0.15604171, -0.17936222, -0.21674247, -0.08793214, -0.014403321, -0.038582608, 0.005953279, 0.30037892, -0.25818214, 0.14401352, 0.00625191, 0.18139648, 0.1905407, -0.30535796, -0.016582137, -0.06380315, 0.19268999, 0.031495668, -0.10360171, -0.07843726, 0.035174046, 0.047639426, 0.09471621, -0.059944917, 0.07799803, 0.42816967, -0.27074027, -0.059864923, 0.094025224, -0.07608084, -0.009240143, 0.2764985, -0.044961445, -0.22325265, 0.28969276, 0.021382106, 0.09409301, 0.3064245, 0.085562065, -0.018245282, -0.12442948, 0.12522374, 0.20399052, -0.07992236, -0.17870936, -0.03290955, 0.20011769, 0.23295887, -0.0011655795, 0.2106421, 0.029463217, 0.049337372, 0.07007421, 0.06657779, 0.12671578, -0.3048649, -0.17952333, -0.20896465, 0.010621702, 0.16129294, 0.24825078, -0.06730439, 0.14417285, 0.14019054, -0.16492297, 0.07709213, 0.18941414, 0.07108727, -0.16543987, -0.1864754, -0.25925547, -0.011538826, 0.12039098, 0.024524461, 0.09829027, -0.020422952, -0.19386753, -0.13779366, 0.06404631, -0.091026954, 0.1432159, -0.1445843, -0.099253185, -0.27379233, 0.07603142, -0.06384298, 0.20024501, 0.14540523, 0.010894625, 0.18515547, 0.23194641, -0.07801862, -0.03515421, 0.005198706, 0.11977995, 0.028442672, -0.26251578, 0.087687396, -0.09812868, -0.021395776, 0.17591082, 0.00079514645, -0.037736632, 0.16991898, 0.020198015, 0.29645926, 0.21168791, -0.37216398, 0.13653347, -0.06943156, -0.014739413, 0.16784102, 0.48688984, 0.10855578, -0.25430948, -0.13242087, 0.36683533, 0.0017357357, -0.3956462, -0.27680144, 0.29430857, -0.09608546, 0.10188929, -0.1437357, 0.26491192, -0.07434953, 0.2738349, 0.074040905, -0.15176898, -0.13395815, 0.3927017, -0.14603326, 0.26794004, 0.06925736, -0.111301675, 0.45458955, -0.21831812, -0.15351343, -0.14352655, 0.2463764, 0.59878033, -0.28609738, 0.21620028, 0.16584155, 0.26237804, 0.639141, 0.48741198, 0.28353006, 0.20943506, -0.005696906, 0.0027122672, -0.2647833, 0.20146331, 0.7051931, -0.33182484, 0.12572102, -0.18048556, -0.886673, -0.18763334, 0.11108457, 0.04415555, -0.4453653, 0.7829914, 0.23367575, 0.07653396, -0.058281526, 0.63499576, -0.12139675, 0.10016927, -0.24464339, -0.169406, -0.37613553, -0.0048745424, -0.05477307, 0.21715853, 0.44753513, 0.08324612, -0.34354436, 0.20547722, 0.14335431, 0.15277404, 0.137537, 0.014170506, -0.48911935, -0.35340762, 0.09423898, 0.56586313, 0.21005873, 0.3004918, 0.20001253, -0.21485168, 0.2627742, -0.077053934, -0.22529292, -0.18517601, -0.077634186, 0.13139398, -0.22406033, 0.2564357, -0.32308036, -0.49612147, 0.6047083, 0.04769512, 0.13776457, -0.45326835, -0.01648686, -0.36213535, -0.12455494, 0.23899081, -0.019421533, -0.11391652, -0.010070331, 0.17823072, 0.12505603, -0.19111425, 0.36101788, 0.35537708, -0.31394705, 0.05328191, -0.30892166, -0.11983204, 0.2664771, -0.0821251, 0.28563684, 0.13965864, 0.5084046, -0.09097928, 0.4581191, -0.094962835, 0.6173904, -0.07400553, -0.17739949, -0.12363535, 0.5033887, 0.23096946, -0.11120826, -0.12198155, -0.10832049
    ];
    // let flat_data: Vec<f32> = blended_style.into_iter().flatten().collect();
    // println!("{:?}",flat_data);
    // return Ok(());
    let data_bytes: Vec<u8> = flat_data
        .into_iter()
        .flat_map(|x| x.to_le_bytes().to_vec())
        .collect();

    // Create the style tensor
    let style_tensor = OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(256)], // Shape [1, 256]
        dtype: DataType::Float,
        data: Arc::new(data_bytes),
    };

    // Create input HashMap
    let mut inputs = HashMap::new();
    inputs.insert("tokens".to_string(), tokens_tensor);
    inputs.insert("style".to_string(), style_tensor);
    inputs.insert("speed".to_string(), speed_tensor);


// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------above is used code-------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
    
   


// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------below is used code-------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
    // Run inference
    // println!("Running inference...");
    let outputs = engine.infer(inputs)?;
    
    // // Process outputs
    println!("Inference complete. Outputs:");
    match outputs.get("audio") {
                Some(a) => {
                    println!("{}",a);
                    // let a=a;
                    // for (key, value) in  {
                    //     println!("Key: {:?}, Value: {:?}", key, value);
                    // }
                }
                Some(_) => println!("Error: 'predictions' output is not a Map"),
                None => println!("Error: 'predictions' output not found"),
            }
    // for (name, value) in &outputs {
    //     println!("  - {}: {:?}", name, value);
    // }
    
    Ok(())
}
// Create a tensor directly instead of a sequence
    // Example input data
//     let tokens: Vec<Vec<i64>> = vec![vec![0, 50, 156, 43, 102, 4, 0]]; // [1, 7]
//     let styles: Vec<Vec<f32>> = vec![vec![0.1, 0.2, 0.3, 0.4]]; // [1, 4] (adjust style_dim as needed)
//     let speed: f32 = 1.0;

//     // Create tokens tensor
//     let batch_size = tokens.len();
//     let sequence_length = tokens[0].len();
//     let tokens_flat: Vec<i64> = tokens.clone().into_iter().flatten().collect();
//     let tokens_tensor = OrtValue::Tensor {
//         shape: vec![Dimensions::Fixed(batch_size), Dimensions::Fixed(sequence_length)],
//         dtype: DataType::Int64,
//         data: Arc::new(
//             tokens_flat
//                 .iter()
//                 .flat_map(|x| x.to_le_bytes())
//                 .collect::<Vec<u8>>(),
//         ),
//     };

//     // Create style tensor
//     let style_dim = styles[0].len();
//     let style_flat: Vec<f32> = styles.into_iter().flatten().collect();
//     let style_tensor = OrtValue::Tensor {
//         shape: vec![Dimensions::Fixed(batch_size), Dimensions::Fixed(style_dim)],
//         dtype: DataType::Float,
//         data: Arc::new(
//             style_flat
//                 .iter()
//                 .flat_map(|x| x.to_le_bytes())
//                 .collect::<Vec<u8>>(),
//         ),
//     };

//     // Create speed tensor
//     let speed_tensor = OrtValue::Tensor {
//         shape: vec![Dimensions::Fixed(1)],
//         dtype: DataType::Float,
//         data: Arc::new(speed.to_le_bytes().to_vec()),
//     };

//     // Create input HashMap
//     let mut inputs = HashMap::new();
//     // fn load_voices(voices_path: &str) -> HashMap<String, Vec<[[f32; 256]; 1]>> {
//         let mut npz = NpzReader::new(File::open("/root/github/Kokoros/data/voices-v1.0.bin").unwrap()).unwrap();
//         let mut map = HashMap::new();

//         for voice in npz.names().unwrap() {
//             let voice_data: Result<Array3<f32>, _> = npz.by_name(&voice);
//             let voice_data = voice_data.unwrap();
//             let mut tensor = vec![[[0.0; 256]; 1]; 511];
//             for (i, inner_value) in voice_data.outer_iter().enumerate() {
//                 for (j, inner_inner_value) in inner_value.outer_iter().enumerate() {
//                     for (k, number) in inner_inner_value.iter().enumerate() {
//                         tensor[i][j][k] = *number;
//                     }
//                 }
//             }
//             map.insert(voice, tensor);
//         }

//         let sorted_voices = {
//             let mut voices = map.keys().collect::<Vec<_>>();
//             voices.sort();
//             voices
//         };

//         eprintln!("voice styles loaded: {:?}", sorted_voices);
//     // }
// //     let vocab_size = 22; // Adjust based on model
// // let embedding_dim = 500; // Adjust based on model
// // let embedding_data: Vec<f32> = vec![0.0; vocab_size * embedding_dim]; // Dummy data
// // let embedding_tensor = OrtValue::Tensor {
// //     shape: vec![Dimensions::Fixed(vocab_size), Dimensions::Fixed(embedding_dim)],
// //     dtype: DataType::Float,
// //     data: Arc::new(
// //         embedding_data
// //             .iter()
// //             .flat_map(|x| x.to_le_bytes())
// //             .collect::<Vec<u8>>(),
// //     ),
// // };
// // inputs.insert("encoder.bert.embeddings.word_embeddings.weight".to_string(), embedding_tensor);

// // let vocab_size = 22; // Adjust based on model
// // let embedding_dim = 500; // Adjust based on model
// // let embedding_data: Vec<f32> = vec![0.0; vocab_size * embedding_dim]; // Dummy data
// // let embedding_tensor = OrtValue::Tensor {
// //     shape: vec![Dimensions::Fixed(vocab_size), Dimensions::Fixed(embedding_dim)],
// //     dtype: DataType::Float,
// //     data: Arc::new(
// //         embedding_data
// //             .iter()
// //             .flat_map(|x| x.to_le_bytes())
// //             .collect::<Vec<u8>>(),
// //     ),
// // };
// // inputs.insert("encoder.text_encoder.embedding.weight".to_string(), embedding_tensor);
// let styles: Vec<&str> = "af_sarah.4+af_nicole.6".split('+').collect();

// let mut style_names = Vec::new();
// let mut style_portions = Vec::new();

// // Parse style names and portions
// for style in styles {
//     if let Some((name, portion)) = style.split_once('.') {
//         if let Ok(portion) = portion.parse::<f32>() {
//             style_names.push(name);
//             style_portions.push(portion * 0.1); // Scale portion to 0.0-1.0 range
//         }
//     }
// }
// eprintln!("styles: {:?}, portions: {:?}", style_names, style_portions);

// // Initialize blended_style as a 1x256 tensor
// let mut blended_style = vec![vec![0.0; 256]; 1];

// // Blend styles from the map
// for (name, portion) in style_names.iter().zip(style_portions.iter()) {
//     if let Some(style) = map.get(*name) {
//         // Ensure tokens.len() is valid for indexing style
//         if tokens.len() < style.len() {
//             let style_slice = &style[tokens.len()][0]; // [256] array
//             for j in 0..256 {
//                 blended_style[0][j] += style_slice[j] * portion;
//             }
//         } else {
//             eprintln!("Error: tokens.len()={} exceeds style dimension={}", tokens.len(), style.len());
//         }
//     } else {
//         eprintln!("Error: style {} not found in map", name);
//     }
// }

// // Convert blended_style to raw bytes for OrtValue::Tensor
// let flat_data: Vec<f32> = blended_style.into_iter().flatten().collect();
// let data_bytes: Vec<u8> = flat_data
//     .into_iter()
//     .flat_map(|x| x.to_le_bytes().to_vec())
//     .collect();

// // Create the OrtValue::Tensor with correct shape and data
// inputs.insert(
//     "style".to_string(),
//     OrtValue::Tensor {
//         shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(256)], // Shape [1, 256]
//         dtype: DataType::Float,
//         data: Arc::new(data_bytes),
//     },
// );
// inputs.insert("tokens".to_string(), tokens_tensor);
//     inputs.insert("speed".to_string(), speed_tensor);
//     let outputs = engine.infer(inputs)?;
//     match outputs.get("audio") {
//         Some(OrtValue::Map(result_map)) => {
//             for (key, value) in result_map {
//                 println!("Key: {:?}, Value: {:?}", key, value);
//             }
//         }
//         Some(_) => println!("Error: 'predictions' output is not a Map"),
//         None => println!("Error: 'predictions' output not found"),
//     }
//     Ok(())
// } 
   // Sequence Operators Implementation
   
