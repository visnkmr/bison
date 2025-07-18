// #![recursion_limit = "5684"]

use std::sync::atomic::{AtomicUsize, Ordering};
mod core_ops;
pub use core_ops::*;
#[cfg(test)]
mod tests {
    pub mod tensor_ops_test;
    pub mod parser_test;
    pub mod onnx_model_test;
    pub mod sequence_map_test;
    pub mod bert_ops_test;
    pub mod kokoro_ops_test;
}

// Static counter for node indexing
static NODE_COUNTER: AtomicUsize = AtomicUsize::new(0);
use anyhow::Result;
use indexmap::IndexMap;
use prost::Message;
use serde::Serialize;
use ndarray_npy::NpzReader;
use std::{
    collections::{HashMap, HashSet},
    fmt,
    fs::File,
    io::Read,
    path::Path,
    sync::Arc,
};
use thiserror::Error;
use ndarray::{Array, Array2, Array3, Array4, ArrayD, Axis, Dimension, Zip};
use num_traits::{Float, ToBytes};

// Simplified ONNX type definitions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    Float,
    Int64,
    String,
}

impl DataType {
    fn try_from(value: i32) -> Result<Self, OrtError> {
        match value {
            1 => Ok(DataType::Float),
            7 => Ok(DataType::Int64),
            8 => Ok(DataType::String),
            other => Err(OrtError::UnknownDataType(other)),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TensorProto {
    pub name: String,
    pub doc_string:String,
    pub data_type: i32,
    pub dims: Vec<i64>,
    pub float_data: Vec<f32>,
    pub int64_data: Vec<i64>,
    pub string_data: Vec<Vec<u8>>,
    pub raw_data: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct AttributeProto {
    pub name: String,
    pub i: i64,
    pub f: f32,
    pub s: Vec<u8>,
    pub t: Option<TensorProto>,
    pub ints: Vec<i64>,
    pub floats: Vec<f32>,
    pub g: Option<GraphProto>, // Added for subgraphs
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
            1 => prost::encoding::string::merge(wire_type, &mut self.domain, buf, ctx)?,
            2 => prost::encoding::int64::merge(wire_type, &mut self.version, buf, ctx)?,
            _ => prost::encoding::skip_field(wire_type, tag, buf, ctx)?,
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
                let mut dim = 0i64;
                prost::encoding::int64::merge(wire_type, &mut dim, buf, ctx)?;
                self.dims.push(dim);
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
                 println!("TensorProto: Added raw_data (length: {})", self.raw_data.len());//nowcommented 
            }
            6 => {
                let mut string_data = Vec::new();
                prost::encoding::bytes::merge(wire_type, &mut string_data, buf, ctx)?;
                self.string_data.push(string_data.clone());
                //nowcommented  println!("TensorProto: Added string_data: {:?}", String::from_utf8_lossy(&string_data));
            }
            4 => {
                // let mut float_data = 0f32;
                prost::encoding::float::merge_repeated(wire_type, &mut self.float_data, buf, ctx)?;
                // self.float_data.push(float_data);
                //nowcommented  println!("TensorProto: Added float_data: {}", float_data);
            }
           7 => {
                let mut int64_data = 0i64;
                prost::encoding::int64::merge(wire_type, &mut int64_data, buf, ctx)?;
                self.int64_data.push(int64_data);
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
                //nowcommented  println!("ValueInfoProto: Skipping unknown tag: {}", tag);
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
                let mut opset = OpSetImport::default();
                prost::encoding::message::merge(wire_type, &mut opset, buf, ctx)?;
                self.opset_import.push(opset);
                //nowcommented  println!("ModelProto: Added opset_import (domain: {}, version: {})", self.opset_import.last().unwrap().domain, self.opset_import.last().unwrap().version);
            }
            _ => {
                //nowcommented  println!("ModelProto: Skipping unknown tag: {}", tag);
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
                let node_index = NODE_COUNTER.fetch_add(1, Ordering::SeqCst);
                let mut node = NodeProto::default();
                prost::encoding::message::merge(wire_type, &mut node, buf, ctx)?;
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
                self.node.push(node);
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
                println!("GraphProto: Set doc_string: {:?}", bytes);
                // println!("GraphProto: Set doc_string: {}", self.doc_string);
                //nowcommented  println!("GraphProto: Set doc_string: {}", self.doc_string);
            }
            5 => {
                let mut init = TensorProto::default();
                prost::encoding::message::merge(wire_type, &mut init, buf, ctx)?;
                  println!("GraphProto: Added initializer (name: {})", init.name); //nowcommented
                self.initializer.push(init);
            }
            12 => {
                let mut output = ValueInfoProto::default();
                prost::encoding::message::merge(wire_type, &mut output, buf, ctx)?;
                //nowcommented  println!("GraphProto: Added output (name: {})", output.name);
                self.output.push(output);
            }
            11 => {
                let mut input = ValueInfoProto::default();
                prost::encoding::message::merge(wire_type, &mut input, buf, ctx)?;
                //nowcommented  println!("GraphProto: Added input (name: {})", input.name);
                self.input.push(input);
            }
            13 => {
                let mut value_info = ValueInfoProto::default();
                prost::encoding::message::merge(wire_type, &mut value_info, buf, ctx)?;
                //nowcommented  println!("GraphProto: Added value_info (name: {})", value_info.name);
                self.value_info.push(value_info);
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
        NODE_COUNTER.store(0, Ordering::SeqCst);
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
                let mut input = String::new();
                prost::encoding::string::merge(wire_type, &mut input, buf, ctx)?;
                self.input.push(input.clone());
                //nowcommented  println!("NodeProto: Added input: {}", input);
            }
            2 => {
                let mut output = String::new();
                prost::encoding::string::merge(wire_type, &mut output, buf, ctx)?;
                self.output.push(output.clone());
                //nowcommented  println!("NodeProto: Added output: {}", output);
            }
            3 => {
                let mut bytes = Vec::new();
                prost::encoding::bytes::merge(wire_type, &mut bytes, buf, ctx)?;
                // Attempt to convert bytes to UTF-8 string, fallback to empty string if invalid
                self.name = String::from_utf8_lossy(&bytes).into_owned();
                // println!("GraphProto: Set doc_string: {:?}", bytes);
                // prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?;
                println!("NodeProto: Set name: {}", self.name);
            }
            4 => {
                prost::encoding::string::merge(wire_type, &mut self.op_type, buf, ctx)?;
                //nowcommented  println!("NodeProto: Set op_type: {}", self.op_type);
            }
            5 => {
                let mut bytes = Vec::new();
                prost::encoding::bytes::merge(wire_type, &mut bytes, buf, ctx)?;
                // Attempt to convert bytes to UTF-8 string, fallback to empty string if invalid
                self.domain = String::from_utf8_lossy(&bytes).into_owned();
                // println!("GraphProto: Set doc_string: {:?}", bytes);
                // prost::encoding::string::merge(wire_type, &mut self.domain, buf, ctx)?;
                println!("NodeProto: Set domain: {}", self.domain);
            }
            7 => {
                let mut attr = AttributeProto::default();
                prost::encoding::message::merge(wire_type, &mut attr, buf, ctx)?;
                //nowcommented  println!("NodeProto: Added attribute (name: {})", attr.name);
                self.attributes.push(attr);
            }
             8 => {
                // Handle subgraph attributes (e.g., then_branch, else_branch, body)
                let mut subgraph = GraphProto::default();
                prost::encoding::message::merge(wire_type, &mut subgraph, buf, ctx)?;
                //nowcommented  println!("NodeProto: Added subgraph for attribute");
                self.subgraphs.insert(format!("subgraph_{}", self.subgraphs.len()), subgraph);
            }
            _ => {
                //nowcommented  println!("NodeProto: Skipping unknown tag: {}", tag);
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
                println!("AttributeProto: Set name: {}", self.name);
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
                //nowcommented  println!("AttributeProto: Skipping unknown tag: {}", tag);
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
                //nowcommented  println!("TypeProto: Skipping unknown tag: {}", tag);
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
                //nowcommented  println!("TensorTypeProto: Skipping unknown tag: {}", tag);
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
                let mut dim = TensorDimension::default();
                prost::encoding::message::merge(wire_type, &mut dim, buf, ctx)?;
                //nowcommented  println!("TensorShapeProto: Added dimension (dim_value: {})", dim.dim_value);
                self.dim.push(dim);
            }
            _ => {
                //nowcommented  println!("TensorShapeProto: Skipping unknown tag: {}", tag);
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
                //nowcommented  println!("Dimension: Skipping unknown tag: {}", tag);
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
    TypeMismatch(&'static str),
    #[error("Index error: {0}")]
    IndexError(&'static str),
    #[error("Invalid model")]
    InvalidModel,
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
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Dimensions {
    Fixed(usize),
    Symbolic(String),
}
#[derive(Clone, Serialize)]
pub enum OrtValue {
    Tensor {
        
        shape: Vec<Dimensions>,
        #[serde(skip_serializing)]
        dtype: DataType,
        #[serde(skip_serializing)]
        data: Arc<Vec<u8>>,
    },
    Sequence(Vec<OrtValue>),
    Map(IndexMap<MapKey, OrtValue>),
    Opaque(Vec<u8>),
}

impl fmt::Debug for OrtValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrtValue::Tensor { shape, dtype, .. } => f
                .debug_struct("Tensor")
                .field("shape", shape)
                .field("dtype", dtype)
                .finish(),
            OrtValue::Sequence(seq) => f.debug_tuple("Sequence").field(seq).finish(),
            OrtValue::Map(map) => f.debug_tuple("Map").field(map).finish(),
            OrtValue::Opaque(o) => f.debug_tuple("Opaque").field(o).finish(),
        }
    }
}

impl OrtValue {
    pub fn shape(&self) -> &Vec<Dimensions> {
        match self {
            OrtValue::Tensor { shape, .. } => shape,
            _ => panic!("Shape only available for Tensor variant"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
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
                return Err(OrtError::TypeMismatch(&(""
                    // "Input shapes must match for {}: got {:?} and {:?}", 
                    // node.op_type, input_shapes[0], input_shapes[1]
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
                return Err(OrtError::TypeMismatch(&(""
                    // "MatMul inner dimensions mismatch: {} vs {}", 
                    // shape1[ndim1 - 1], shape2[ndim2 - 2]
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
                    return Err(OrtError::TypeMismatch(&(""
                    // return Err(OrtError::TypeMismatch(&format!(
                    //     "Concat input shapes must have same rank: {:?}", input_shapes
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
                        return Err(OrtError::TypeMismatch(&(""
                        // return Err(OrtError::TypeMismatch(&format!(""
                            // "Concat non-axis dimensions must match: {:?}", input_shapes
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
pub struct OrtEngine {
    model: ModelProto,
    node_registry: HashMap<String, fn(&NodeProto, &[OrtValue]) -> OrtResult<OrtValue>>,
    vendor_ops: HashMap<String, fn(&[u8], &[OrtValue]) -> OrtResult<OrtValue>>,
    shape_inference: ShapeInference, // Added
}

// Helper function to convert OrtValue to ndarray
fn ort_to_ndarray(ort: &OrtValue) -> OrtResult<ArrayD<f32>> {
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

fn ndarray_to_ort(array: ArrayD<f32>, dtype: DataType) -> OrtValue {
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
        self.node_registry.insert("STFT".into(), Self::op_stft);
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
        self.node_registry.insert("ConvTranspose".into(), Self::op_conv_transpose);
        self.node_registry.insert("Pad".into(), Self::op_pad);
        self.node_registry.insert("Reshape".into(), Self::op_reshape);
        self.node_registry.insert("ScatterND".into(), Self::op_scatter_nd);
        self.node_registry.insert("Where".into(), Self::op_where);
        self.node_registry.insert("Sin".into(), Self::op_sin);
        self.node_registry.insert("LSTM".into(), Self::op_lstm);
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
        self.node_registry.insert("Conv".into(), Self::op_conv);
        self.node_registry.insert("LayerNormalization".into(), Self::op_layer_normalization);
        self.node_registry.insert("Gemm".into(), Self::op_gemm);
        
        // BERT-specific operators
        self.node_registry.insert("Erf".into(), Self::op_erf);
        self.node_registry.insert("Gelu".into(), Self::op_gelu);
        self.node_registry.insert("Split".into(), Self::op_split);
        self.node_registry.insert("Dropout".into(), Self::op_dropout);
        self.node_registry.insert("Einsum".into(), Self::op_einsum);
        self.node_registry.insert("TopK".into(), Self::op_topk);
        self.node_registry.insert("GatherElements".into(), Self::op_gather_elements);
        self.node_registry.insert("GatherND".into(), Self::op_gather_nd);
        self.node_registry.insert("ReduceMax".into(), Self::op_reduce_max);
        self.node_registry.insert("Attention".into(), Self::op_attention);
        
        // Kokoro-specific operators
        self.node_registry.insert("Embedding".into(), Self::op_embedding);
        self.node_registry.insert("ConstantOfShapeInt64".into(), Self::op_constant_of_shape_int64);
        self.node_registry.insert("LayerNormalizationWithEpsilon".into(), Self::op_layer_normalization_with_epsilon);
        self.node_registry.insert("Expand".into(), Self::op_expand);
        self.node_registry.insert("PositionEmbeddings".into(), Self::op_position_embeddings);
        self.node_registry.insert("TokenTypeEmbeddings".into(), Self::op_token_type_embeddings);
        self.node_registry.insert("BertAttention".into(), Self::op_bert_attention);
        self.node_registry.insert("BertIntermediate".into(), Self::op_bert_intermediate);
        self.node_registry.insert("BertOutput".into(), Self::op_bert_output);
        self.node_registry.insert("BertPooler".into(), Self::op_bert_pooler);
        
        // Sequence operators
        self.node_registry.insert("SequenceAt".into(), Self::op_sequence_at);
        self.node_registry.insert("SequenceConstruct".into(), Self::op_sequence_construct);
        self.node_registry.insert("SequenceEmpty".into(), Self::op_sequence_empty);
        self.node_registry.insert("SequenceErase".into(), Self::op_sequence_erase);
        self.node_registry.insert("SequenceInsert".into(), Self::op_sequence_insert);
        self.node_registry.insert("SequenceLength".into(), Self::op_sequence_length);
        
        // Map operators
        self.node_registry.insert("MapFromTensor".into(), Self::op_map_from_tensor);
        self.node_registry.insert("MapToTensor".into(), Self::op_map_to_tensor);
        self.node_registry.insert("MapGet".into(), Self::op_map_get);
        self.node_registry.insert("MapHasKey".into(), Self::op_map_has_key);
        self.node_registry.insert("MapKeys".into(), Self::op_map_keys);
        
        // Control flow operators (improved)
        
        
        // Other operators
        self.node_registry.insert("CumSum".into(), Self::op_cumsum);
        self.node_registry.insert("NonZero".into(), Self::op_nonzero);
        self.node_registry.insert("ScatterND".into(), Self::op_scatter_nd);
        self.node_registry.insert("Conv".into(), Self::op_conv);
        self.node_registry.insert("ConvTranspose".into(), Self::op_conv_transpose);
        self.node_registry.insert("LSTM".into(), Self::op_lstm);
        self.node_registry.insert("STFT".into(), Self::op_stft);
        self.node_registry.insert("Resize".into(), Self::op_resize);
    }
    pub fn infer(&self, inputs: HashMap<String, OrtValue>) -> OrtResult<HashMap<String, OrtValue>> {
        // println!("starting inference1");
        let graph = self.model.graph.as_ref().ok_or(OrtError::InvalidModel)?;
        // println!("starting inference3");

        // let mut shape_inference = ShapeInference::new(graph);
        // shape_inference.infer_shapes(&inputs)?;
    
        let mut tensor_map: HashMap<String, OrtValue> = HashMap::new();
        // println!("starting inference2");
        // Load all initializers into tensor_map
        for tensor in &graph.initializer {
            println!("{:?}",tensor);

            if !tensor.name.is_empty() {
                match self.parse_tensor(tensor) {
                    Ok(parsed_tensor) => {
                        tensor_map.insert(tensor.name.clone(), parsed_tensor);
                        println!("{:?}",tensor_map);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to parse initializer {}: {:?}", tensor.name, e);
                        // Continue to allow partial parsing; skip invalid initializers
                    }
                }
            }
        }
        // Add user-provided inputs, overriding initializers if provided
        for (name, value) in inputs {
            tensor_map.insert(name, value);
        }
    
        // Process each node
        for node in &graph.node {
            if node.output.is_empty() {
                return Err(OrtError::InvalidModel);
            }
            // Collect inputs for the node, allowing for optional inputs
            let node_inputs = node.input.iter()
                .map(|name| {
                    if name.is_empty() {
                        // Handle optional inputs (empty name indicates optional input)
                        Ok(None)
                    } else {
                        tensor_map.get(name)
                            .cloned()
                            .ok_or_else(|| OrtError::MissingInput(format!("Required input missing: {}", name)))
                            .map(Some)
                    }
                })
                .collect::<OrtResult<Vec<Option<OrtValue>>>>()?;
    
            // Filter out None values (optional inputs)
            let node_inputs: Vec<OrtValue> = node_inputs.into_iter().flatten().collect();
    
            let output = if let Some(op) = self.node_registry.get(&node.op_type) {
                op(node, &node_inputs)?
            } else if node.op_type == "If" {
                self.op_if(node, &node_inputs)?
            } else if node.op_type == "Loop" {
                self.op_loop(node, &node_inputs)?
            } else if node.op_type == "Scan" {
                self.op_scan(node, &node_inputs)?
            } else {
                return Err(OrtError::UnsupportedOp(node.op_type.clone()));
            };
            // Handle multiple outputs (if node.output.len() > 1)
            if node.output.len() == 1 {
                tensor_map.insert(node.output[0].clone(), output);
            } else {
                // If the operator returns multiple outputs, assume output is a Vec<OrtValue>
                if let OrtValue::Sequence(outputs) = output {
                    if outputs.len() != node.output.len() {
                        return Err(OrtError::InvalidModel);
                    }
                    for (out_name, out_value) in node.output.iter().zip(outputs.into_iter()) {
                        tensor_map.insert(out_name.clone(), out_value);
                    }
                } else {
                    return Err(OrtError::TypeMismatch("Expected multiple outputs but got single value"));
                }
            }
        }
    
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
pub fn print_model_info<P: AsRef<Path>>(path: P) -> OrtResult<()> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let model = ModelProto::decode(&*buffer)?;

    println!("Opcode Versions:");
    for opset in &model.opset_import {
        println!(
            "  Domain: {}, Version: {}",
            opset.domain,
            opset.version
        );
    }

    let graph = model.graph.as_ref().ok_or(OrtError::InvalidModel)?;
    let mut core_ops = HashSet::new();
    let mut vendor_ops = HashSet::new();
    println!("\nModel Initializers:");

    for initializers in &graph.initializer {
        println!("{:?}",initializers)
    }
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
    //  OrtEngine::print_model_info("./kokoro-v1.0.onnx")?;
    let engine = OrtEngine::new("./kokoro-v1.0-simplified.onnx")?;
    let graph=engine.model.graph.unwrap();
    for i in graph.initializer{
        println!("{:?}",i);
    }
// let mut npz = NpzReader::new(File::open("./voices-v1.0.bin").unwrap()).unwrap();
//     let mut voices = HashMap::new();

//     for voice in npz.names().unwrap() {
//         let voice_data: Result<Array3<f32>, _> = npz.by_name(&voice);
//         if let Ok(voice_data) = voice_data {
//             voices.insert(voice, voice_data);
//         }
//     }

//     let sorted_voices = {
//         let mut voice_names = voices.keys().cloned().collect::<Vec<_>>();
//         voice_names.sort();
//         voice_names
//     };

//     println!("Loaded {} voices: {:?}", voices.len(), sorted_voices);
//     // Example input data
//     let tokens: Vec<Vec<i64>> = vec![vec![0, 50, 156, 43, 102, 4, 0]]; // [1, 7]
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

//     // Create speed tensor
//     let speed_tensor = OrtValue::Tensor {
//         shape: vec![Dimensions::Fixed(1)],
//         dtype: DataType::Float,
//         data: Arc::new(speed.to_le_bytes().to_vec()),
//     };

//     // Parse style string and blend styles
//     let style_str = "af_sarah.4+af_nicole.6";
//     let styles: Vec<&str> = style_str.split('+').collect();

//     let mut style_names = Vec::new();
//     let mut style_portions = Vec::new();

//     // Parse style names and portions
//     for style in styles {
//         if let Some((name, portion)) = style.split_once('.') {
//             if let Ok(portion) = portion.parse::<f32>() {
//                 style_names.push(name);
//                 style_portions.push(portion * 0.1); // Scale portion to 0.0-1.0 range
//             }
//         }
//     }
//     println!("Using styles: {:?}, portions: {:?}", style_names, style_portions);

//     // Initialize blended_style as a 1x256 tensor
//     let mut blended_style = vec![vec![0.0; 256]; 1];

//     // Blend styles from the voices map
//     for (name, portion) in style_names.iter().zip(style_portions.iter()) {
//         if let Some(voice_data) = voices.get(*name) {
//             // Get the style vector for the first token position
//             let style_slice = voice_data.slice(ndarray::s![0, 0, ..]);
//             for j in 0..256 {
//                 blended_style[0][j] += style_slice[j] * portion;
//             }
//         } else {
//             println!("Warning: style {} not found in voices", name);
//         }
//     }

//     // Convert blended_style to raw bytes for OrtValue::Tensor
//     let flat_data: Vec<f32> = blended_style.into_iter().flatten().collect();
//     let data_bytes: Vec<u8> = flat_data
//         .into_iter()
//         .flat_map(|x| x.to_le_bytes().to_vec())
//         .collect();

//     // Create the style tensor
//     let style_tensor = OrtValue::Tensor {
//         shape: vec![Dimensions::Fixed(1), Dimensions::Fixed(256)], // Shape [1, 256]
//         dtype: DataType::Float,
//         data: Arc::new(data_bytes),
//     };

//     // Create input HashMap
//     let mut inputs = HashMap::new();
//     inputs.insert("tokens".to_string(), tokens_tensor);
//     inputs.insert("style".to_string(), style_tensor);
//     inputs.insert("speed".to_string(), speed_tensor);


// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------above is used code-------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
    
    // Add the missing embeddings weight tensors
    // Creating placeholder tensors with appropriate dimensions
    // Typically these would be loaded from a pre-trained model
    // let embedding_dim = 768; // Standard embedding dimension
    // let vocab_size = 30522; // Standard vocabulary size
    
    // Create zero-filled tensors for the embeddings
    // let bert_embeddings_data = vec![0.0f32; vocab_size * embedding_dim];
    // let bert_embeddings_bytes: Vec<u8> = bert_embeddings_data
    //     .iter()
    //     .flat_map(|x| x.to_le_bytes().to_vec())
    //     .collect();
        
    // let bert_embeddings_tensor = OrtValue::Tensor {
    //     shape: vec![Dimensions::Fixed(vocab_size), Dimensions::Fixed(embedding_dim)],
    //     dtype: DataType::Float,
    //     data: Arc::new(bert_embeddings_bytes),
    // };
    
    // Create another embedding tensor for the text encoder
    // let text_encoder_embeddings_data = vec![0.0f32; vocab_size * embedding_dim];
    // let text_encoder_embeddings_bytes: Vec<u8> = text_encoder_embeddings_data
    //     .iter()
    //     .flat_map(|x| x.to_le_bytes().to_vec())
    //     .collect();
        
    // let text_encoder_embeddings_tensor = OrtValue::Tensor {
    //     shape: vec![Dimensions::Fixed(vocab_size), Dimensions::Fixed(embedding_dim)],
    //     dtype: DataType::Float,
    //     data: Arc::new(text_encoder_embeddings_bytes),
    // };
    
    // inputs.insert("encoder.bert.embeddings.word_embeddings.weight".to_string(), bert_embeddings_tensor);
    // inputs.insert("encoder.text_encoder.embedding.weight".to_string(), text_encoder_embeddings_tensor);
    
    // // Add the constant tensor needed by the encoder
    // // This is likely a shape or dimension tensor used in the model
    // let constant_data = vec![1.0f32; 1]; // Simple 1-element tensor with value 1.0
    // let constant_bytes: Vec<u8> = constant_data
    //     .iter()
    //     .flat_map(|x| x.to_le_bytes().to_vec())
    //     .collect();
        
    // let constant_tensor = OrtValue::Tensor {
    //     shape: vec![Dimensions::Fixed(1)],
    //     dtype: DataType::Float,
    //     data: Arc::new(constant_bytes),
    // };
    
    // inputs.insert("shape/encoder/Constant_3_output_0".to_string(), constant_tensor);


// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------below is used code-------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
    // Run inference
    // println!("Running inference...");
    // let outputs = engine.infer(inputs)?;
    
    // // Process outputs
    // println!("Inference complete. Outputs:");
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
   
