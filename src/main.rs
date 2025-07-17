// #![recursion_limit = "5684"]

use std::sync::atomic::{AtomicUsize, Ordering};

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
                let mut dim = 0i64;
                prost::encoding::int64::merge(wire_type, &mut dim, buf, ctx)?;
                self.dims.push(dim);
                println!("TensorProto: Added dimension: {}", dim);
            }
            2 => {
                prost::encoding::int32::merge(wire_type, &mut self.data_type, buf, ctx)?;
                println!("TensorProto: Set data_type: {}", self.data_type);
            }
            4 => {
                prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?;
                println!("TensorProto: Set name: {}", self.name);
            }
            7 => {
                prost::encoding::bytes::merge(wire_type, &mut self.raw_data, buf, ctx)?;
                println!("TensorProto: Added raw_data (length: {})", self.raw_data.len());
            }
            8 => {
                let mut string_data = Vec::new();
                prost::encoding::bytes::merge(wire_type, &mut string_data, buf, ctx)?;
                self.string_data.push(string_data.clone());
                println!("TensorProto: Added string_data: {:?}", String::from_utf8_lossy(&string_data));
            }
            9 => {
                let mut float_data = 0f32;
                prost::encoding::float::merge(wire_type, &mut float_data, buf, ctx)?;
                self.float_data.push(float_data);
                println!("TensorProto: Added float_data: {}", float_data);
            }
            11 => {
                let mut int64_data = 0i64;
                prost::encoding::int64::merge(wire_type, &mut int64_data, buf, ctx)?;
                self.int64_data.push(int64_data);
                println!("TensorProto: Added int64_data: {}", int64_data);
            }
            _ => {
                println!("TensorProto: Skipping unknown tag: {}", tag);
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        unimplemented!("Encoded length not needed")
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
                println!("ValueInfoProto: Set name: {}", self.name);
            }
            2 => {
                let mut type_proto = TypeProto::default();
                prost::encoding::message::merge(wire_type, &mut type_proto, buf, ctx)?;
                println!("ValueInfoProto: Added type");
                self.type_proto = Some(type_proto);
            }
            _ => {
                println!("ValueInfoProto: Skipping unknown tag: {}", tag);
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
                println!("ModelProto: Parsed ir_version: {}", ir_version);
            }
            2 => {
                prost::encoding::string::merge(wire_type, &mut self.producer_name, buf, ctx)?;
                println!("ModelProto: Set producer_name: {}", self.producer_name);
            }
            3 => {
                prost::encoding::string::merge(wire_type, &mut self.producer_version, buf, ctx)?;
                println!("ModelProto: Set producer_version: {}", self.producer_version);
            }
            7 => {
                let mut graph = GraphProto::default();
                prost::encoding::message::merge(wire_type, &mut graph, buf, ctx)?;
                self.graph = Some(graph);
                println!("ModelProto: Parsed graph");
            }
            8 => {
                let mut opset = OpSetImport::default();
                prost::encoding::message::merge(wire_type, &mut opset, buf, ctx)?;
                self.opset_import.push(opset);
                println!("ModelProto: Added opset_import (domain: {}, version: {})", self.opset_import.last().unwrap().domain, self.opset_import.last().unwrap().version);
            }
            _ => {
                println!("ModelProto: Skipping unknown tag: {}", tag);
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
                println!(
                    "GraphProto: Parsed Node {} (OpType: {})",
                    node_index,
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
                self.node.push(node);
            }
            2 => {
                prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?;
                println!("GraphProto: Set name: {}", self.name);
            }
            // 5 => {
            //     prost::encoding::string::merge(wire_type, &mut self.doc_string, buf, ctx)?;
            //     println!("GraphProto: Set doc_string: {}", self.doc_string);
            // }
            8 => {
                let mut init = TensorProto::default();
                prost::encoding::message::merge(wire_type, &mut init, buf, ctx)?;
                println!("GraphProto: Added initializer (name: {})", init.name);
                self.initializer.push(init);
            }
            11 => {
                let mut output = ValueInfoProto::default();
                prost::encoding::message::merge(wire_type, &mut output, buf, ctx)?;
                println!("GraphProto: Added output (name: {})", output.name);
                self.output.push(output);
            }
            12 => {
                let mut input = ValueInfoProto::default();
                prost::encoding::message::merge(wire_type, &mut input, buf, ctx)?;
                println!("GraphProto: Added input (name: {})", input.name);
                self.input.push(input);
            }
            13 => {
                let mut value_info = ValueInfoProto::default();
                prost::encoding::message::merge(wire_type, &mut value_info, buf, ctx)?;
                println!("GraphProto: Added value_info (name: {})", value_info.name);
                self.value_info.push(value_info);
            }
            _ => {
                println!("GraphProto: Skipping unknown tag: {}", tag);
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
                println!("NodeProto: Added input: {}", input);
            }
            2 => {
                let mut output = String::new();
                prost::encoding::string::merge(wire_type, &mut output, buf, ctx)?;
                self.output.push(output.clone());
                println!("NodeProto: Added output: {}", output);
            }
            // 3 => {
            //     prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?;
            //     println!("NodeProto: Set name: {}", self.name);
            // }
            4 => {
                prost::encoding::string::merge(wire_type, &mut self.op_type, buf, ctx)?;
                println!("NodeProto: Set op_type: {}", self.op_type);
            }
            // 5 => {
            //     prost::encoding::string::merge(wire_type, &mut self.domain, buf, ctx)?;
            //     println!("NodeProto: Set domain: {}", self.domain);
            // }
            7 => {
                let mut attr = AttributeProto::default();
                prost::encoding::message::merge(wire_type, &mut attr, buf, ctx)?;
                println!("NodeProto: Added attribute (name: {})", attr.name);
                self.attributes.push(attr);
            }
             8 => {
                // Handle subgraph attributes (e.g., then_branch, else_branch, body)
                let mut subgraph = GraphProto::default();
                prost::encoding::message::merge(wire_type, &mut subgraph, buf, ctx)?;
                println!("NodeProto: Added subgraph for attribute");
                self.subgraphs.insert(format!("subgraph_{}", self.subgraphs.len()), subgraph);
            }
            _ => {
                println!("NodeProto: Skipping unknown tag: {}", tag);
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
            // 1 => {
            //     prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?;
            //     println!("AttributeProto: Set name: {}", self.name);
            // }
            2 => {
                prost::encoding::int64::merge(wire_type, &mut self.i, buf, ctx)?;
                println!("AttributeProto: Set int: {}", self.i);
            }
            3 => {
                prost::encoding::float::merge(wire_type, &mut self.f, buf, ctx)?;
                println!("AttributeProto: Set float: {}", self.f);
            }
            4 => {
                prost::encoding::bytes::merge(wire_type, &mut self.s, buf, ctx)?;
                println!("AttributeProto: Set bytes: {:?}", String::from_utf8_lossy(&self.s));
            }
            5 => {
                let mut tensor = TensorProto::default();
                prost::encoding::message::merge(wire_type, &mut tensor, buf, ctx)?;
                println!("AttributeProto: Added tensor (name: {})", tensor.name);
                self.t = Some(tensor);
            }
            6 => {
                let mut i = 0i64;
                prost::encoding::int64::merge(wire_type, &mut i, buf, ctx)?;
                self.ints.push(i);
                println!("AttributeProto: Added int to ints: {}", i);
            }
            7 => {
                let mut f = 0.0f32;
                prost::encoding::float::merge(wire_type, &mut f, buf, ctx)?;
                self.floats.push(f);
                println!("AttributeProto: Added float to floats: {}", f);
            }
             8 => {
                let mut subgraph = GraphProto::default();
                prost::encoding::message::merge(wire_type, &mut subgraph, buf, ctx)?;
                println!("AttributeProto: Added subgraph");
                self.g = Some(subgraph);
            }
            _ => {
                println!("AttributeProto: Skipping unknown tag: {}", tag);
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
        println!("Parsing TypeProto (tag: {})", tag);
        match tag {
            1 => {
                let mut tensor_type = TensorTypeProto::default();
                prost::encoding::message::merge(wire_type, &mut tensor_type, buf, ctx)?;
                println!("TypeProto: Added tensor_type");
                self.tensor_type = Some(tensor_type);
            }
            _ => {
                println!("TypeProto: Skipping unknown tag: {}", tag);
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
        println!("Parsing TensorTypeProto (tag: {})", tag);
        match tag {
            1 => {
                prost::encoding::int32::merge(wire_type, &mut self.elem_type, buf, ctx)?;
                println!("TensorTypeProto: Set elem_type: {}", self.elem_type);
            }
            2 => {
                let mut shape = TensorShapeProto::default();
                prost::encoding::message::merge(wire_type, &mut shape, buf, ctx)?;
                println!("TensorTypeProto: Added shape");
                self.shape = Some(shape);
            }
            _ => {
                println!("TensorTypeProto: Skipping unknown tag: {}", tag);
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
        println!("Parsing TensorShapeProto (tag: {})", tag);
        match tag {
            1 => {
                let mut dim = TensorDimension::default();
                prost::encoding::message::merge(wire_type, &mut dim, buf, ctx)?;
                println!("TensorShapeProto: Added dimension (dim_value: {})", dim.dim_value);
                self.dim.push(dim);
            }
            _ => {
                println!("TensorShapeProto: Skipping unknown tag: {}", tag);
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
        println!("Parsing Dimension (tag: {})", tag);
        match tag {
            1 => {
                prost::encoding::int64::merge(wire_type, &mut self.dim_value, buf, ctx)?;
                println!("Dimension: Set dim_value: {}", self.dim_value);
            }
            _ => {
                println!("Dimension: Skipping unknown tag: {}", tag);
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
fn op_resize(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let input = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Resize requires input tensor"))?)?;
    let scales = inputs.get(1).map(|s| ort_to_ndarray(s)).transpose()?;
    let sizes = inputs.get(2).map(|s| match s {
        OrtValue::Tensor { dtype: DataType::Int64, data, .. } => {
            data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect::<Vec<_>>()
        }
        _ => vec![],
    }).unwrap_or_default();
    let mode = node.attributes.iter().find(|a| a.name == "mode")
        .map(|a| String::from_utf8_lossy(&a.s).to_string())
        .unwrap_or("nearest".to_string());

    let input_shape = input.shape();
    let output_shape = if !sizes.is_empty() {
        sizes
    } else if let Some(scales) = scales {
        input_shape.iter().zip(scales.iter()).map(|(&dim, &scale)| (dim as f32 * scale).round() as usize).collect()
    } else {
        return Err(OrtError::InvalidTensorData("Resize requires scales or sizes".into()));
    };

    let mut result = ArrayD::zeros(output_shape.clone());

    if mode == "nearest" {
        for idx in ndarray::indices(&output_shape[..]) {
            let mut in_idx = Vec::new();
            for (i, &dim) in idx.slice().iter().enumerate() {
                let in_pos = (dim as f32) * (input_shape[i] as f32) / (output_shape[i] as f32);
                in_idx.push(in_pos.round() as usize);
            }
            result[idx.slice()] = input[&in_idx[..]];
        }
    } else {
        return Err(OrtError::UnsupportedOp("Resize only supports nearest mode".into()));
    }

    Ok(ndarray_to_ort(result, DataType::Float))
}
    fn op_stft(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let signal = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("STFT requires signal tensor"))?)?;
    let frame_length = inputs.get(1).map(|v| match v {
        OrtValue::Tensor { dtype: DataType::Int64, data, .. } => {
            Ok(i64::from_le_bytes(data[..8].try_into().unwrap()) as usize)
        }
        _ => return Err(OrtError::TypeMismatch("STFT requires Int64 frame_length")),
    }).ok_or_else(|| OrtError::InvalidTensorData("STFT requires frame_length tensor".into()))?;
    let frame_step = node.attributes.iter().find(|a| a.name == "frame_step")
        .map(|a| a.i as usize)
        .ok_or_else(|| OrtError::InvalidTensorData("STFT requires frame_step attribute".into()))?;
    let window = inputs.get(2).map(|w| ort_to_ndarray(w)).transpose()?;

    let signal_shape = signal.shape();
    if signal_shape.len() != 2 {
        return Err(OrtError::TypeMismatch("STFT requires 2D signal tensor [batch_size, signal_length]"));
    }
    let (batch_size, signal_length) = (signal_shape[0], signal_shape[1]);

    let n_fft = frame_length.unwrap();
    let num_frames = (signal_length - n_fft.clone()) / frame_step + 1;
    let n_freq = n_fft / 2 + 1; // Number of frequency bins
    let mut result = ArrayD::zeros(vec![batch_size, num_frames, n_fft, 2]); // Real and imaginary parts

    for b in 0..batch_size {
        for f in 0..num_frames {
            let start = f * frame_step;
            let end = start + n_fft.clone();
            if end > signal_length {
                continue;
            }
            let frame = signal.slice_axis(Axis(1), ndarray::Slice::from(start..end)).to_owned();
            let frame = if let Some(w) = &window {
                frame * w
            } else {
                frame
            };

            // Simple DFT implementation
            for k in 0..n_freq {
                let mut real = 0.0;
                let mut imag = 0.0;
                for n in 0..n_fft.clone() {
                    let angle = -2.0 * std::f32::consts::PI * (k as f32) * (n as f32) / (n_fft as f32);
                    real += frame[[b, n]] * angle.cos();
                    imag += frame[[b, n]] * angle.sin();
                }
                result[[b, f, k, 0]] = real;
                result[[b, f, k, 1]] = imag;
            }
        }
    }

    Ok(ndarray_to_ort(result, DataType::Float))
}
    fn op_lstm(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let x = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("LSTM requires input tensor"))?)?;
    let w = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("LSTM requires weight tensor"))?)?;
    let r = ort_to_ndarray(inputs.get(2).ok_or_else(|| OrtError::TypeMismatch("LSTM requires recurrence weight tensor"))?)?;
    let b = inputs.get(3).map(|b| ort_to_ndarray(b)).transpose()?;

    let direction = node.attributes.iter().find(|a| a.name == "direction")
        .map(|a| String::from_utf8_lossy(&a.s).to_string())
        .unwrap_or("forward".to_string());
    let hidden_size = node.attributes.iter().find(|a| a.name == "hidden_size")
        .map(|a| a.i as usize)
        .ok_or_else(|| OrtError::InvalidTensorData("LSTM requires hidden_size attribute".into()))?;

    let x_shape = x.shape();
    if x_shape.len() != 3 {
        return Err(OrtError::TypeMismatch("LSTM requires 3D input tensor [seq_length, batch_size, input_size]"));
    }
    let (seq_length, batch_size, input_size) = (x_shape[0], x_shape[1], x_shape[2]);

    let w_shape = w.shape();
    if w_shape.len() != 3 || w_shape[0] != 1 || w_shape[1] != 4 * hidden_size || w_shape[2] != input_size {
        return Err(OrtError::TypeMismatch("LSTM weight tensor must be [1, 4*hidden_size, input_size]"));
    }
    let r_shape = r.shape();
    if r_shape.len() != 3 || r_shape[0] != 1 || r_shape[1] != 4 * hidden_size || r_shape[2] != hidden_size {
        return Err(OrtError::TypeMismatch("LSTM recurrence weight tensor must be [1, 4*hidden_size, hidden_size]"));
    }

    let mut h_t = Array2::zeros((batch_size, hidden_size));
    let mut c_t = Array2::zeros((batch_size, hidden_size));
    let mut output = Array3::zeros((seq_length, batch_size, hidden_size));

    let forward = direction != "reverse";
    let range: Box<dyn Iterator<Item = usize>> = if forward {
        Box::new(0..seq_length)
    } else {
        Box::new((0..seq_length).rev())
    };

    for t in range {
        let x_t = x.slice_axis(Axis(0), ndarray::Slice::from(t..t + 1)).into_shape((batch_size, input_size)).unwrap();
        let gates: Array2<f32> = x_t.dot(&w.slice_axis(Axis(0), ndarray::Slice::from(0..1)).into_shape((4 * hidden_size, input_size)).unwrap().t())
    + h_t.dot(&r.slice_axis(Axis(0), ndarray::Slice::from(0..1)).into_shape((4 * hidden_size, hidden_size)).unwrap().t());
let gates = if let Some(bias) = &b {
    gates + bias.slice_axis(Axis(0), ndarray::Slice::from(0..4 * hidden_size)).into_shape((batch_size, 4 * hidden_size)).unwrap()
} else {
    gates.into_shape((batch_size, 4 * hidden_size)).unwrap()
};

        let (i, f, c, o) = (
            gates.slice_axis(Axis(1), ndarray::Slice::from(0..hidden_size)).mapv(|x| 1.0 / (1.0 + (-x).exp())),
            gates.slice_axis(Axis(1), ndarray::Slice::from(hidden_size..2 * hidden_size)).mapv(|x| 1.0 / (1.0 + (-x).exp())),
            gates.slice_axis(Axis(1), ndarray::Slice::from(2 * hidden_size..3 * hidden_size)).mapv(|x| x.tanh()),
            gates.slice_axis(Axis(1), ndarray::Slice::from(3 * hidden_size..4 * hidden_size)).mapv(|x| 1.0 / (1.0 + (-x).exp())),
        );

        c_t = f * &c_t + i * c;
        h_t = o * c_t.mapv(|x| x.tanh());
        output.slice_mut(ndarray::s![t, .., ..]).assign(&h_t);
    }

    Ok(ndarray_to_ort(output.into_dyn(), DataType::Float))
}
    fn op_conv_transpose(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let input = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("ConvTranspose requires input tensor"))?)?;
    let weight = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("ConvTranspose requires weight tensor"))?)?;
    let bias = inputs.get(2).map(|b| ort_to_ndarray(b)).transpose()?;

    let strides = node.attributes.iter().find(|a| a.name == "strides")
        .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
        .unwrap_or(vec![1, 1]);
    let pads = node.attributes.iter().find(|a| a.name == "pads")
        .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
        .unwrap_or(vec![0, 0, 0, 0]);
    let dilations = node.attributes.iter().find(|a| a.name == "dilations")
        .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
        .unwrap_or(vec![1, 1]);

    let input_shape = input.shape();
    let weight_shape = weight.shape();
    if input_shape.len() != 4 || weight_shape.len() != 4 {
        return Err(OrtError::TypeMismatch("ConvTranspose requires 4D input and weight tensors"));
    }
    let (n, c_in, h_in, w_in) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    let (c_in_w, c_out, k_h, k_w) = (weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]);

    if c_in != c_in_w {
        return Err(OrtError::TypeMismatch("ConvTranspose input and weight channels must match"));
    }

    // Compute output dimensions
    let h_out = (h_in - 1) * strides[0] + dilations[0] * (k_h - 1) + 1 - 2 * pads[0];
    let w_out = (w_in - 1) * strides[1] + dilations[1] * (k_w - 1) + 1 - 2 * pads[1];
    let mut result = ArrayD::zeros(vec![n, c_out, h_out, w_out]);

    // Perform transposed convolution
    for b in 0..n {
        for ic in 0..c_in {
            for ih in 0..h_in {
                for iw in 0..w_in {
                    for oc in 0..c_out {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let oh = ih * strides[0] + kh * dilations[0] - pads[0];
                                let ow = iw * strides[1] + kw * dilations[1] - pads[1];
                                if oh < h_out && ow < w_out {
                                    result[[b, oc, oh, ow]] += input[[b, ic, ih, iw]] * weight[[ic, oc, kh, kw]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias if provided
    if let Some(b) = bias {
        for b_idx in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        result[[b_idx, oc, oh, ow]] += b[[oc]];
                    }
                }
            }
        }
    }

    Ok(ndarray_to_ort(result, DataType::Float))
}
    fn op_conv(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let input = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Conv requires input tensor"))?)?;
    let weight = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Conv requires weight tensor"))?)?;
    let bias = inputs.get(2).map(|b| ort_to_ndarray(b)).transpose()?;

    let strides = node.attributes.iter().find(|a| a.name == "strides")
        .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
        .unwrap_or(vec![1, 1]);
    let pads = node.attributes.iter().find(|a| a.name == "pads")
        .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
        .unwrap_or(vec![0, 0, 0, 0]);
    let dilations = node.attributes.iter().find(|a| a.name == "dilations")
        .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
        .unwrap_or(vec![1, 1]);

    let input_shape = input.shape();
    let weight_shape = weight.shape();
    if input_shape.len() != 4 || weight_shape.len() != 4 {
        return Err(OrtError::TypeMismatch("Conv requires 4D input and weight tensors"));
    }
    let (n, c_in, h_in, w_in) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    let (c_out, c_in_w, k_h, k_w) = (weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]);

    if c_in != c_in_w {
        return Err(OrtError::TypeMismatch("Conv input and weight channels must match"));
    }

    // Compute output dimensions
    let h_out = (h_in + 2 * pads[0] - dilations[0] * (k_h - 1) - 1) / strides[0] + 1;
    let w_out = (w_in + 2 * pads[1] - dilations[1] * (k_w - 1) - 1) / strides[1] + 1;
    let mut result = ArrayD::zeros(vec![n, c_out, h_out, w_out]);

    // Perform convolution
    for b in 0..n {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0;
                    for ic in 0..c_in {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = oh * strides[0] + kh * dilations[0] - pads[0];
                                let iw = ow * strides[1] + kw * dilations[1] - pads[1];
                                if ih < h_in && iw < w_in {
                                    sum += input[[b, ic, ih, iw]] * weight[[oc, ic, kh, kw]];
                                }
                            }
                        }
                    }
                    result[[b, oc, oh, ow]] = sum;
                }
            }
        }
    }

    // Add bias if provided
    if let Some(b) = bias {
        for b_idx in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        result[[b_idx, oc, oh, ow]] += b[[oc]];
                    }
                }
            }
        }
    }

    Ok(ndarray_to_ort(result, DataType::Float))
}
    fn op_scatter_nd(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let data = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("ScatterND requires data tensor"))?)?;
    let indices = match inputs.get(1) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, shape, .. }) => {
            let idx_shape = shape.iter().filter_map(|d| match d {
                Dimensions::Fixed(n) => Some(*n),
                _ => None,
            }).collect::<Vec<_>>();
            let idx_data: Vec<i64> = data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
            (idx_shape, idx_data)
        }
        _ => return Err(OrtError::TypeMismatch("ScatterND requires Int64 indices")),
    };
    let updates = ort_to_ndarray(inputs.get(2).ok_or_else(|| OrtError::TypeMismatch("ScatterND requires updates tensor"))?)?;

    let mut result = data.clone();
    let (idx_shape, idx_data) = indices;
    let idx_depth = idx_shape[idx_shape.len() - 1]; // Last dimension of indices gives the depth of each index
    let num_indices = idx_data.len() / idx_depth;

    // Iterate over indices and update result
    for i in 0..num_indices {
        let idx_start = i * idx_depth;
        let mut index = Vec::new();
        for j in 0..idx_depth {
            index.push(idx_data[idx_start + j] as usize);
        }
        let update_idx = i; // Assuming updates is 1D or matches index structure
        result[&index[..]] = updates[[update_idx]];
    }

    Ok(ndarray_to_ort(result, DataType::Float))
}
fn op_nonzero(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("NonZero requires one float tensor"))?)?;
    let shape = array.shape();
    let ndim = shape.len();

    // Collect indices of non-zero elements
    let mut indices: Vec<Vec<i64>> = Vec::new();
    for idx in ndarray::indices(&shape[..]) {
        let val = array[idx.slice()];
        if val != 0.0 {
            indices.push(idx.as_array_view().to_vec().into_iter().map(|x| x as i64).collect());
        }
    }

    // Transpose indices to [ndim, num_nonzero]
    let num_nonzero = indices.len();
    let mut data = Vec::with_capacity(ndim * num_nonzero * 8);
    for dim in 0..ndim {
        for idx in &indices {
            data.extend_from_slice(&idx[dim].to_le_bytes());
        }
    }

    Ok(OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(ndim), Dimensions::Fixed(num_nonzero)],
        dtype: DataType::Int64,
        data: Arc::new(data),
    })
}


    fn op_cumsum(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("CumSum requires one float tensor"))?)?;
    let axis_tensor = inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("CumSum requires axis tensor"))?;
    let axis = match axis_tensor {
        OrtValue::Tensor { dtype: DataType::Int64, data, .. } => {
            i64::from_le_bytes(data[..8].try_into().unwrap()) as usize
        }
        _ => return Err(OrtError::TypeMismatch("CumSum requires Int64 axis tensor")),
    };
    let exclusive = node.attributes.iter().find(|a| a.name == "exclusive").map(|a| a.i != 0).unwrap_or(false);
    let reverse = node.attributes.iter().find(|a| a.name == "reverse").map(|a| a.i != 0).unwrap_or(false);

    if axis >= array.ndim() {
        return Err(OrtError::InvalidTensorData("CumSum axis out of bounds".into()));
    }

    let mut result = ArrayD::zeros(array.shape());
    let shape = array.shape();

    // Iterate over all elements, computing cumulative sum along the specified axis
    for idx in ndarray::indices(&shape[..]) {
        let mut sum = 0.0;
        let mut indices: Vec<usize> = idx.as_array_view().to_vec();
        if reverse {
            // Reverse cumulative sum
            for i in (0..=indices[axis]).rev() {
                indices[axis] = i;
                if !exclusive || i < indices[axis] {
                    sum += array[&indices[..]];
                }
                result[&indices[..]] = sum;
            }
        } else {
            // Forward cumulative sum
            for i in 0..=indices[axis] {
                indices[axis] = i;
                if !exclusive || i < indices[axis] {
                    sum += array[&indices[..]];
                }
                result[&indices[..]] = sum;
            }
        }
    }

    Ok(ndarray_to_ort(result, DataType::Float))
}

    // Arithmetic Operations
    fn op_add(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let (shape1, data1, data2) = match (inputs.get(0), inputs.get(1)) {
            (
                Some(OrtValue::Tensor {
                    dtype: DataType::Float,
                    shape: shape1,
                    data: data1,
                    ..
                }),
                Some(OrtValue::Tensor {
                    dtype: DataType::Float,
                    shape: shape2,
                    data: data2,
                    ..
                }),
            ) if shape1 == shape2 => (shape1, data1, data2),
            _ => {
                return Err(OrtError::TypeMismatch(
                    "Add requires two float tensors with matching shapes",
                ))
            }
        };
        let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        let result = array1 + array2;
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_sub(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let (shape1, data1, data2) = match (inputs.get(0), inputs.get(1)) {
            (
                Some(OrtValue::Tensor {
                    dtype: DataType::Float,
                    shape: shape1,
                    data: data1,
                    ..
                }),
                Some(OrtValue::Tensor {
                    dtype: DataType::Float,
                    shape: shape2,
                    data: data2,
                    ..
                }),
            ) if shape1 == shape2 => (shape1, data1, data2),
            _ => {
                return Err(OrtError::TypeMismatch(
                    "Sub requires two float tensors with matching shapes",
                ))
            }
        };
        let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        let result = array1 - array2;
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_mul(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let (shape1, data1, data2) = match (inputs.get(0), inputs.get(1)) {
            (
                Some(OrtValue::Tensor {
                    dtype: DataType::Float,
                    shape: shape1,
                    data: data1,
                    ..
                }),
                Some(OrtValue::Tensor {
                    dtype: DataType::Float,
                    shape: shape2,
                    data: data2,
                    ..
                }),
            ) if shape1 == shape2 => (shape1, data1, data2),
            _ => {
                return Err(OrtError::TypeMismatch(
                    "Mul requires two float tensors with matching shapes",
                ))
            }
        };
        let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        let result = array1 * array2;
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_div(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let (shape1, data1, data2) = match (inputs.get(0), inputs.get(1)) {
            (
                Some(OrtValue::Tensor {
                    dtype: DataType::Float,
                    shape: shape1,
                    data: data1,
                    ..
                }),
                Some(OrtValue::Tensor {
                    dtype: DataType::Float,
                    shape: shape2,
                    data: data2,
                    ..
                }),
            ) if shape1 == shape2 => (shape1, data1, data2),
            _ => {
                return Err(OrtError::TypeMismatch(
                    "Div requires two float tensors with matching shapes",
                ))
            }
        };
        let array1 = ort_to_ndarray(inputs.get(0).unwrap())?;
        let array2 = ort_to_ndarray(inputs.get(1).unwrap())?;
        let result = array1 / array2;
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_pow(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Pow requires two inputs"))?)?;
    let array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Pow requires two inputs"))?)?;
    let result = if array2.len() == 1 {
        let b = array2.iter().next().ok_or_else(|| OrtError::InvalidTensorData("Pow requires non-empty exponent".into()))?;
        array1.mapv(|a| a.powf(*b))
    } else {
        ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| a.powf(b))
    };
    Ok(ndarray_to_ort(result, DataType::Float))
}

    fn op_sqrt(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Sqrt requires one float tensor"))?)?;
        let result = array.mapv(|a| a.sqrt());
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    // Element-wise Operations
    fn op_exp(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Exp requires one float tensor"))?)?;
        let result = array.mapv(|a| a.exp());
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_sigmoid(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Sigmoid requires one float tensor"))?)?;
        let result = array.mapv(|a| 1.0 / (1.0 + (-a).exp()));
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_tanh(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Tanh requires one float tensor"))?)?;
        let result = array.mapv(|a| a.tanh());
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_sin(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Sin requires one float tensor"))?)?;
        let result = array.mapv(|a| a.sin());
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_cos(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Cos requires one float tensor"))?)?;
        let result = array.mapv(|a| a.cos());
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_atan(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Atan requires one float tensor"))?)?;
        let result = array.mapv(|a| a.atan());
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_floor(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Floor requires one float tensor"))?)?;
        let result = array.mapv(|a| a.floor());
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_round(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Round requires one float tensor"))?)?;
        let result = array.mapv(|a| a.round());
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_leaky_relu(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let alpha = node.attributes.iter().find(|a| a.name == "alpha").map(|a| a.f).unwrap_or(0.01);
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("LeakyRelu requires one float tensor"))?)?;
        let result = array.mapv(|a| if a >= 0.0 { a } else { alpha * a });
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    // Comparison Operations
    fn op_less(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Less requires two float tensors"))?)?;
        let array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Less requires two float tensors"))?)?;
        let result = ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| if a < b { 1.0 } else { 0.0 });
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_greater(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Greater requires two float tensors"))?)?;
        let array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Greater requires two float tensors"))?)?;
        let result = ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| if a > b { 1.0 } else { 0.0 });
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_equal(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Equal requires two float tensors"))?)?;
        let array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Equal requires two float tensors"))?)?;
        let result = ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| if a == b { 1.0 } else { 0.0 });
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_greater_or_equal(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("GreaterOrEqual requires two float tensors"))?)?;
        let array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("GreaterOrEqual requires two float tensors"))?)?;
        let result = ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| if a >= b { 1.0 } else { 0.0 });
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_and(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("And requires two float tensors"))?)?;
        let array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("And requires two float tensors"))?)?;
        let result = ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 });
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    // Shape Manipulation Operations
    fn op_reshape(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let tensor = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Reshape requires tensor input"))?;
        let shape_tensor = inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Reshape requires shape input"))?;
        let (data, dtype, shape_data) = match (tensor, shape_tensor) {
            (
                OrtValue::Tensor { shape: _, dtype, data, .. },
                OrtValue::Tensor { dtype: DataType::Int64, shape: _, data: s_data, .. },
            ) => (data, dtype, s_data),
            _ => return Err(OrtError::TypeMismatch("Reshape requires tensor and Int64 shape inputs")),
        };
        let shape: Vec<usize> = shape_data
            .chunks(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect();
        Ok(OrtValue::Tensor {
            shape:vec![Dimensions::Fixed(shape.len())],
            dtype: *dtype,
            data: Arc::clone(data),
        })
    }

    fn op_squeeze(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let axes = node.attributes.iter().find(|a| a.name == "axes")
            .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
            .unwrap_or_default();
        let tensor = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Squeeze requires one tensor"))?;
        match tensor {
            OrtValue::Tensor { shape, dtype, data } => {
                let mut new_shape = shape.clone();
                for &axis in axes.iter().rev() {
                    if axis < shape.len() && shape[axis] == Dimensions::Fixed(1) {
                        new_shape.remove(axis);
                    }
                }
                Ok(OrtValue::Tensor {
                    shape: new_shape,
                    dtype: *dtype,
                    data: Arc::clone(data),
                })
            }
            _ => Err(OrtError::TypeMismatch("Squeeze requires a tensor")),
        }
    }

    fn op_unsqueeze(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let axes = node.attributes.iter().find(|a| a.name == "axes")
            .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
            .unwrap_or_default();
        let tensor = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Unsqueeze requires one tensor"))?;
        match tensor {
            OrtValue::Tensor { shape, dtype, data } => {
                let mut new_shape = shape.clone();
                for &axis in axes.iter() {
                    new_shape.insert(axis, Dimensions::Fixed(1));
                }
                Ok(OrtValue::Tensor {
                    shape: new_shape,
                    dtype: *dtype,
                    data: Arc::clone(data),
                })
            }
            _ => Err(OrtError::TypeMismatch("Unsqueeze requires a tensor")),
        }
    }

    fn op_transpose(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let perm = node.attributes.iter().find(|a| a.name == "perm")
            .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
            .unwrap_or_else(|| (0..inputs[0].shape().len()).collect());
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Transpose requires one float tensor"))?)?;
        let result = array.permuted_axes(perm);
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_concat(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let axis = node.attributes.iter().find(|a| a.name == "axis")
            .map(|a| a.i as usize)
            .unwrap_or(0);
        let tensors: Vec<ArrayD<f32>> = inputs.iter()
            .map(|input| ort_to_ndarray(input))
            .collect::<OrtResult<Vec<_>>>()?;
        let output_shape = {
            let mut shape = tensors[0].shape().to_vec();
            shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();
            shape
        };
        let views: Vec<_> = tensors.iter().map(|t| t.view()).collect();
        let result = ndarray::concatenate(Axis(axis), &views)
            .map_err(|_| OrtError::InvalidTensorData("Concat shape mismatch".into()))?;
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_shape(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let tensor = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Shape requires one tensor"))?;
        let shape = tensor.shape();
        let data: Vec<u8> = shape.iter().map(|s| s).flat_map(|s| match(s){
            Dimensions::Fixed(n) => n.to_le_bytes(),
            Dimensions::Symbolic(n) => todo!(),
        }).collect();
        Ok(OrtValue::Tensor {
            shape: vec![Dimensions::Fixed(shape.len())],
            dtype: DataType::Int64,
            data: Arc::new(data),
        })
    }

    fn op_constant_of_shape(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let value = node.attributes.iter().find(|a| a.name == "value")
            .and_then(|a| a.t.as_ref())
            .map(|t| t.float_data.get(0).copied().unwrap_or(0.0))
            .unwrap_or(0.0);
        let shape_tensor = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("ConstantOfShape requires shape tensor"))?;
        let shape = match shape_tensor {
            OrtValue::Tensor { dtype: DataType::Int64, data, .. } => {
                data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect::<Vec<_>>()
            }
            _ => return Err(OrtError::TypeMismatch("ConstantOfShape requires Int64 shape tensor")),
        };
        let total_elements = shape.iter().product::<usize>();
        let result = ArrayD::from_elem(shape, value);
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_expand(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Expand requires tensor and shape"))?)?;
        let shape_tensor = inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Expand requires shape tensor"))?;
        let new_shape = match shape_tensor {
            OrtValue::Tensor { dtype: DataType::Int64, data, .. } => {
                data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect::<Vec<_>>()
            }
            _ => return Err(OrtError::TypeMismatch("Expand requires Int64 shape tensor")),
        };
        let result = array.broadcast(new_shape)
            .ok_or_else(|| OrtError::InvalidTensorData("Invalid broadcast shape".into()))?
            .to_owned();
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    // Reduction Operations
    fn op_reduce_sum(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let axes = node.attributes.iter().find(|a| a.name == "axes")
            .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
            .unwrap_or_default();
        let keepdims = node.attributes.iter().find(|a| a.name == "keepdims")
            .map(|a| a.i != 0)
            .unwrap_or(true);
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("ReduceSum requires one float tensor"))?)?;
        let mut result = array;
        for &axis in axes.iter() {
            result = result.sum_axis(Axis(axis));
        }
        if keepdims && !axes.is_empty() {
            let mut new_shape = result.shape().to_vec();
            for &axis in axes.iter() {
                new_shape.insert(axis, 1);
            }
            result = result.into_shape(new_shape)
                .map_err(|_| OrtError::InvalidTensorData("ReduceSum shape mismatch".into()))?;
        }
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    fn op_reduce_mean(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let axes = node.attributes.iter().find(|a| a.name == "axes")
            .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
            .unwrap_or_default();
        let keepdims = node.attributes.iter().find(|a| a.name == "keepdims")
            .map(|a| a.i != 0)
            .unwrap_or(true);
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("ReduceMean requires one float tensor"))?)?;
        let mut result = array;
        for &axis in axes.iter() {
            result = result.mean_axis(Axis(axis))
                .ok_or_else(|| OrtError::InvalidTensorData("ReduceMean axis error".into()))?;
        }
        if keepdims && !axes.is_empty() {
            let mut new_shape = result.shape().to_vec();
            for &axis in axes.iter() {
                new_shape.insert(axis, 1);
            }
            result = result.into_shape(new_shape)
                .map_err(|_| OrtError::InvalidTensorData("ReduceMean shape mismatch".into()))?;
        }
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    // Matrix Operations
    
fn op_matmul(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("MatMul requires two float tensors"))?)?;
    let array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("MatMul requires two float tensors"))?)?;
    let shape1 = array1.shape();
    let shape2 = array2.shape();
    let ndim1 = shape1.len();
    let ndim2 = shape2.len();
    
    // Validate dimensions: 2D or 3D (batched 2D)
    if ndim1 < 2 || ndim2 < 2 || shape1[ndim1 - 1] != shape2[ndim2 - 2] {
        // let errtext: String=format!(
        //     "Invalid shapes for MatMul: {:?} × {:?}", shape1, shape2
        // );
        return Err(OrtError::TypeMismatch(&"invalid shape"));
    }
    if ndim1 > 3 || ndim2 > 3 {
        return Err(OrtError::TypeMismatch("MatMul supports only 2D or batched 2D tensors"));
    }

    let result = if ndim1 == 2 && ndim2 == 2 {
        // 2D matrix multiplication: [m, k] × [k, n] → [m, n]
        let a1 = array1
        .clone()
            .into_shape((shape1[0], shape1[1]))
            .map_err(|_| OrtError::TypeMismatch("Failed to reshape array1 to 2D"))?;
        let a2 = array2.clone()
            .into_shape((shape2[0], shape2[1]))
            .map_err(|_| OrtError::TypeMismatch("Failed to reshape array2 to 2D"))?;
        a1.dot(&a2).into_dyn()
    } else {
        // Batched 2D (3D): [batch, m, k] × [batch, k, n] → [batch, m, n]
        let batch1 = if ndim1 == 3 { shape1[0] } else { 1 };
        let batch2 = if ndim2 == 3 { shape2[0] } else { 1 };
        let batch = if batch1 == 1 {
            batch2
        } else if batch2 == 1 {
            batch1
        } else if batch1 == batch2 {
            batch1
        } else {
            // format!(
            //     "Incompatible batch dimensions: {} vs {}", batch1, batch2
            // ))
            return Err(OrtError::TypeMismatch(&"Incompatible batch dimensions"));
        };
        let m = shape1[ndim1 - 2];
        let k = shape1[ndim1 - 1];
        let n = shape2[ndim2 - 1];
        let a1 = array1
            .into_shape(if ndim1 == 3 { (batch1, m, k) } else { (1, m, k) })
            .map_err(|_| OrtError::TypeMismatch("Failed to reshape array1 to 3D"))?;
        let a2 = array2
            .into_shape(if ndim2 == 3 { (batch2, k, n) } else { (1, k, n) })
            .map_err(|_| OrtError::TypeMismatch("Failed to reshape array2 to 3D"))?;
        let mut result = Array3::zeros((batch, m, n));
        for b in 0..batch {
            let a1_slice = a1.slice_axis(Axis(0), ndarray::Slice::from(b..b + 1)).into_shape((m, k)).unwrap();
            let a2_slice = a2.slice_axis(Axis(0), ndarray::Slice::from(b..b + 1)).into_shape((k, n)).unwrap();
            result.slice_mut(ndarray::s![b, .., ..]).assign(&a1_slice.dot(&a2_slice));
        }
        result.into_dyn()
    };
    Ok(ndarray_to_ort(result, DataType::Float))
}

fn op_sequence_at(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // SequenceAt extracts a tensor from a sequence at a given position
    let sequence = match inputs.get(0) {
        Some(OrtValue::Sequence(seq)) => seq,
        _ => return Err(OrtError::TypeMismatch("SequenceAt requires a sequence as first input")),
    };
    
    let position = match inputs.get(1) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
            i64::from_le_bytes(data[..8].try_into().unwrap())
        },
        _ => return Err(OrtError::TypeMismatch("SequenceAt requires an Int64 position tensor as second input")),
    };
    
    let pos = if position < 0 {
        (sequence.len() as i64 + position) as usize
    } else {
        position as usize
    };
    
    if pos >= sequence.len() {
        return Err(OrtError::IndexError("Position out of bounds in SequenceAt"));
    }
    
    Ok(sequence[pos].clone())
}

fn op_sequence_construct(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // SequenceConstruct creates a sequence from input tensors
    if inputs.is_empty() {
        return Err(OrtError::InvalidTensorData("SequenceConstruct requires at least one input tensor".into()));
    }
    
    Ok(OrtValue::Sequence(inputs.to_vec()))
}

fn op_sequence_empty(node: &NodeProto, _inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // SequenceEmpty creates an empty sequence
    // Optional attribute 'dtype' specifies the element type (not used in this implementation)
    let _dtype = node.attributes.iter()
        .find(|a| a.name == "dtype")
        .map(|a| a.i)
        .unwrap_or(1); // Default to float
        
    Ok(OrtValue::Sequence(Vec::new()))
}

fn op_sequence_erase(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // SequenceErase removes a tensor from a sequence at a given position
    let sequence = match inputs.get(0) {
        Some(OrtValue::Sequence(seq)) => seq.clone(),
        _ => return Err(OrtError::TypeMismatch("SequenceErase requires a sequence as first input")),
    };
    
    let position = match inputs.get(1) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
            i64::from_le_bytes(data[..8].try_into().unwrap())
        },
        _ => return Err(OrtError::TypeMismatch("SequenceErase requires an Int64 position tensor as second input")),
    };
    
    let pos = if position < 0 {
        (sequence.len() as i64 + position) as usize
    } else {
        position as usize
    };
    
    if pos >= sequence.len() {
        return Err(OrtError::IndexError("Position out of bounds in SequenceErase"));
    }
    
    let mut result = sequence.clone();
    result.remove(pos);
    
    Ok(OrtValue::Sequence(result))
}

fn op_sequence_insert(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // SequenceInsert inserts a tensor into a sequence at a given position
    let sequence = match inputs.get(0) {
        Some(OrtValue::Sequence(seq)) => seq.clone(),
        _ => return Err(OrtError::TypeMismatch("SequenceInsert requires a sequence as first input")),
    };
    
    let tensor = match inputs.get(1) {
        Some(tensor) => tensor.clone(),
        _ => return Err(OrtError::TypeMismatch("SequenceInsert requires a tensor as second input")),
    };
    
    let position = if inputs.len() > 2 {
        match inputs.get(2) {
            Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
                i64::from_le_bytes(data[..8].try_into().unwrap())
            },
            _ => return Err(OrtError::TypeMismatch("SequenceInsert requires an Int64 position tensor as third input")),
        }
    } else {
        sequence.len() as i64 // Default to append
    };
    
    let pos = if position < 0 {
        (sequence.len() as i64 + position) as usize
    } else {
        position as usize
    };
    
    if pos > sequence.len() {
        return Err(OrtError::IndexError("Position out of bounds in SequenceInsert"));
    }
    
    let mut result = sequence.clone();
    result.insert(pos, tensor);
    
    Ok(OrtValue::Sequence(result))
}

fn op_sequence_length(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // SequenceLength returns the length of a sequence
    let sequence = match inputs.get(0) {
        Some(OrtValue::Sequence(seq)) => seq,
        _ => return Err(OrtError::TypeMismatch("SequenceLength requires a sequence as input")),
    };
    
    let length = sequence.len() as i64;
    let data = length.to_le_bytes().to_vec();
    
    Ok(OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1)],
        dtype: DataType::Int64,
        data: Arc::new(data),
    })
}

// Map Operators Implementation
fn op_map_from_tensor(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // MapFromTensor creates a map from key and value tensors
    let keys = match inputs.get(0) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
            data.chunks(8).map(|c| MapKey::Int64(i64::from_le_bytes(c.try_into().unwrap()))).collect::<Vec<_>>()
        },
        Some(OrtValue::Tensor { dtype: DataType::String, data, .. }) => {
            // Simple implementation - assumes each 8 bytes is a string length followed by string data
            let mut keys = Vec::new();
            let mut i = 0;
            while i < data.len() {
                let len = u64::from_le_bytes(data[i..i+8].try_into().unwrap()) as usize;
                i += 8;
                if i + len > data.len() {
                    break;
                }
                let s = String::from_utf8_lossy(&data[i..i+len]).to_string();
                keys.push(MapKey::String(s));
                i += len;
            }
            keys
        },
        _ => return Err(OrtError::TypeMismatch("MapFromTensor requires keys tensor as first input")),
    };
    
    let values = match inputs.get(1) {
        Some(value_tensor @ OrtValue::Tensor { .. }) => {
            // For simplicity, we'll assume the values tensor can be split evenly among the keys
            let shape = value_tensor.shape();
            if shape.is_empty() {
                return Err(OrtError::InvalidTensorData("Values tensor must have at least one dimension".into()));
            }
            
            // Extract values based on the first dimension
            let mut values = Vec::new();
            // This is a simplified implementation - in a real implementation, we would need to split the tensor properly
            values.push(value_tensor.clone());
            values
        },
        _ => return Err(OrtError::TypeMismatch("MapFromTensor requires values tensor as second input")),
    };
    
    if keys.len() != values.len() {
        return Err(OrtError::InvalidTensorData(format!(
            "Number of keys ({}) must match number of values ({})",
            keys.len(), values.len()
        )));
    }
    
    let mut map = IndexMap::new();
    for (key, value) in keys.into_iter().zip(values.into_iter()) {
        map.insert(key, value);
    }
    
    Ok(OrtValue::Map(map))
}

fn op_map_to_tensor(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // MapToTensor converts a map to key and value tensors
    let map = match inputs.get(0) {
        Some(OrtValue::Map(map)) => map,
        _ => return Err(OrtError::TypeMismatch("MapToTensor requires a map as input")),
    };
    
    // Extract keys
    let keys: Vec<_> = map.keys().collect();
    
    // Create key tensor
    let key_tensor = match keys.first() {
        Some(MapKey::Int64(_)) => {
            let data: Vec<u8> = keys.iter().flat_map(|k| {
                if let MapKey::Int64(i) = k {
                    i.to_le_bytes().to_vec()
                } else {
                    vec![] // Should not happen if all keys are the same type
                }
            }).collect();
            
            OrtValue::Tensor {
                shape: vec![Dimensions::Fixed(keys.len())],
                dtype: DataType::Int64,
                data: Arc::new(data),
            }
        },
        Some(MapKey::String(_)) => {
            // Simple implementation - concatenate all strings with their lengths
            let mut data = Vec::new();
            for k in keys.clone() {
                if let MapKey::String(s) = k {
                    let bytes = s.as_bytes();
                    let len = bytes.len() as u64;
                    data.extend_from_slice(&len.to_le_bytes());
                    data.extend_from_slice(bytes);
                }
            }
            
            OrtValue::Tensor {
                shape: vec![Dimensions::Fixed(keys.len())],
                dtype: DataType::String,
                data: Arc::new(data),
            }
        },
        None => {
            // Empty map
            OrtValue::Tensor {
                shape: vec![Dimensions::Fixed(0)],
                dtype: DataType::Int64, // Default
                data: Arc::new(Vec::new()),
            }
        },
    };
    
    // Create value tensor - simplified implementation
    // In a real implementation, we would need to combine all values into a single tensor
    let value_tensor = if let Some(first_value) = map.values().next() {
        first_value.clone()
    } else {
        // Empty map
        OrtValue::Tensor {
            shape: vec![Dimensions::Fixed(0)],
            dtype: DataType::Float, // Default
            data: Arc::new(Vec::new()),
        }
    };
    
    // Return both tensors as a sequence
    Ok(OrtValue::Sequence(vec![key_tensor, value_tensor]))
}

fn op_map_get(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // MapGet retrieves a value from a map by key
    let map = match inputs.get(0) {
        Some(OrtValue::Map(map)) => map,
        _ => return Err(OrtError::TypeMismatch("MapGet requires a map as first input")),
    };
    
    let key = match inputs.get(1) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
            MapKey::Int64(i64::from_le_bytes(data[..8].try_into().unwrap()))
        },
        Some(OrtValue::Tensor { dtype: DataType::String, data, .. }) => {
            // Simple implementation - assumes the tensor contains a single string
            MapKey::String(String::from_utf8_lossy(data).to_string())
        },
        _ => return Err(OrtError::TypeMismatch("MapGet requires a key tensor as second input")),
    };
    
    match map.get(&key) {
        Some(value) => Ok(value.clone()),
        None => {
            // Return default value if provided, otherwise error
            if inputs.len() > 2 {
                Ok(inputs[2].clone())
            } else {
                Err(OrtError::IndexError("Key not found in map"))
            }
        }
    }
}

fn op_map_has_key(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // MapHasKey checks if a key exists in a map
    let map = match inputs.get(0) {
        Some(OrtValue::Map(map)) => map,
        _ => return Err(OrtError::TypeMismatch("MapHasKey requires a map as first input")),
    };
    
    let key = match inputs.get(1) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
            MapKey::Int64(i64::from_le_bytes(data[..8].try_into().unwrap()))
        },
        Some(OrtValue::Tensor { dtype: DataType::String, data, .. }) => {
            // Simple implementation - assumes the tensor contains a single string
            MapKey::String(String::from_utf8_lossy(data).to_string())
        },
        _ => return Err(OrtError::TypeMismatch("MapHasKey requires a key tensor as second input")),
    };
    
    let has_key = map.contains_key(&key) as i64;
    let data = has_key.to_le_bytes().to_vec();
    
    Ok(OrtValue::Tensor {
        shape: vec![Dimensions::Fixed(1)],
        dtype: DataType::Int64,
        data: Arc::new(data),
    })
}

fn op_map_keys(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // MapKeys returns all keys in a map
    let map = match inputs.get(0) {
        Some(OrtValue::Map(map)) => map,
        _ => return Err(OrtError::TypeMismatch("MapKeys requires a map as input")),
    };
    
    // Extract keys
    let keys: Vec<_> = map.keys().collect();
    
    // Create key tensor
    let key_tensor = match keys.first() {
        Some(MapKey::Int64(_)) => {
            let data: Vec<u8> = keys.iter().flat_map(|k| {
                if let MapKey::Int64(i) = k {
                    i.to_le_bytes().to_vec()
                } else {
                    vec![] // Should not happen if all keys are the same type
                }
            }).collect();
            
            OrtValue::Tensor {
                shape: vec![Dimensions::Fixed(keys.len())],
                dtype: DataType::Int64,
                data: Arc::new(data),
            }
        },
        Some(MapKey::String(_)) => {
            // Simple implementation - concatenate all strings with their lengths
            let mut data = Vec::new();
            for k in keys.clone() {
                if let MapKey::String(s) = k {
                    let bytes = s.as_bytes();
                    let len = bytes.len() as u64;
                    data.extend_from_slice(&len.to_le_bytes());
                    data.extend_from_slice(bytes);
                }
            }
            
            OrtValue::Tensor {
                shape: vec![Dimensions::Fixed(keys.len())],
                dtype: DataType::String,
                data: Arc::new(data),
            }
        },
        None => {
            // Empty map
            OrtValue::Tensor {
                shape: vec![Dimensions::Fixed(0)],
                dtype: DataType::Int64, // Default
                data: Arc::new(Vec::new()),
            }
        },
    };
    
    Ok(key_tensor)
} 
// BERT-specific operations
fn op_erf(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // Erf computes the error function of the input tensor
    let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Erf requires one float tensor"))?)?;
    
    // Real implementation of the error function using Abramowitz and Stegun approximation
    let result = array.mapv(|x| {
        // Constants for approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        
        // Save the sign of x
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        
        // A&S formula 7.1.26
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        sign * y
    });
    
    Ok(ndarray_to_ort(result, DataType::Float))
}

fn op_gelu(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // GELU (Gaussian Error Linear Unit) activation function
    let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("GELU requires one float tensor"))?)?;
    
    // Real implementation of GELU using the formula: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    // or the approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let result = array.mapv(|x| {
        // Using the approximation formula which is faster and still accurate
        let sqrt_2_over_pi = 0.7978845608; // sqrt(2/π)
        let coef = 0.044715;
        
        let inner = sqrt_2_over_pi * (x + coef * x.powi(3));
        0.5 * x * (1.0 + inner.tanh())
        
        // Alternative implementation using erf:
        // let a1 = 0.254829592;
        // let a2 = -0.284496736;
        // let a3 = 1.421413741;
        // let a4 = -1.453152027;
        // let a5 = 1.061405429;
        // let p = 0.3275911;
        // 
        // // Calculate x / sqrt(2)
        // let x_scaled = x / 1.4142135623730951;
        // 
        // // Save the sign of x_scaled
        // let sign = if x_scaled < 0.0 { -1.0 } else { 1.0 };
        // let x_abs = x_scaled.abs();
        // 
        // // A&S formula 7.1.26 for erf
        // let t = 1.0 / (1.0 + p * x_abs);
        // let erf = sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp());
        // 
        // // GELU formula
        // 0.5 * x * (1.0 + erf)
    });
    
    Ok(ndarray_to_ort(result, DataType::Float))
}

fn op_split(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // Split divides a tensor into multiple parts along a specified axis
    let input = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Split requires input tensor"))?)?;
    
    // Get axis attribute
    let axis = node.attributes.iter().find(|a| a.name == "axis")
        .map(|a| a.i as usize)
        .unwrap_or(0);
    
    // Get split attribute (sizes of each output)
    let split = if inputs.len() > 1 {
        // Split sizes provided as input tensor
        match inputs.get(1) {
            Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
                data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect::<Vec<_>>()
            }
            _ => return Err(OrtError::TypeMismatch("Split requires Int64 split tensor as second input")),
        }
    } else {
        // Split sizes provided as attribute
        node.attributes.iter().find(|a| a.name == "split")
            .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
            .unwrap_or_else(|| {
                // If not provided, split equally
                let dim_size = input.shape()[axis];
                let num_outputs = node.output.len();
                let size = dim_size / num_outputs;
                vec![size; num_outputs]
            })
    };
    
    // Create a sequence of output tensors
    let mut outputs = Vec::new();
    let mut start_idx = 0;
    
    for &size in &split {
        // Create slice for this split
        let mut indices = Vec::new();
        for dim in 0..input.ndim() {
            if dim == axis {
                indices.push(ndarray::SliceInfoElem::Slice {
                    start: start_idx as isize,
                    end: Some((start_idx + size) as isize),
                    step: 1,
                });
                start_idx += size;
            } else {
                indices.push(ndarray::SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                });
            }
        }
        
        // Extract the slice
        let slice = input.slice(&indices[..]);
        let output = ndarray_to_ort(slice.to_owned(), DataType::Float);
        outputs.push(output);
    }
    
    Ok(OrtValue::Sequence(outputs))
}

fn op_dropout(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // Dropout randomly zeroes elements of the input tensor with probability p
    // In inference mode, dropout is a no-op (just returns the input)
    let input = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Dropout requires input tensor"))?)?;
    
    // Get ratio attribute (probability of dropping)
    let _ratio = if inputs.len() > 1 {
        match inputs.get(1) {
            Some(OrtValue::Tensor { dtype: DataType::Float, data, .. }) => {
                f32::from_le_bytes(data[..4].try_into().unwrap())
            }
            _ => 0.5, // Default ratio
        }
    } else {
        node.attributes.iter().find(|a| a.name == "ratio")
            .map(|a| a.f)
            .unwrap_or(0.5)
    };
    
    // In inference mode, dropout is a no-op
    // Return the input tensor and a mask of ones
    let mask = ArrayD::ones(input.shape());
    
    // Return both the input and the mask as a sequence
    let output_tensor = ndarray_to_ort(input, DataType::Float);
    let mask_tensor = ndarray_to_ort(mask, DataType::Float);
    
    Ok(OrtValue::Sequence(vec![output_tensor, mask_tensor]))
}

fn op_einsum(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // Einsum performs tensor contractions according to the Einstein summation convention
    // Get equation attribute
    let equation = node.attributes.iter().find(|a| a.name == "equation")
        .map(|a| String::from_utf8_lossy(&a.s).to_string())
        .ok_or_else(|| OrtError::InvalidTensorData("Einsum requires equation attribute".into()))?;
    
    // Parse equation
    let parts: Vec<&str> = equation.split("->").collect();
    if parts.len() != 2 {
        return Err(OrtError::InvalidTensorData("Invalid Einsum equation format".into()));
    }
    
    let input_subscripts: Vec<&str> = parts[0].split(',').collect();
    let output_subscript = parts[1].trim();
    
    // For now, implement only the most common case in BERT: batch matrix multiplication
    // Example: "abc,acd->abd" (batched matrix multiplication)
    if input_subscripts.len() == 2 && 
       input_subscripts[0].len() == 3 && 
       input_subscripts[1].len() == 3 && 
       output_subscript.len() == 3 {
        
        let a = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Einsum requires input tensors"))?)?;
        let b = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Einsum requires input tensors"))?)?;
        
        // Check if this is a batched matrix multiplication pattern
        if input_subscripts[0].chars().nth(0) == input_subscripts[1].chars().nth(0) && 
           input_subscripts[0].chars().nth(2) == input_subscripts[1].chars().nth(1) && 
           output_subscript.chars().nth(0) == input_subscripts[0].chars().nth(0) && 
           output_subscript.chars().nth(1) == input_subscripts[0].chars().nth(1) && 
           output_subscript.chars().nth(2) == input_subscripts[1].chars().nth(2) {
            
            // This is batched matrix multiplication: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
            let batch = a.shape()[0];
            let m = a.shape()[1];
            let k = a.shape()[2];
            let n = b.shape()[2];
            
            let mut result = Array3::zeros((batch, m, n));
            
            for b_idx in 0..batch {
                let a_slice = a.slice(ndarray::s![b_idx, .., ..]);
                let b_slice = b.slice(ndarray::s![b_idx, .., ..]);
                let mut res_slice = result.slice_mut(ndarray::s![b_idx, .., ..]);
                res_slice.assign(&a_slice.dot(&b_slice));
            }
            
            return Ok(ndarray_to_ort(result.into_dyn(), DataType::Float));
        }
    }
    
    // Fallback for unsupported patterns
    Err(OrtError::UnsupportedOp(format!("Unsupported Einsum equation: {}", equation)))
}

fn op_topk(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // TopK finds the k largest or smallest elements along a specified axis
    let input = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("TopK requires input tensor"))?)?;
    
    // Get k value
    let k = if inputs.len() > 1 {
        match inputs.get(1) {
            Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
                i64::from_le_bytes(data[..8].try_into().unwrap()) as usize
            }
            _ => return Err(OrtError::TypeMismatch("TopK requires Int64 k tensor as second input")),
        }
    } else {
        node.attributes.iter().find(|a| a.name == "k")
            .map(|a| a.i as usize)
            .ok_or_else(|| OrtError::InvalidTensorData("TopK requires k attribute or input".into()))?
    };
    
    // Get axis attribute
    let axis = node.attributes.iter().find(|a| a.name == "axis")
        .map(|a| a.i as usize)
        .unwrap_or(input.ndim() - 1); // Default to last axis
    
    // Get largest attribute
    let largest = node.attributes.iter().find(|a| a.name == "largest")
        .map(|a| a.i != 0)
        .unwrap_or(true); // Default to true
    
    // Get sorted attribute
    let sorted = node.attributes.iter().find(|a| a.name == "sorted")
        .map(|a| a.i != 0)
        .unwrap_or(true); // Default to true
    
    // Create output arrays for values and indices
    let mut shape = input.shape().to_vec();
    shape[axis] = k;
    let mut values = ArrayD::zeros(shape.clone());
    let mut indices = ArrayD::zeros(shape.clone());
    
    // Process each slice along the specified axis
    let axis_size = input.shape()[axis];
    
    // Simple implementation for 2D case
    if input.ndim() == 2 {
        if axis == 1 {
            // Process each row
            for i in 0..input.shape()[0] {
                let row = input.slice(ndarray::s![i, ..]);
                let mut pairs: Vec<(usize, f32)> = row.iter()
                    .enumerate()
                    .map(|(j, &val)| (j, val))
                    .collect();
                
                // Sort by value
                if largest {
                    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }
                
                // Take top k
                pairs.truncate(k);
                
                // If sorted is false, restore original order
                if !sorted {
                    pairs.sort_by_key(|&(idx, _)| idx);
                }
                
                // Fill output arrays
                for (j, (idx, val)) in pairs.iter().enumerate() {
                    values[[i, j]] = *val;
                    indices[[i, j]] = *idx as f32;
                }
            }
        } else {
            // Process each column
            for j in 0..input.shape()[1] {
                let col = input.slice(ndarray::s![.., j]);
                let mut pairs: Vec<(usize, f32)> = col.iter()
                    .enumerate()
                    .map(|(i, &val)| (i, val))
                    .collect();
                
                // Sort by value
                if largest {
                    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }
                
                // Take top k
                pairs.truncate(k);
                
                // If sorted is false, restore original order
                if !sorted {
                    pairs.sort_by_key(|&(idx, _)| idx);
                }
                
                // Fill output arrays
                for (i, (idx, val)) in pairs.iter().enumerate() {
                    values[[i, j]] = *val;
                    indices[[i, j]] = *idx as f32;
                }
            }
        }
    } else {
        // For higher dimensions, we'd need a more complex implementation
        return Err(OrtError::UnsupportedOp("TopK for tensors with more than 2 dimensions is not implemented".into()));
    }
    
    // Convert indices to Int64
    let indices_data: Vec<u8> = indices.iter()
        .map(|&idx| (idx as i64).to_le_bytes())
        .flatten()
        .collect();
    
    let indices_tensor = OrtValue::Tensor {
        shape: shape.iter().map(|&d| Dimensions::Fixed(d)).collect(),
        dtype: DataType::Int64,
        data: Arc::new(indices_data),
    };
    
    // Return both values and indices as a sequence
    Ok(OrtValue::Sequence(vec![
        ndarray_to_ort(values, DataType::Float),
        indices_tensor,
    ]))
}

fn op_gather_elements(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // GatherElements gathers elements from a tensor at specified indices
    let data = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("GatherElements requires data tensor"))?)?;
    
    let indices = match inputs.get(1) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data: idx_data, shape: idx_shape, .. }) => {
            // Convert indices to a usable format
            let concrete_shape: Vec<usize> = idx_shape.iter().map(|d| match d {
                Dimensions::Fixed(n) => Ok(*n),
                Dimensions::Symbolic(_) => return Err(OrtError::InvalidTensorData("Cannot use symbolic shape for indices".into())),
            }).collect::<Result<_, _>>()?;
            
            let indices_vec: Vec<i64> = idx_data.chunks(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            
            ArrayD::from_shape_vec(concrete_shape, indices_vec)
                .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for indices".into()))?
        }
        _ => return Err(OrtError::TypeMismatch("GatherElements requires Int64 indices tensor")),
    };
    
    // Get axis attribute
    let axis = node.attributes.iter().find(|a| a.name == "axis")
        .map(|a| a.i as usize)
        .unwrap_or(0); // Default to first axis
    
    // Create output tensor with same shape as indices
    let mut result = ArrayD::zeros(indices.shape());
    
    // Gather elements
    for idx in ndarray::indices(indices.shape()) {
        let mut data_idx = idx.slice().to_vec();
        let index = indices[idx.slice()];
        
        // Handle negative indices
        let axis_size = data.shape()[axis];
        let normalized_index = if index < 0 {
            (axis_size as i64 + index) as usize
        } else {
            index as usize
        };
        
        // Replace the axis index with the gathered index
        data_idx[axis] = normalized_index;
        
        // Get the value from data tensor
        result[idx.slice()] = data[&data_idx[..]];
    }
    
    Ok(ndarray_to_ort(result, DataType::Float))
}

fn op_gather_nd(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // GatherND gathers slices from a tensor at specified indices
    let data = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("GatherND requires data tensor"))?)?;
    
    let indices = match inputs.get(1) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data: idx_data, shape: idx_shape, .. }) => {
            // Convert indices to a usable format
            let concrete_shape: Vec<usize> = idx_shape.iter().map(|d| match d {
                Dimensions::Fixed(n) => Ok(*n),
                Dimensions::Symbolic(_) => return Err(OrtError::InvalidTensorData("Cannot use symbolic shape for indices".into())),
            }).collect::<Result<_, _>>()?;
            
            let indices_vec: Vec<i64> = idx_data.chunks(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            
            ArrayD::from_shape_vec(concrete_shape, indices_vec)
                .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for indices".into()))?
        }
        _ => return Err(OrtError::TypeMismatch("GatherND requires Int64 indices tensor")),
    };
    
    // Get batch_dims attribute
    let batch_dims = node.attributes.iter().find(|a| a.name == "batch_dims")
        .map(|a| a.i as usize)
        .unwrap_or(0); // Default to 0
    
    // Calculate output shape
    let indices_shape = indices.shape();
    let data_shape = data.shape();
    
    let indices_rank = indices_shape.len();
    let last_dim = indices_shape[indices_rank - 1];
    
    // Output shape is indices.shape[:-1] + data.shape[indices.shape[-1]:]
    let mut output_shape = Vec::new();
    output_shape.extend_from_slice(&indices_shape[..indices_rank - 1]);
    output_shape.extend_from_slice(&data_shape[batch_dims + last_dim..]);
    
    let mut result = ArrayD::zeros(output_shape.clone());
    
    // Gather elements
    for idx in ndarray::indices(&indices_shape[..indices_rank - 1]) {
        // Get the indices for this element
        let mut gather_indices = Vec::new();
        for i in 0..last_dim {
            let index = indices[&[idx.slice(), &[i]].concat()[..]];
            
            // Handle negative indices
            let axis_size = data_shape[batch_dims + i];
            let normalized_index = if index < 0 {
                (axis_size as i64 + index) as usize
            } else {
                index as usize
            };
            
            gather_indices.push(normalized_index);
        }
        
        // Construct full index into data tensor
        let mut data_idx = Vec::new();
        data_idx.extend_from_slice(&idx.slice()[..batch_dims]); // Batch dimensions
        data_idx.extend_from_slice(&gather_indices); // Gathered indices
        
        // Get the slice from data tensor
        let slice = data.select(Axis(0), &data_idx);
        
        // Copy to result
        let mut result_idx = idx.slice().to_vec();
        result_idx.extend_from_slice(&[0; 0]); // Placeholder for remaining dimensions
        
        // This is a simplified implementation - in a real implementation, we would need to handle arbitrary slicing
        result[&result_idx[..]] = slice[[0]];
    }
    
    Ok(ndarray_to_ort(result, DataType::Float))
}

fn op_reduce_max(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // ReduceMax computes the maximum value of elements across dimensions of a tensor
    let input = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("ReduceMax requires input tensor"))?)?;
    
    // Get axes attribute or input
    let axes = if inputs.len() > 1 {
        match inputs.get(1) {
            Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
                data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect::<Vec<_>>()
            }
            _ => return Err(OrtError::TypeMismatch("ReduceMax requires Int64 axes tensor as second input")),
        }
    } else {
        node.attributes.iter().find(|a| a.name == "axes")
            .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
            .unwrap_or_else(|| (0..input.ndim()).collect()) // Default to all axes
    };
    
    // Get keepdims attribute
    let keepdims = node.attributes.iter().find(|a| a.name == "keepdims")
        .map(|a| a.i != 0)
        .unwrap_or(true); // Default to true
    
    // Sort axes in descending order to avoid dimension issues when reducing
    let mut sorted_axes = axes.clone();
    sorted_axes.sort_by(|a, b| b.cmp(a));
    
    // Reduce along each axis
    let mut result = input.clone();
    for &axis in &sorted_axes {
        result = result.map_axis(Axis(axis), |view| {
            view.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x))
        });
    }
    
    // Reshape if keepdims is true
    if keepdims {
        let mut new_shape = input.shape().to_vec();
        for &axis in &axes {
            new_shape[axis] = 1;
        }
        result = result.into_shape(new_shape).unwrap();
    }
    
    Ok(ndarray_to_ort(result, DataType::Float))
}

fn op_attention(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // Real implementation of scaled dot-product attention
    // Inputs: query, key, value, mask (optional)
    let query = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Attention requires query tensor"))?)?;
    let key = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Attention requires key tensor"))?)?;
    let value = ort_to_ndarray(inputs.get(2).ok_or_else(|| OrtError::TypeMismatch("Attention requires value tensor"))?)?;
    
    // Optional mask
    let mask = if inputs.len() > 3 {
        Some(ort_to_ndarray(inputs.get(3).unwrap())?)
    } else {
        None
    };
    
    // Get num_heads attribute
    let num_heads = node.attributes.iter().find(|a| a.name == "num_heads")
        .map(|a| a.i as usize)
        .unwrap_or(1); // Default to 1
    
    // Shape validation
    let q_shape = query.shape();
    let k_shape = key.shape();
    let v_shape = value.shape();
    
    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
        return Err(OrtError::TypeMismatch("Attention requires 4D tensors [batch, seq_len, num_heads, head_dim]"));
    }
    
    let (batch_size, q_seq_len, _, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let k_seq_len = k_shape[1];
    
    // Compute attention scores: Q * K^T / sqrt(head_dim)
    let mut scores = Array4::zeros((batch_size, num_heads, q_seq_len, k_seq_len));
    
    // Scale factor for attention scores
    let scale = (head_dim as f32).sqrt();
    
    // Compute scaled dot-product for each batch and head
    for b in 0..batch_size {
        for h in 0..num_heads {
            // Extract query and key for current batch and head
            let q = query.slice(ndarray::s![b, .., h, ..]);
            let k = key.slice(ndarray::s![b, .., h, ..]);
            
            // Compute Q * K^T (matrix multiplication)
            let qk = q.dot(&k.t());
            
            // Scale by sqrt(head_dim)
            let scaled_qk = qk / scale;
            
            // Store in scores tensor
            scores.slice_mut(ndarray::s![b, h, .., ..]).assign(&scaled_qk);
        }
    }
    
    // Apply mask if provided (for causal attention or padding)
    if let Some(m) = mask {
        // Apply mask to attention scores
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..q_seq_len {
                    for j in 0..k_seq_len {
                        // If mask value is 0, set score to -infinity to ensure 0 attention weight
                        if m[[b, 0, i, j]] == 0.0 {
                            scores[[b, h, i, j]] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }
    }
    
    // Apply softmax to get attention weights
    let mut attention_weights = Array4::zeros(scores.raw_dim());
    
    for b in 0..batch_size {
        for h in 0..num_heads {
            for i in 0..q_seq_len {
                // Get row of scores for current query position
                let row = scores.slice(ndarray::s![b, h, i, ..]);
                
                // Compute softmax: exp(x_i - max) / sum(exp(x_j - max))
                // Subtract max for numerical stability
                let max_val = row.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                let exp_row: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let sum_exp: f32 = exp_row.iter().sum();
                
                // Normalize to get probabilities
                for (j, &exp_val) in exp_row.iter().enumerate() {
                    attention_weights[[b, h, i, j]] = exp_val / sum_exp;
                }
            }
        }
    }
    
    // Compute weighted sum: attention_weights * V
    let mut output = Array4::zeros((batch_size, q_seq_len, num_heads, head_dim));
    
    for b in 0..batch_size {
        for h in 0..num_heads {
            for i in 0..q_seq_len {
                // Get attention weights for current query position
                let weights = attention_weights.slice(ndarray::s![b, h, i, ..]);
                // Get value vectors
                let v = value.slice(ndarray::s![b, .., h, ..]);
                
                // Compute weighted sum for each dimension of the value vectors
                for d in 0..head_dim {
                    let mut sum = 0.0;
                    for j in 0..k_seq_len {
                        sum += weights[j] * v[[j, d]];
                    }
                    output[[b, i, h, d]] = sum;
                }
            }
        }
    }
    
    Ok(ndarray_to_ort(output.into_dyn(), DataType::Float))
}
// Kokoro-specific operations
fn op_embedding(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // Real implementation of embedding lookup
    // Embedding performs a lookup in an embedding matrix using input indices
    let indices = match inputs.get(0) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, shape, .. }) => {
            // Convert indices to a usable format
            let concrete_shape: Vec<usize> = shape.iter().map(|d| match d {
                Dimensions::Fixed(n) => Ok(*n),
                Dimensions::Symbolic(_) => return Err(OrtError::InvalidTensorData("Cannot use symbolic shape for indices".into())),
            }).collect::<Result<_, _>>()?;
            
            let indices_vec: Vec<i64> = data.chunks(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            
            ArrayD::from_shape_vec(concrete_shape, indices_vec)
                .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for indices".into()))?
        }
        _ => return Err(OrtError::TypeMismatch("Embedding requires Int64 indices tensor")),
    };
    
    let weights = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Embedding requires weights tensor"))?)?;
    
    // Get embedding size
    let weights_shape = weights.shape();
    let vocab_size = weights_shape[0];
    let embedding_size = weights_shape[1];
    
    // Create output tensor with shape [*indices_shape, embedding_size]
    let mut output_shape = indices.shape().to_vec();
    output_shape.push(embedding_size);
    let mut result = ArrayD::zeros(output_shape.clone());
    
    // Perform embedding lookup for each index
    for idx in ndarray::indices(indices.shape()) {
        let index = indices[idx.slice()];
        
        // Handle negative indices (wrap around) and validate bounds
        let normalized_index = if index < 0 {
            let wrapped_index = (vocab_size as i64 + index) as usize;
            if wrapped_index >= vocab_size {
                return Err(OrtError::IndexError("Negative embedding index out of bounds after wrapping"));
            }
            wrapped_index
        } else if index as usize >= vocab_size {
            return Err(OrtError::IndexError("Embedding index out of bounds"));
        } else {
            index as usize
        };
        
        // Get embedding vector for this index
        let embedding = weights.slice(ndarray::s![normalized_index, ..]);
        
        // Copy embedding vector to result tensor at the correct position
        let mut result_idx = idx.slice().to_vec();
        result_idx.push(0); // Add dimension for embedding vector
        
        
        for i in 0..embedding_size {
            let lenidx=result_idx.len();
            result_idx[lenidx - 1 as usize] = i;
            result[&result_idx[..]] = embedding[i];
        }
    }
    
    Ok(ndarray_to_ort(result, DataType::Float))
}

fn op_constant_of_shape_int64(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // ConstantOfShape creates a tensor with a specified shape and filled with a constant value
    // This version specifically handles Int64 output type for attention masks in BERT
    
    // Get shape from input tensor
    let shape = match inputs.get(0) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
            data.chunks(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
                .collect::<Vec<_>>()
        }
        _ => return Err(OrtError::TypeMismatch("ConstantOfShape requires Int64 shape tensor")),
    };
    
    // Get value attribute (default to 0)
    let value = node.attributes.iter()
        .find(|a| a.name == "value")
        .and_then(|a| a.t.as_ref())
        .map(|t| {
            if t.data_type == 7 { // INT64
                if !t.int64_data.is_empty() {
                    t.int64_data[0]
                } else if !t.raw_data.is_empty() && t.raw_data.len() >= 8 {
                    i64::from_le_bytes(t.raw_data[0..8].try_into().unwrap())
                } else {
                    0
                }
            } else {
                0
            }
        })
        .unwrap_or(0);
    
    // Create tensor filled with the value
    let total_elements: usize = shape.iter().product();
    let data: Vec<u8> = std::iter::repeat(value)
        .take(total_elements)
        .flat_map(|v| v.to_le_bytes())
        .collect();
    
    Ok(OrtValue::Tensor {
        shape: shape.iter().map(|&d| Dimensions::Fixed(d)).collect(),
        dtype: DataType::Int64,
        data: Arc::new(data),
    })
}

fn op_layer_normalization_with_epsilon(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // Enhanced LayerNormalization with proper epsilon handling for BERT
    let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("LayerNormalization requires input tensor"))?)?;
    let scale = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("LayerNormalization requires scale tensor"))?)?;
    let bias = inputs.get(2).map(|b| ort_to_ndarray(b)).transpose()?;
    
    // Get epsilon attribute with proper handling for small values
    let epsilon = node.attributes.iter()
        .find(|a| a.name == "epsilon")
        .map(|a| a.f)
        .unwrap_or(1e-5);
    
    // Get axis attribute (default to -1, which is the last dimension)
    let axis = node.attributes.iter()
        .find(|a| a.name == "axis")
        .map(|a| {
            if a.i < 0 {
                (array.ndim() as i64 + a.i) as usize
            } else {
                a.i as usize
            }
        })
        .unwrap_or(array.ndim() - 1);
    
    // Calculate mean along the specified axis
    let mean = array.mean_axis(Axis(axis))
        .ok_or_else(|| OrtError::InvalidTensorData("LayerNormalization mean error".into()))?;
    
    // Calculate variance along the specified axis
    let mean_squared = mean.mapv(|m| m * m); // Compute mu^2 once
    let var = array.mapv(|x| x * x)
        .mean_axis(Axis(axis))
        .ok_or_else(|| OrtError::InvalidTensorData("LayerNormalization variance error".into()))?
        - &mean_squared; // Element-wise subtraction
    
    // Check for negative variance
    if var.iter().any(|&v| v + epsilon < 0.0) {
        return Err(OrtError::InvalidTensorData("Negative variance detected".into()));
    }

    // Vectorized normalization
    let mut result = &array - &mean;
    result /= &(&var + epsilon).mapv(|x| x.sqrt());
    
    // Apply scale and bias
    if scale.ndim() > 0 {
        for idx in ndarray::indices(result.shape()) {
            let mut scale_idx = vec![0; scale.ndim()];
            for (i, &dim) in idx.slice()[axis..].iter().enumerate() {
                if i < scale_idx.len() {
                    scale_idx[i] = dim;
                }
            }
            
            result[idx.slice()] *= scale[&scale_idx[..]];
            
            if let Some(ref b) = bias {
                result[idx.slice()] += b[&scale_idx[..]];
            }
        }
    }
    
    Ok(ndarray_to_ort(result, DataType::Float))
}


fn op_position_embeddings(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // Position embeddings for BERT - adds position information to token embeddings
    let token_embeddings = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("PositionEmbeddings requires token embeddings tensor"))?)?;
    
    // Get position embeddings from input or attribute
    let position_embeddings = if inputs.len() > 1 {
        ort_to_ndarray(inputs.get(1).unwrap())?
    } else {
        // If not provided as input, use attribute
        let position_embedding_attr = node.attributes.iter()
            .find(|a| a.name == "position_embeddings")
            .and_then(|a| a.t.as_ref())
            .ok_or_else(|| OrtError::InvalidTensorData("PositionEmbeddings requires position_embeddings attribute".into()))?;
        
        // Convert tensor proto to ndarray
        let shape: Vec<usize> = position_embedding_attr.dims.iter().map(|&d| d as usize).collect();
        let data: Vec<f32> = if !position_embedding_attr.float_data.is_empty() {
            position_embedding_attr.float_data.clone()
        } else if !position_embedding_attr.raw_data.is_empty() {
            position_embedding_attr.raw_data.chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        } else {
            return Err(OrtError::InvalidTensorData("Position embeddings tensor has no data".into()));
        };
        
        ArrayD::from_shape_vec(shape, data)
            .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for position embeddings".into()))?
    };
    
    // Get max sequence length
    let token_shape = token_embeddings.shape();
    let seq_length = token_shape[1]; // Assuming shape is [batch_size, seq_length, hidden_size]
    
    // Add position embeddings to token embeddings
    let mut result = token_embeddings.clone();
    
    for b in 0..token_shape[0] {
        for pos in 0..seq_length {
            for h in 0..token_shape[2] {
                result[[b, pos, h]] += position_embeddings[[pos, h]];
            }
        }
    }
    
    Ok(ndarray_to_ort(result, DataType::Float))
}

fn op_token_type_embeddings(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // Token type embeddings for BERT - adds token type information to embeddings
    let embeddings = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("TokenTypeEmbeddings requires embeddings tensor"))?)?;
    
    // Get token type ids
    let token_type_ids = match inputs.get(1) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, shape, .. }) => {
            // Convert token type ids to a usable format
            let concrete_shape: Vec<usize> = shape.iter().map(|d| match d {
                Dimensions::Fixed(n) => Ok(*n),
                Dimensions::Symbolic(_) => return Err(OrtError::InvalidTensorData("Cannot use symbolic shape for token type ids".into())),
            }).collect::<Result<_, _>>()?;
            
            let ids_vec: Vec<i64> = data.chunks(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            
            ArrayD::from_shape_vec(concrete_shape, ids_vec)
                .map_err(|_| OrtError::InvalidTensorData("Shape mismatch for token type ids".into()))?
        }
        _ => return Err(OrtError::TypeMismatch("TokenTypeEmbeddings requires Int64 token type ids tensor")),
    };
    
    // Get token type embeddings
    let token_type_embeddings = ort_to_ndarray(inputs.get(2).ok_or_else(|| OrtError::TypeMismatch("TokenTypeEmbeddings requires token type embeddings tensor"))?)?;
    
    // Add token type embeddings to input embeddings
    let mut result = embeddings.clone();
    
    let emb_shape = embeddings.shape();
    let hidden_size = emb_shape[2]; // Assuming shape is [batch_size, seq_length, hidden_size]
    
    for b in 0..emb_shape[0] {
        for pos in 0..emb_shape[1] {
            let token_type = token_type_ids[[b, pos]] as usize;
            for h in 0..hidden_size {
                result[[b, pos, h]] += token_type_embeddings[[token_type, h]];
            }
        }
    }
    
    Ok(ndarray_to_ort(result, DataType::Float))
}

fn op_bert_attention(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // BERT-specific attention implementation
    let hidden_states = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("BertAttention requires hidden states tensor"))?)?;
    
    // Get attention mask (optional)
    let attention_mask = if inputs.len() > 1 {
        Some(ort_to_ndarray(inputs.get(1).unwrap())?)
    } else {
        None
    };
    
    // Get query, key, value weights
    let query_weight = ort_to_ndarray(inputs.get(2).ok_or_else(|| OrtError::TypeMismatch("BertAttention requires query weight tensor"))?)?;
    let key_weight = ort_to_ndarray(inputs.get(3).ok_or_else(|| OrtError::TypeMismatch("BertAttention requires key weight tensor"))?)?;
    let value_weight = ort_to_ndarray(inputs.get(4).ok_or_else(|| OrtError::TypeMismatch("BertAttention requires value weight tensor"))?)?;
    
    // Get query, key, value biases (optional)
    let query_bias = if inputs.len() > 5 { Some(ort_to_ndarray(inputs.get(5).unwrap())?) } else { None };
    let key_bias = if inputs.len() > 6 { Some(ort_to_ndarray(inputs.get(6).unwrap())?) } else { None };
    let value_bias = if inputs.len() > 7 { Some(ort_to_ndarray(inputs.get(7).unwrap())?) } else { None };
    
    // Get attributes
    let num_heads = node.attributes.iter().find(|a| a.name == "num_attention_heads")
        .map(|a| a.i as usize)
        .unwrap_or(12); // Default for BERT-base
    
    let hidden_size = hidden_states.shape()[2];
    let attention_head_size = hidden_size / num_heads;
    let batch_size = hidden_states.shape()[0];
    let seq_length = hidden_states.shape()[1];
    
    // Compute query, key, value projections
    let mut query = Array3::zeros((batch_size, seq_length, hidden_size));
    let mut key = Array3::zeros((batch_size, seq_length, hidden_size));
    let mut value = Array3::zeros((batch_size, seq_length, hidden_size));
    
    // Simplified matrix multiplication for projections
    for b in 0..batch_size {
        for s in 0..seq_length {
            // Query projection
            for i in 0..hidden_size {
                let mut sum = 0.0;
                for j in 0..hidden_size {
                    sum += hidden_states[[b, s, j]] * query_weight[[j, i]];
                }
                if let Some(ref bias) = query_bias {
                    sum += bias[[i]];
                }
                query[[b, s, i]] = sum;
            }
            
            // Key projection
            for i in 0..hidden_size {
                let mut sum = 0.0;
                for j in 0..hidden_size {
                    sum += hidden_states[[b, s, j]] * key_weight[[j, i]];
                }
                if let Some(ref bias) = key_bias {
                    sum += bias[[i]];
                }
                key[[b, s, i]] = sum;
            }
            
            // Value projection
            for i in 0..hidden_size {
                let mut sum = 0.0;
                for j in 0..hidden_size {
                    sum += hidden_states[[b, s, j]] * value_weight[[j, i]];
                }
                if let Some(ref bias) = value_bias {
                    sum += bias[[i]];
                }
                value[[b, s, i]] = sum;
            }
        }
    }
    
    // Reshape to [batch_size, seq_length, num_heads, attention_head_size]
    let mut query_reshaped = Array4::zeros((batch_size, seq_length, num_heads, attention_head_size));
    let mut key_reshaped = Array4::zeros((batch_size, seq_length, num_heads, attention_head_size));
    let mut value_reshaped = Array4::zeros((batch_size, seq_length, num_heads, attention_head_size));
    
    for b in 0..batch_size {
        for s in 0..seq_length {
            for h in 0..num_heads {
                for d in 0..attention_head_size {
                    let idx = h * attention_head_size + d;
                    query_reshaped[[b, s, h, d]] = query[[b, s, idx]];
                    key_reshaped[[b, s, h, d]] = key[[b, s, idx]];
                    value_reshaped[[b, s, h, d]] = value[[b, s, idx]];
                }
            }
        }
    }
    
    // Transpose key for matrix multiplication: [batch_size, num_heads, seq_length, attention_head_size]
    let mut key_transposed = Array4::zeros((batch_size, num_heads, seq_length, attention_head_size));
    for b in 0..batch_size {
        for h in 0..num_heads {
            for s in 0..seq_length {
                for d in 0..attention_head_size {
                    key_transposed[[b, h, s, d]] = key_reshaped[[b, s, h, d]];
                }
            }
        }
    }
    
    // Compute attention scores: [batch_size, num_heads, seq_length, seq_length]
    let mut attention_scores = Array4::zeros((batch_size, num_heads, seq_length, seq_length));
    
    for b in 0..batch_size {
        for h in 0..num_heads {
            for q_seq in 0..seq_length {
                for k_seq in 0..seq_length {
                    let mut score = 0.0;
                    for d in 0..attention_head_size {
                        score += query_reshaped[[b, q_seq, h, d]] * key_transposed[[b, h, k_seq, d]];
                    }
                    attention_scores[[b, h, q_seq, k_seq]] = score / (attention_head_size as f32).sqrt();
                }
            }
        }
    }
    
    // Apply attention mask if provided
    if let Some(mask) = attention_mask {
        for b in 0..batch_size {
            for h in 0..num_heads {
                for q_seq in 0..seq_length {
                    for k_seq in 0..seq_length {
                        if mask[[b, k_seq]] == 0.0 {
                            attention_scores[[b, h, q_seq, k_seq]] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }
    }
    
    // Apply softmax to get attention probabilities
    let mut attention_probs = Array4::zeros(attention_scores.raw_dim());
    
    for b in 0..batch_size {
        for h in 0..num_heads {
            for q_seq in 0..seq_length {
                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for k_seq in 0..seq_length {
                    max_val = max_val.max(attention_scores[[b, h, q_seq, k_seq]]);
                }
                
                // Compute softmax: exp(x_i - max) / sum(exp(x_j - max))
                let mut sum_exp = 0.0;
                for k_seq in 0..seq_length {
                    sum_exp += (attention_scores[[b, h, q_seq, k_seq]] - max_val).exp();
                }
                
                for k_seq in 0..seq_length {
                    attention_probs[[b, h, q_seq, k_seq]] = 
                        (attention_scores[[b, h, q_seq, k_seq]] - max_val).exp() / sum_exp;
                }
            }
        }
    }
    
    // Apply attention to values
    let mut context = Array4::zeros((batch_size, seq_length, num_heads, attention_head_size));
    
    for b in 0..batch_size {
        for h in 0..num_heads {
            for q_seq in 0..seq_length {
                for d in 0..attention_head_size {
                    let mut sum = 0.0;
                    for k_seq in 0..seq_length {
                        sum += attention_probs[[b, h, q_seq, k_seq]] * value_reshaped[[b, k_seq, h, d]];
                    }
                    context[[b, q_seq, h, d]] = sum;
                }
            }
        }
    }
    
    // Reshape back to [batch_size, seq_length, hidden_size]
    let mut result = Array3::zeros((batch_size, seq_length, hidden_size));
    
    for b in 0..batch_size {
        for s in 0..seq_length {
            for h in 0..num_heads {
                for d in 0..attention_head_size {
                    let idx = h * attention_head_size + d;
                    result[[b, s, idx]] = context[[b, s, h, d]];
                }
            }
        }
    }
    
    Ok(ndarray_to_ort(result.into_dyn(), DataType::Float))
}

fn op_bert_intermediate(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // BERT intermediate layer with GELU activation
    let hidden_states = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("BertIntermediate requires hidden states tensor"))?)?;
    let weights = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("BertIntermediate requires weights tensor"))?)?;
    
    // Get bias (optional)
    let bias = if inputs.len() > 2 {
        Some(ort_to_ndarray(inputs.get(2).unwrap())?)
    } else {
        None
    };
    
    // Get hidden size and intermediate size
    let hidden_shape = hidden_states.shape();
    let batch_size = hidden_shape[0];
    let seq_length = hidden_shape[1];
    let hidden_size = hidden_shape[2];
    
    let weights_shape = weights.shape();
    let intermediate_size = weights_shape[0];
    
    // Compute intermediate output
    let mut intermediate_output = Array3::zeros((batch_size, seq_length, intermediate_size));
    
    // Linear projection
    for b in 0..batch_size {
        for s in 0..seq_length {
            for i in 0..intermediate_size {
                let mut sum = 0.0;
                for j in 0..hidden_size {
                    sum += hidden_states[[b, s, j]] * weights[[i, j]];
                }
                if let Some(ref b) = bias {
                    sum += b[[i]];
                }
                intermediate_output[[b, s, i]] = sum;
            }
        }
    }
    
    // Apply GELU activation
    let result = intermediate_output.mapv(|x| {
        // Constants for erf approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        
        // Calculate x / sqrt(2)
        let x_scaled = x / 1.4142135623730951;
        
        // Save the sign of x_scaled
        let sign = if x_scaled < 0.0 { -1.0 } else { 1.0 };
        let x_abs = x_scaled.abs();
        
        // A&S formula 7.1.26 for erf
        let t = 1.0 / (1.0 + p * x_abs);
        let erf = sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp());
        
        // GELU formula
        0.5 * x * (1.0 + erf)
    });
    
    Ok(ndarray_to_ort(result.into_dyn(), DataType::Float))
}

fn op_bert_output(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // BERT output layer with residual connection and layer normalization
    let hidden_states = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("BertOutput requires hidden states tensor"))?)?;
    let input_tensor = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("BertOutput requires input tensor"))?)?;
    let weights = ort_to_ndarray(inputs.get(2).ok_or_else(|| OrtError::TypeMismatch("BertOutput requires weights tensor"))?)?;
    
    // Get bias (optional)
    let bias = if inputs.len() > 3 {
        Some(ort_to_ndarray(inputs.get(3).unwrap())?)
    } else {
        None
    };
    
    // Get layer norm weights and bias
    let layer_norm_weight = ort_to_ndarray(inputs.get(4).ok_or_else(|| OrtError::TypeMismatch("BertOutput requires layer norm weight tensor"))?)?;
    let layer_norm_bias = if inputs.len() > 5 {
        Some(ort_to_ndarray(inputs.get(5).unwrap())?)
    } else {
        None
    };
    
    // Get hidden size
    let hidden_shape = hidden_states.shape();
    let batch_size = hidden_shape[0];
    let seq_length = hidden_shape[1];
    let intermediate_size = hidden_shape[2];
    
    let weights_shape = weights.shape();
    let hidden_size = weights_shape[0];
    
    // Compute output
    let mut output = Array3::zeros((batch_size, seq_length, hidden_size));
    
    // Linear projection
    for b in 0..batch_size {
        for s in 0..seq_length {
            for i in 0..hidden_size {
                let mut sum = 0.0;
                for j in 0..intermediate_size {
                    sum += hidden_states[[b, s, j]] * weights[[i, j]];
                }
                if let Some(ref b) = bias {
                    sum += b[[i]];
                }
                output[[b, s, i]] = sum;
            }
        }
    }
    
    // Add residual connection
    for b in 0..batch_size {
        for s in 0..seq_length {
            for i in 0..hidden_size {
                output[[b, s, i]] += input_tensor[[b, s, i]];
            }
        }
    }
    
    // Apply layer normalization
    let mut result = output.clone();
    
    // For each token, normalize across hidden dimension
    for b in 0..batch_size {
        for s in 0..seq_length {
            // Compute mean
            let mut mean = 0.0;
            for i in 0..hidden_size {
                mean += output[[b, s, i]];
            }
            mean /= hidden_size as f32;
            
            // Compute variance
            let mut var = 0.0;
            for i in 0..hidden_size {
                var += (output[[b, s, i]] - mean).powi(2);
            }
            var /= hidden_size as f32;
            
            // Normalize, scale, and shift
            for i in 0..hidden_size {
                let normalized = (output[[b, s, i]] - mean) / (var + 1e-12).sqrt();
                result[[b, s, i]] = normalized * layer_norm_weight[[i]];
                if let Some(ref bias) = layer_norm_bias {
                    result[[b, s, i]] += bias[[i]];
                }
            }
        }
    }
    
    Ok(ndarray_to_ort(result.into_dyn(), DataType::Float))
}

fn op_bert_pooler(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // BERT pooler for sentence representation
    let hidden_states = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("BertPooler requires hidden states tensor"))?)?;
    let weights = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("BertPooler requires weights tensor"))?)?;
    
    // Get bias (optional)
    let bias = if inputs.len() > 2 {
        Some(ort_to_ndarray(inputs.get(2).unwrap())?)
    } else {
        None
    };
    
    // Get hidden size
    let hidden_shape = hidden_states.shape();
    let batch_size = hidden_shape[0];
    let hidden_size = hidden_shape[2];
    
    // Extract first token ([CLS]) representation
    let mut first_token = Array2::zeros((batch_size, hidden_size));
    for b in 0..batch_size {
        for i in 0..hidden_size {
            first_token[[b, i]] = hidden_states[[b, 0, i]];
        }
    }
    
    // Apply linear transformation
    let mut pooled_output = Array2::zeros((batch_size, hidden_size));
    
    for b in 0..batch_size {
        for i in 0..hidden_size {
            let mut sum = 0.0;
            for j in 0..hidden_size {
                sum += first_token[[b, j]] * weights[[i, j]];
            }
            if let Some(ref b) = bias {
                sum += b[[i]];
            }
            pooled_output[[b, i]] = sum;
        }
    }
    
    // Apply tanh activation
    let result = pooled_output.mapv(|x| x.tanh());
    
    Ok(ndarray_to_ort(result.into_dyn(), DataType::Float))
}
    fn op_gemm(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let alpha = node.attributes.iter().find(|a| a.name == "alpha").map(|a| a.f).unwrap_or(1.0);
    let beta = node.attributes.iter().find(|a| a.name == "beta").map(|a| a.f).unwrap_or(1.0);
    let trans_a = node.attributes.iter().find(|a| a.name == "transA").map(|a| a.i != 0).unwrap_or(false);
    let trans_b = node.attributes.iter().find(|a| a.name == "transB").map(|a| a.i != 0).unwrap_or(false);
    let mut array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Gemm requires two float tensors"))?)?;
    let mut array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Gemm requires two float tensors"))?)?;
    if trans_a {
        array1 = array1.t().to_owned();
    }
    if trans_b {
        array2 = array2.t().to_owned();
    }
    let shape1 = array1.shape();
    let shape2 = array2.shape();
    let ndim1 = shape1.len();
    let ndim2 = shape2.len();

    // Validate dimensions: 2D or 3D (batched 2D)
    if ndim1 < 2 || ndim2 < 2 || shape1[ndim1 - 1] != shape2[ndim2 - 2] {
        // format!(
        //     "Invalid shapes for Gemm: {:?} × {:?}", shape1, shape2
        // )
        return Err(OrtError::TypeMismatch(&"Invalid shapes for Gemm"));
    }
    if ndim1 > 3 || ndim2 > 3 {
        return Err(OrtError::TypeMismatch("Gemm supports only 2D or batched 2D tensors"));
    }

    let mut result = if ndim1 == 2 && ndim2 == 2 {
    // 2D matrix multiplication: [m, k] × [k, n] → [m, n]
    let a1 = array1.clone()
        .into_shape((shape1[0], shape1[1]))
        .map_err(|_| OrtError::TypeMismatch("Failed to reshape array1 to 2D"))?;
    let a2 = array2.clone()
        .into_shape((shape2[0], shape2[1]))
        .map_err(|_| OrtError::TypeMismatch("Failed to reshape array2 to 2D"))?;
    (a1.dot(&a2) * alpha).into_dyn() // Changed line
} else {
    // Batched 2D (3D): [batch, m, k] × [batch, k, n] → [batch, m, n]
    let batch1 = if ndim1 == 3 { shape1[0] } else { 1 };
    let batch2 = if ndim2 == 3 { shape2[0] } else { 1 };
    let batch = if batch1 == 1 {
        batch2
    } else if batch2 == 1 {
        batch1
    } else if batch1 == batch2 {
        batch1
    } else {
        // format!(
        //     "Incompatible batch dimensions: {} vs {}", batch1, batch2
        // )
        return Err(OrtError::TypeMismatch(&"Incompatible batch dimensions"));
    };
    let m = shape1[ndim1 - 2];
    let k = shape1[ndim1 - 1];
    let n = shape2[ndim2 - 1];
    let a1 = array1
        .into_shape(if ndim1 == 3 { (batch1, m, k) } else { (1, m, k) })
        .map_err(|_| OrtError::TypeMismatch("Failed to reshape array1 to 3D"))?;
    let a2 = array2
        .into_shape(if ndim2 == 3 { (batch2, k, n) } else { (1, k, n) })
        .map_err(|_| OrtError::TypeMismatch("Failed to reshape array2 to 3D"))?;
    let mut result = Array3::zeros((batch, m, n));
    for b in 0..batch {
        let a1_slice = a1.slice_axis(Axis(0), ndarray::Slice::from(b..b + 1)).into_shape((m, k)).unwrap();
        let a2_slice = a2.slice_axis(Axis(0), ndarray::Slice::from(b..b + 1)).into_shape((k, n)).unwrap();
        result.slice_mut(ndarray::s![b, .., ..]).assign(&(a1_slice.dot(&a2_slice) * alpha));
    }
    result.into_dyn()
};
    if let Some(bias) = inputs.get(2) {
        let bias_array = ort_to_ndarray(bias)?;
        let bias_shape = bias_array.shape();
        let result_shape = result.shape();
        if bias_shape.len() == 2 || (bias_shape.len() == 3 && bias_shape[0] == 1) {
            let bias_2d = bias_array.clone()
                .into_shape((bias_shape[bias_shape.len() - 2], bias_shape[bias_shape.len() - 1]))
                .map_err(|_| OrtError::TypeMismatch("Failed to reshape bias to 2D"))?;
            if result_shape[result_shape.len() - 1] != bias_shape[bias_shape.len() - 1] {
                return Err(OrtError::TypeMismatch("Bias shape incompatible with result"));
            }
            if result_shape.len() == 2 {
                result = result + (bias_2d * beta);
            } else {
                for b in 0..result_shape[0] {
                    let temp = result.slice(ndarray::s![b, .., ..]).to_owned() + (bias_2d.clone() * beta);
result.slice_mut(ndarray::s![b, .., ..]).assign(&temp);
                }
            }
        } else {
            return Err(OrtError::TypeMismatch("Bias must be 2D or batched 2D"));
        }
    }
    Ok(ndarray_to_ort(result.into_dyn(), DataType::Float))
}
    // Indexing Operations
    fn op_gather(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let axis = node.attributes.iter().find(|a| a.name == "axis").map(|a| a.i as usize).unwrap_or(0);
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Gather requires data and indices tensors"))?)?;
        let indices = match inputs.get(1) {
            Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
                data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect::<Vec<_>>()
            }
            _ => return Err(OrtError::TypeMismatch("Gather requires Int64 indices")),
        };
        let result = array.select(Axis(axis), &indices);
        Ok(ndarray_to_ort(result, DataType::Float))
    }

fn op_slice(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let tensor = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Slice requires data tensor"))?;
    let array = match tensor {
        OrtValue::Tensor { dtype: DataType::Float, .. } => ort_to_ndarray(tensor)?,
        OrtValue::Tensor { dtype: DataType::Int64, .. } => {
            return Err(OrtError::TypeMismatch("Slice requires first input to be a float tensor"))
        }
        OrtValue::Tensor { dtype: DataType::String, .. } => {
            return Err(OrtError::TypeMismatch("Slice requires first input to be a float tensor"))
        }
        OrtValue::Sequence(_) | OrtValue::Map(_) | OrtValue::Opaque(_) => {
            return Err(OrtError::TypeMismatch("Slice requires first input to be a tensor"))
        }
    };
    let starts = match inputs.get(1) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
            data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect::<Vec<_>>()
        }
        Some(OrtValue::Tensor { dtype: DataType::Float, .. }) | Some(OrtValue::Tensor { dtype: DataType::String, .. }) => {
            return Err(OrtError::TypeMismatch("Slice requires Int64 starts"))
        }
        Some(OrtValue::Sequence(_)) | Some(OrtValue::Map(_)) | Some(OrtValue::Opaque(_)) => {
            return Err(OrtError::TypeMismatch("Slice requires Int64 starts"))
        }
        _ => return Err(OrtError::TypeMismatch("Slice requires Int64 starts")),
    };
    let ends = match inputs.get(2) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
            data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect::<Vec<_>>()
        }
        Some(OrtValue::Tensor { dtype: DataType::Float, .. }) | Some(OrtValue::Tensor { dtype: DataType::String, .. }) => {
            return Err(OrtError::TypeMismatch("Slice requires Int64 ends"))
        }
        Some(OrtValue::Sequence(_)) | Some(OrtValue::Map(_)) | Some(OrtValue::Opaque(_)) => {
            return Err(OrtError::TypeMismatch("Slice requires Int64 ends"))
        }
        _ => return Err(OrtError::TypeMismatch("Slice requires Int64 ends")),
    };
    let axes = match inputs.get(3) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
            data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect::<Vec<_>>()
        }
        Some(OrtValue::Tensor { dtype: DataType::Float, .. }) | Some(OrtValue::Tensor { dtype: DataType::String, .. }) => {
            return Err(OrtError::TypeMismatch("Slice requires Int64 axes"))
        }
        Some(OrtValue::Sequence(_)) | Some(OrtValue::Map(_)) | Some(OrtValue::Opaque(_)) => {
            return Err(OrtError::TypeMismatch("Slice requires Int64 axes"))
        }
        None => (0..starts.len()).collect(),
    };
    let steps = match inputs.get(4) {
        Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
            data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect::<Vec<_>>()
        }
        Some(OrtValue::Tensor { dtype: DataType::Float, .. }) | Some(OrtValue::Tensor { dtype: DataType::String, .. }) => {
            return Err(OrtError::TypeMismatch("Slice requires Int64 steps"))
        }
        Some(OrtValue::Sequence(_)) | Some(OrtValue::Map(_)) | Some(OrtValue::Opaque(_)) => {
            return Err(OrtError::TypeMismatch("Slice requires Int64 steps"))
        }
        None => vec![1; starts.len()],
    };
    let mut result = array;
    for (start, end, axis, step) in starts.iter().zip(ends.iter()).zip(axes.iter()).zip(steps.iter()).map(|(((&start, &end), &axis), &step)| (start, end, axis, step)) {
        let slice = ndarray::Slice::from(start as isize..end as isize).step_by(step as isize);
        result = result.slice_axis(Axis(axis), slice).to_owned();
    }
    Ok(ndarray_to_ort(result, DataType::Float))
}
   

    fn op_where(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let condition = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Where requires condition tensor"))?)?;
        let x = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Where requires x tensor"))?)?;
        let y = ort_to_ndarray(inputs.get(2).ok_or_else(|| OrtError::TypeMismatch("Where requires y tensor"))?)?;
        let result = ndarray::Zip::from(&condition).and(&x).and(&y)
            .map_collect(|&c, &x, &y| if c != 0.0 { x } else { y });
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    // Softmax
    fn op_softmax(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let axis = node.attributes.iter().find(|a| a.name == "axis")
            .map(|a| a.i as usize)
            .unwrap_or_else(|| inputs[0].shape().len().saturating_sub(1));
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Softmax requires one float tensor"))?)?;
        let exp_array = array.mapv(|a| a.exp());
        let sum = exp_array.sum_axis(Axis(axis));
        let result = exp_array / sum;
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    // Reduction Operations
    
    // Range
    fn op_range(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let start = match inputs.get(0) {
            Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
                i64::from_le_bytes(data[..8].try_into().unwrap())
            }
            _ => return Err(OrtError::TypeMismatch("Range requires Int64 start")),
        };
        let limit = match inputs.get(1) {
            Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
                i64::from_le_bytes(data[..8].try_into().unwrap())
            }
            _ => return Err(OrtError::TypeMismatch("Range requires Int64 limit")),
        };
        let delta = match inputs.get(2) {
            Some(OrtValue::Tensor { dtype: DataType::Int64, data, .. }) => {
                i64::from_le_bytes(data[..8].try_into().unwrap())
            }
            _ => return Err(OrtError::TypeMismatch("Range requires Int64 delta")),
        };
        let range: Vec<i64> = (start..limit).step_by(delta as usize).collect();
        let data: Vec<u8> = range.iter().flat_map(|x| x.to_le_bytes()).collect();
        Ok(OrtValue::Tensor {
            shape: vec![Dimensions::Fixed(range.len())],
            dtype: DataType::Int64,
            data: Arc::new(data),
        })
    }

    // Clip
    fn op_clip(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let min = node.attributes.iter().find(|a| a.name == "min").map(|a| a.f).unwrap_or(f32::NEG_INFINITY);
        let max = node.attributes.iter().find(|a| a.name == "max").map(|a| a.f).unwrap_or(f32::INFINITY);
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Clip requires one float tensor"))?)?;
        let result = array.mapv(|a| a.clamp(min, max));
        Ok(ndarray_to_ort(result, DataType::Float))
    }

    // Cast
    fn op_cast(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let to = node.attributes.iter().find(|a| a.name == "to")
            .map(|a| a.i)
            .ok_or_else(|| OrtError::InvalidTensorData("Cast requires 'to' attribute".into()))?;
        let tensor = inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Cast requires one tensor"))?;
        match (tensor, to) {
            (OrtValue::Tensor { shape, dtype: DataType::Float, data, .. }, 7) => {
                let float_data: Vec<f32> = data.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                let int_data: Vec<u8> = float_data.iter().flat_map(|&f| (f as i64).to_le_bytes()).collect();
                Ok(OrtValue::Tensor {
                    shape: shape.clone(),
                    dtype: DataType::Int64,
                    data: Arc::new(int_data),
                })
            }
            (OrtValue::Tensor { shape, dtype: DataType::Int64, data, .. }, 1) => {
                let int_data: Vec<i64> = data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
                let float_data: Vec<u8> = int_data.iter().flat_map(|&i| (i as f32).to_le_bytes()).collect();
                Ok(OrtValue::Tensor {
                    shape: shape.clone(),
                    dtype: DataType::Float,
                    data: Arc::new(float_data),
                })
            }
            _ => Err(OrtError::TypeMismatch("Unsupported cast operation")),
        }
    }


    fn op_layer_normalization(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        // Real implementation of Layer Normalization
        // Get epsilon parameter (small constant for numerical stability)
        let epsilon = node.attributes.iter().find(|a| a.name == "epsilon").map(|a| a.f).unwrap_or(1e-5);
        
        // Get input tensor, scale, and optional bias
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("LayerNormalization requires input tensor"))?)?;
        let scale = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("LayerNormalization requires scale tensor"))?)?;
        let bias = inputs.get(2).map(|b| ort_to_ndarray(b)).transpose()?;
        
        // Compute mean along the last dimension (normalization axis)
        let mean = array.mean_axis(Axis(array.ndim() - 1))
            .ok_or_else(|| OrtError::InvalidTensorData("LayerNormalization mean calculation error".into()))?;
        
        // Compute variance along the last dimension
        // First, calculate squared differences from mean
        let mut var = Array::zeros(mean.raw_dim());
        for idx in ndarray::indices(mean.shape()) {
            let mut sum_squared_diff = 0.0;
            let mut count = 0;
            
            // Iterate over the last dimension
            for i in 0..array.shape()[array.ndim() - 1] {
                let mut full_idx = idx.slice().to_vec();
                full_idx.push(i);
                
                let diff = array[&full_idx[..]] - mean[idx.slice()];
                sum_squared_diff += diff * diff;
                count += 1;
            }
            
            // Calculate variance
            var[idx.slice()] = sum_squared_diff / count as f32;
        }
        
        // Normalize the input: (x - mean) / sqrt(var + epsilon)
        let mut result = ArrayD::zeros(array.raw_dim());
        
        for idx in ndarray::indices(array.shape()) {
            let last_dim = idx[array.ndim() - 1];
            let mut mean_idx = idx.slice().to_vec();
            mean_idx.pop(); // Remove last dimension for mean/var indexing
            
            // Normalize
            let normalized = (array[idx.slice()] - mean[&mean_idx[..]]) / 
                             (var[&mean_idx[..]] + epsilon).sqrt();
            
            // Scale and shift
            result[idx.slice()] = normalized * scale[last_dim];
            
            // Add bias if provided
            if let Some(ref b) = bias {
                result[idx.slice()] += b[last_dim];
            }
        }
        
        Ok(ndarray_to_ort(result, DataType::Float))
    }

   fn op_pad(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    let pads = node.attributes.iter().find(|a| a.name == "pads")
        .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
        .ok_or_else(|| OrtError::InvalidTensorData("Pad requires pads attribute".into()))?;
    let mode = node.attributes.iter().find(|a| a.name == "mode")
        .map(|a| String::from_utf8_lossy(&a.s).to_string())
        .unwrap_or("constant".to_string());
    let value = node.attributes.iter().find(|a| a.name == "value").map(|a| a.f).unwrap_or(0.0);
    let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Pad requires one float tensor"))?)?;
    let mut new_shape = array.shape().to_vec();
    for i in 0..new_shape.len() {
        new_shape[i] += pads[i] + pads[i + new_shape.len()];
    }
    let mut result = ArrayD::from_elem(new_shape, value);
    let orig_shape = array.shape();
    if orig_shape.len() != pads.len() / 2 {
        return Err(OrtError::InvalidTensorData("Pads length must match twice the input dimensions".into()));
    }
    let mut slices = Vec::new();
    for i in 0..orig_shape.len() {
        let start = pads[i];
        let end = start + orig_shape[i];
        slices.push(ndarray::SliceInfoElem::Slice {
            start: start as isize,
            end: Some(end as isize),
            step: 1,
        });
    }
    // Explicitly pass as &[SliceInfoElem] and validate dimensions
    if slices.len() == result.shape().len() {
        result.slice_mut(&slices[..]).assign(&array);
    } else {
        return Err(OrtError::InvalidTensorData("Slice dimensions do not match array dimensions".into()));
    }
    Ok(ndarray_to_ort(result, DataType::Float))
}

fn op_if(&self, node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let condition = match inputs.get(0) {
            Some(OrtValue::Tensor { dtype: DataType::Float, data, .. }) => {
                let value = f32::from_le_bytes(data[..4].try_into().unwrap());
                value != 0.0
            }
            _ => return Err(OrtError::TypeMismatch("If requires a float condition tensor")),
        };
        let then_branch = node.attributes.iter().find(|a| a.name == "then_branch")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| OrtError::InvalidTensorData("If requires then_branch subgraph".into()))?;
        let else_branch = node.attributes.iter().find(|a| a.name == "else_branch")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| OrtError::InvalidTensorData("If requires else_branch subgraph".into()))?;
        let subgraph = if condition { then_branch } else { else_branch };
        let mut subgraph_inputs = HashMap::new();
        for (i, input) in node.input.iter().skip(1).enumerate() {
            subgraph_inputs.insert(subgraph.input[i].name.clone(), inputs[i + 1].clone());
        }
        let outputs = self.execute_subgraph(subgraph, subgraph_inputs)?;
        Ok(outputs.into_iter().next().unwrap().1) // Assuming single output for simplicity
    }

   fn op_loop(&self, node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
    // Extract max trip count (M)
    let max_trip_count = inputs.get(0).map(|v| match v {
        OrtValue::Tensor { dtype: DataType::Int64, data, .. } => {
            i64::from_le_bytes(data[..8].try_into().unwrap())
        }
        _ => i64::MAX,
    }).unwrap_or(i64::MAX);

    // Extract initial condition
    let mut condition = inputs.get(1).map(|v| match v {
        OrtValue::Tensor { dtype: DataType::Float, data, .. } => {
            f32::from_le_bytes(data[..4].try_into().unwrap()) != 0.0
        }
        _ => true,
    }).unwrap_or(true);

    // Get the loop body subgraph
    let body = node.attributes.iter().find(|a| a.name == "body")
        .and_then(|a| a.g.as_ref())
        .ok_or_else(|| OrtError::InvalidTensorData("Loop requires body subgraph".into()))?;

    // Initialize state with inputs (skip M and condition)
    let mut state = inputs[2..].to_vec();

    // Collect scan outputs
    let mut scan_outputs = vec![];

    let mut trip_count = 0;
    while condition && trip_count < max_trip_count {
        // Prepare subgraph inputs
        let mut subgraph_inputs = HashMap::new();
        for (i, input) in body.input.iter().enumerate() {
            if i < state.len() {
                subgraph_inputs.insert(input.name.clone(), state[i].clone());
            } else {
                return Err(OrtError::InvalidTensorData(format!(
                    "Loop subgraph input {} not found in state",
                    input.name
                )));
            }
        }

        // Execute subgraph
        let subgraph_outputs = self.execute_subgraph(body, subgraph_inputs)?;

        // Update state with subgraph outputs (first N outputs are state variables)
        let mut new_state = vec![];
        for (i, output) in body.output.iter().enumerate().take(state.len()) {
            if let Some(value) = subgraph_outputs.get(&output.name) {
                new_state.push(value.clone());
            } else {
                return Err(OrtError::MissingOutput(output.name.clone()));
            }
        }
        state = new_state;

        // Collect scan outputs (outputs beyond the state variables)
        for output in body.output.iter().skip(state.len()) {
            if let Some(value) = subgraph_outputs.get(&output.name) {
                scan_outputs.push(value.clone());
            } else {
                return Err(OrtError::MissingOutput(output.name.clone()));
            }
        }

        // Update condition from subgraph outputs (if provided)
        if let Some(cond_tensor) = subgraph_outputs.iter().find(|(name, _)| {
            body.output.iter().any(|o| o.name == name.to_string() && o.type_proto.as_ref().map_or(false, |t| {
                t.tensor_type.as_ref().map_or(false, |tt| tt.elem_type == DataType::Float as i32)
            }))
        }) {
            if let OrtValue::Tensor { dtype: DataType::Float, data, .. } = &cond_tensor.1 {
                condition = f32::from_le_bytes(data[..4].try_into().unwrap()) != 0.0;
            }
        }

        trip_count += 1;
    }

    // Return scan outputs as a sequence (or final state if no scan outputs)
    if !scan_outputs.is_empty() {
        Ok(OrtValue::Sequence(scan_outputs))
    } else {
        // If no scan outputs, return the final state as a sequence
        Ok(OrtValue::Sequence(state))
    }
}


    
    fn op_scan(&self,node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let body = node.attributes.iter().find(|a| a.name == "body")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| OrtError::InvalidTensorData("Scan requires body subgraph".into()))?;
        let num_scan_inputs = node.attributes.iter().find(|a| a.name == "num_scan_inputs")
            .map(|a| a.i as usize)
            .ok_or_else(|| OrtError::InvalidTensorData("Scan requires num_scan_inputs".into()))?;
        let mut state = inputs[..num_scan_inputs].to_vec();
        let scan_inputs = inputs[num_scan_inputs..].to_vec();
        let mut outputs = vec![];
        for scan_input in scan_inputs {
            let mut subgraph_inputs = HashMap::new();
            for (i, input) in body.input.iter().enumerate() {
                if i < num_scan_inputs {
                    subgraph_inputs.insert(input.name.clone(), state[i].clone());
                } else {
                    subgraph_inputs.insert(input.name.clone(), scan_input.clone());
                }
            }
            let subgraph_outputs = self.execute_subgraph(body, subgraph_inputs)?;
            state = subgraph_outputs.iter().take(num_scan_inputs).map(|(_, v)| v.clone()).collect();
            outputs.extend(subgraph_outputs.into_iter().skip(num_scan_inputs).map(|(_, v)| v));
        }
        Ok(OrtValue::Sequence(outputs))
    }

    fn execute_subgraph(&self,graph: &GraphProto, inputs: HashMap<String, OrtValue>) -> OrtResult<HashMap<String, OrtValue>> {
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

  pub fn infer(&self, inputs: HashMap<String, OrtValue>) -> OrtResult<HashMap<String, OrtValue>> {
    let graph = self.model.graph.as_ref().ok_or(OrtError::InvalidModel)?;
    let mut shape_inference = ShapeInference::new(graph);
    shape_inference.infer_shapes(&inputs)?;

    let mut tensor_map: HashMap<String, OrtValue> = HashMap::new();
    for tensor in &graph.initializer {
        tensor_map.insert(tensor.name.clone(), self.parse_tensor(tensor)?);
    }
    tensor_map.extend(inputs);

    for node in &graph.node {
        if node.output.is_empty() {
            return Err(OrtError::InvalidModel);
        }
        let node_inputs = node
            .input
            .iter()
            .map(|name| tensor_map.get(name).cloned().ok_or_else(|| OrtError::MissingInput("from infer".to_string()+&name.clone())))
            .collect::<OrtResult<Vec<_>>>()?;

        let output = if let Some(op) = self.node_registry.get(&node.op_type) {
            op(node, &node_inputs)?
        } else if(node.op_type=="If"){
            (self.op_if(node,&node_inputs)?)
        }
        else if(node.op_type=="Loop"){
            (self.op_loop(node,&node_inputs)?)
        }
        else if(node.op_type=="Scan"){
            (self.op_scan(node,&node_inputs)?)
        }
        else{
        //     self.node_registry.insert("If".into(), Self::op_if);
        
        // self.node_registry.insert("Loop".into(), Self::op_loop);
        // self.node_registry.insert("Scan".into(), Self::op_scan);

                return Err(OrtError::UnsupportedOp(node.op_type.clone()));
        };
        tensor_map.insert(node.output[0].clone(), output);
    }

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

fn main() -> Result<()> {
    //  OrtEngine::print_model_info("./kokoro-v1.0.onnx")?;
    let engine = OrtEngine::new("./kokoro-v1.0.onnx")?;
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
    let flat_data: Vec<f32> = blended_style.into_iter().flatten().collect();
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

    // Run inference
    println!("Running inference...");
    let outputs = engine.infer(inputs)?;
    
    // Process outputs
    println!("Inference complete. Outputs:");
    for (name, value) in &outputs {
        println!("  - {}: {:?}", name, value);
    }
    
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
   
