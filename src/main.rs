// #![recursion_limit = "5684"]

use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
mod tests {
    pub mod tensor_ops_test;
    pub mod parser_test;
    pub mod onnx_model_test;
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
use ndarray::{Array2, Array3, ArrayD, Axis, Dimension};
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
        }
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
        let epsilon = node.attributes.iter().find(|a| a.name == "epsilon").map(|a| a.f).unwrap_or(1e-5);
        let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("LayerNormalization requires input tensor"))?)?;
        let scale = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("LayerNormalization requires scale tensor"))?)?;
        let bias = inputs.get(2).map(|b| ort_to_ndarray(b)).transpose()?;
        let mean = array.mean_axis(Axis(array.ndim() - 1))
            .ok_or_else(|| OrtError::InvalidTensorData("LayerNormalization mean error".into()))?;
        let var = array.var_axis(Axis(array.ndim() - 1), 0.0);
        let result = (array - mean) / (var + epsilon).mapv(|v| v.sqrt()) * scale;
        let result = if let Some(b) = bias { result + b } else { result };
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

fn op_if(&self,node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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

    fn op_scan(self,node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        } else {
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
     OrtEngine::print_model_info("/root/github/Kokoros/checkpoints/kokoro-v1.0.onnx")?;
    // let engine = OrtEngine::new("/root/github/Kokoros/checkpoints/kokoro-v1.0.onnx")?;
    // let input_data1: Vec<i64> = (0..128).collect();
    // let input_data2: Vec<i64> = (128..256).collect();

    // let tokens = vec![
    //     OrtValue::Tensor {
    //         shape: vec![1, 128],
    //         dtype: DataType::Int64,
    //         data: Arc::new(
    //             input_data1
    //                 .iter()
    //                 .flat_map(|x| x.to_le_bytes())
    //                 .collect::<Vec<u8>>(),
    //         ),
    //     },
    //     OrtValue::Tensor {
    //         shape: vec![1, 128],
    //         dtype: DataType::Int64,
    //         data: Arc::new(
    //             input_data2
    //                 .iter()
    //                 .flat_map(|x| x.to_le_bytes())
    //                 .collect::<Vec<u8>>(),
    //         ),
    //     },
    // ];
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
    Ok(())
}