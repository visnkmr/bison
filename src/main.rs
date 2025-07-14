use anyhow::Result;
use indexmap::IndexMap;
use prost::Message;
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    fmt,
    fs::File,
    io::Read,
    path::Path,
    sync::Arc,
};
use thiserror::Error;

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
pub struct NodeProto {
    pub input: Vec<String>,
    pub output: Vec<String>,
    pub op_type: String,
}

#[derive(Debug, Clone, Default)]
pub struct GraphProto {
    pub node: Vec<NodeProto>,
    pub initializer: Vec<TensorProto>,
    pub output: Vec<ValueInfoProto>,
}

#[derive(Debug, Clone, Default)]
pub struct ValueInfoProto {
    pub name: String,
}

#[derive(Debug, Clone, Default)]
pub struct OpSetImport {
    pub domain: String,
    pub version: i64,
}

#[derive(Debug, Clone, Default)]
pub struct ModelProto {
    pub opset_import: Vec<OpSetImport>,
    pub graph: Option<GraphProto>,
}

impl Message for ModelProto {
    fn encode_raw<B: prost::bytes::BufMut>(&self, _buf: &mut B) {
        unimplemented!("Encoding not needed for this implementation")
    }

    fn merge_field<B: prost::bytes::Buf>(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut B,
        ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
        println!("ModelProto: tag={}, wire_type={:?}", tag, wire_type);
        match tag {
            1 => {
                // Handle potential ir_version (int64, Varint) instead of opset_import
                let mut ir_version = 0i64;
                prost::encoding::int64::merge(wire_type, &mut ir_version, buf, ctx)?;
                // Optionally store ir_version if needed
                println!("ModelProto: Parsed ir_version={}", ir_version);
            }
            8 => {
                // opset_import (repeated OperatorSetIdProto, LengthDelimited)
                let mut opset = OpSetImport::default();
                prost::encoding::message::merge(wire_type, &mut opset, buf, ctx)?;
                self.opset_import.push(opset);
            }
            7 => {
                // graph (GraphProto, LengthDelimited)
                let mut graph = GraphProto::default();
                prost::encoding::message::merge(wire_type, &mut graph, buf, ctx)?;
                self.graph = Some(graph);
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
        println!("GraphProto: tag={}, wire_type={:?}", tag, wire_type);
        match tag {
            1 => {
                let mut node = NodeProto::default();
                prost::encoding::message::merge(wire_type, &mut node, buf, ctx)?;
                self.node.push(node);
            }
            8 => {
                let mut init = TensorProto::default();
                prost::encoding::message::merge(wire_type, &mut init, buf, ctx)?;
                self.initializer.push(init);
            }
            11 => {
                let mut output = ValueInfoProto::default();
                prost::encoding::message::merge(wire_type, &mut output, buf, ctx)?;
                self.output.push(output);
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
        println!("NodeProto: tag={}, wire_type={:?}", tag, wire_type);
        match tag {
            1 => {
                let mut input = String::new();
                prost::encoding::string::merge(wire_type, &mut input, buf, ctx)?;
                self.input.push(input);
            }
            2 => {
                let mut output = String::new();
                prost::encoding::string::merge(wire_type, &mut output, buf, ctx)?;
                self.output.push(output);
            }
            4 => {
                prost::encoding::string::merge(wire_type, &mut self.op_type, buf, ctx)?;
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
        println!("TensorProto: tag={}, wire_type={:?}", tag, wire_type);
        match tag {
            1 => {
                let mut dim = 0i64;
                prost::encoding::int64::merge(wire_type, &mut dim, buf, ctx)?;
                self.dims.push(dim);
            }
            2 => {
                prost::encoding::int32::merge(wire_type, &mut self.data_type, buf, ctx)?;
            }
            4 => {
                prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?;
            }
            7 => {
                prost::encoding::bytes::merge(wire_type, &mut self.raw_data, buf, ctx)?;
            }
            8 => {
                let mut string_data = Vec::new();
                prost::encoding::bytes::merge(wire_type, &mut string_data, buf, ctx)?;
                self.string_data.push(string_data);
            }
            9 => {
                let mut float_data = 0f32;
                prost::encoding::float::merge(wire_type, &mut float_data, buf, ctx)?;
                self.float_data.push(float_data);
            }
            11 => {
                let mut int64_data = 0i64;
                prost::encoding::int64::merge(wire_type, &mut int64_data, buf, ctx)?;
                self.int64_data.push(int64_data);
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
        println!("ValueInfoProto: tag={}, wire_type={:?}", tag, wire_type);
        match tag {
            1 => {
                prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?;
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
        println!("OpSetImport: tag={}, wire_type={:?}", tag, wire_type);
        match tag {
            1 => {
                prost::encoding::string::merge(wire_type, &mut self.domain, buf, ctx)?;
            }
            2 => {
                prost::encoding::int64::merge(wire_type, &mut self.version, buf, ctx)?;
            }
            _ => {
                println!("OpSetImport: Skipping unknown tag: {}", tag);
                prost::encoding::skip_field(wire_type, tag, buf, ctx)?;
            }
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
    #[error("Missing input: {0}")]
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

#[derive(Clone, Serialize)]
pub enum OrtValue {
    Tensor {
        shape: Vec<usize>,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum MapKey {
    String(String),
    Int64(i64),
}

pub struct OrtEngine {
    model: ModelProto,
    node_registry: HashMap<String, fn(&NodeProto, &[OrtValue]) -> OrtResult<OrtValue>>,
    vendor_ops: HashMap<String, fn(&[u8], &[OrtValue]) -> OrtResult<OrtValue>>,
}

impl OrtEngine {
    pub fn new<P: AsRef<Path>>(path: P) -> OrtResult<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let model = ModelProto::decode(&*buffer)?;

        let mut engine = Self {
            model,
            node_registry: HashMap::new(),
            vendor_ops: HashMap::new(),
        };

        engine.register_core_ops();
        engine.register_vendor_ops();
        Ok(engine)
    }

    fn register_core_ops(&mut self) {
        self.node_registry.insert("Add".into(), Self::op_add);
        self.node_registry
            .insert("SequenceConstruct".into(), Self::op_sequence_construct);
        self.node_registry
            .insert("SequenceAt".into(), Self::op_sequence_at);
    }

    fn register_vendor_ops(&mut self) {
        self.vendor_ops.insert(
            "com.custom.Tokenizer".into(),
            |_serialized_attrs, _inputs| {
                Ok(OrtValue::Tensor {
                    shape: vec![],
                    dtype: DataType::Int64,
                    data: Arc::new(vec![]),
                })
            },
        );
    }

    fn op_add(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let (shape1, _data1, _data2) = match (inputs.get(0), inputs.get(1)) {
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

        // Placeholder for actual addition logic
        Ok(OrtValue::Tensor {
            shape: shape1.clone(),
            dtype: DataType::Float,
            data: Arc::new(vec![]), // Replace with actual computation
        })
    }

    fn op_sequence_construct(_: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        Ok(OrtValue::Sequence(inputs.to_vec()))
    }

    fn op_sequence_at(_: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        let sequence = match inputs.first() {
            Some(OrtValue::Sequence(s)) => s,
            _ => return Err(OrtError::TypeMismatch("Expected sequence")),
        };

        let index = match inputs.get(1) {
            Some(OrtValue::Tensor {
                dtype: DataType::Int64,
                data,
                ..
            }) => {
                if data.len() >= 8 {
                    let bytes = &data[..8];
                    i64::from_le_bytes(bytes.try_into().map_err(|_| {
                        OrtError::InvalidTensorData("Invalid index tensor data".into())
                    })?) as usize
                } else {
                    return Err(OrtError::InvalidTensorData("Index tensor too small".into()));
                }
            }
            _ => return Err(OrtError::TypeMismatch("Expected index tensor")),
        };

        sequence
            .get(index)
            .cloned()
            .ok_or(OrtError::IndexError("Sequence index out of range"))
    }

    pub fn infer(&self, inputs: HashMap<String, OrtValue>) -> OrtResult<HashMap<String, OrtValue>> {
        let graph = self.model.graph.as_ref().ok_or(OrtError::InvalidModel)?;
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
                .map(|name| {
                    tensor_map
                        .get(name)
                        .cloned()
                        .ok_or_else(|| OrtError::MissingInput(name.clone()))
                })
                .collect::<OrtResult<Vec<_>>>()?;

            let output = if let Some(op) = self.node_registry.get(&node.op_type) {
                op(node, &node_inputs)?
            } else if let Some(vendor_op) = self.vendor_ops.get(&node.op_type) {
                // No attributes needed for current vendor ops
                vendor_op(&[], &node_inputs)?
            } else {
                return Err(OrtError::UnsupportedOp(node.op_type.clone()));
            };

            tensor_map.insert(node.output[0].clone(), output);
        }

        graph
            .output
            .iter()
            .map(|output| {
                tensor_map
                    .get(&output.name)
                    .cloned()
                    .ok_or_else(|| OrtError::MissingOutput(output.name.clone()))
                    .map(|v| (output.name.clone(), v))
            })
            .collect()
    }

    fn parse_tensor(&self, proto: &TensorProto) -> OrtResult<OrtValue> {
        let shape: Vec<usize> = proto.dims.iter().map(|d| *d as usize).collect();
        let total_elements = shape.iter().product::<usize>();

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

        // Print opcode version (opset_import)
        println!("Opcode Versions:");
        for opset in &model.opset_import {
            println!(
                "  Domain: {}, Version: {}",
                opset.domain,
                opset.version
            );
        }

        // Collect unique operations
        let graph = model.graph.as_ref().ok_or(OrtError::InvalidModel)?;
        let mut core_ops = HashSet::new();
        let mut vendor_ops = HashSet::new();

        for node in &graph.node {
            if node.op_type.contains('.') {
                vendor_ops.insert(node.op_type.clone());
            } else {
                core_ops.insert(node.op_type.clone());
            }
        }

        // Print unique operations
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
    // Run the test function to print model info
    OrtEngine::print_model_info("/root/github/Kokoros/checkpoints/kokoro-v1.0.onnx")?;

    // let engine = OrtEngine::new("model.onnx")?;

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
    // let seq_input = OrtValue::Sequence(tokens);

    // let mut inputs = HashMap::new();
    // inputs.insert("token_sequence".to_string(), seq_input);

    // let outputs = engine.infer(inputs)?;

    // match outputs.get("predictions") {
    //     Some(OrtValue::Map(result_map)) => {
    //         for (key, value) in result_map {
    //             println!("Key: {:?}, Value: {:?}", key, value);
    //         }
    //     }
    //     Some(_) => println!("Error: 'predictions' output is not a Map"),
    //     None => println!("Error: 'predictions' output not found"),
    // }
    Ok(())
}