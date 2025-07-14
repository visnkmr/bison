// #![recursion_limit = "5684"]


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
use ndarray::{ArrayD, Axis, Array2, Array3};
use num_traits::Float;

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
}

#[derive(Debug, Clone, Default)]
pub struct NodeProto {
    pub input: Vec<String>,
    pub output: Vec<String>,
    pub op_type: String,
    pub attributes: Vec<AttributeProto>,
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
    pub ir_version: i64,
    pub opset_import: Vec<OpSetImport>,
    pub graph: Option<GraphProto>,
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
            1 => prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?,
            2 => prost::encoding::int32::merge(wire_type, &mut self.data_type, buf, ctx)?,
            4 => {
                let mut dim = 0i64;
                prost::encoding::int64::merge(wire_type, &mut dim, buf, ctx)?;
                self.dims.push(dim);
            }
            5 => {
                let mut f = 0.0f32;
                prost::encoding::float::merge(wire_type, &mut f, buf, ctx)?;
                self.float_data.push(f);
            }
            6 => {
                let mut i = 0i64;
                prost::encoding::int64::merge(wire_type, &mut i, buf, ctx)?;
                self.int64_data.push(i);
            }
            7 => {
                let mut s = Vec::new();
                prost::encoding::bytes::merge(wire_type, &mut s, buf, ctx)?;
                self.string_data.push(s);
            }
            8 => prost::encoding::bytes::merge(wire_type, &mut self.raw_data, buf, ctx)?,
            _ => prost::encoding::skip_field(wire_type, tag, buf, ctx)?,
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
            1 => prost::encoding::string::merge(wire_type, &mut self.name, buf, ctx)?,
            _ => prost::encoding::skip_field(wire_type, tag, buf, ctx)?,
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
                prost::encoding::int64::merge(wire_type, &mut self.ir_version, buf, ctx)?;
            }
            8 => {
                let mut opset = OpSetImport::default();
                prost::encoding::message::merge(wire_type, &mut opset, buf, ctx)?;
                self.opset_import.push(opset);
            }
            7 => {
                let mut graph = GraphProto::default();
                prost::encoding::message::merge(wire_type, &mut graph, buf, ctx)?;
                self.graph = Some(graph);
            }
            _ => prost::encoding::skip_field(wire_type, tag, buf, ctx)?,
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
            _ => prost::encoding::skip_field(wire_type, tag, buf, ctx)?,
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
            3 => {
                let mut attr = AttributeProto::default();
                prost::encoding::message::merge(wire_type, &mut attr, buf, ctx)?;
                self.attributes.push(attr);
            }
            _ => prost::encoding::skip_field(wire_type, tag, buf, ctx)?,
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
            }
            2 => {
                prost::encoding::int64::merge(wire_type, &mut self.i, buf, ctx)?;
            }
            3 => {
                prost::encoding::float::merge(wire_type, &mut self.f, buf, ctx)?;
            }
            4 => {
                prost::encoding::bytes::merge(wire_type, &mut self.s, buf, ctx)?;
            }
            5 => {
                let mut tensor = TensorProto::default();
                prost::encoding::message::merge(wire_type, &mut tensor, buf, ctx)?;
                self.t = Some(tensor);
            }
            6 => {
                let mut i = 0i64;
                prost::encoding::int64::merge(wire_type, &mut i, buf, ctx)?;
                self.ints.push(i);
            }
            7 => {
                let mut f = 0.0f32;
                prost::encoding::float::merge(wire_type, &mut f, buf, ctx)?;
                self.floats.push(f);
            }
            _ => prost::encoding::skip_field(wire_type, tag, buf, ctx)?,
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

impl OrtValue {
    pub fn shape(&self) -> &Vec<usize> {
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

pub struct OrtEngine {
    model: ModelProto,
    node_registry: HashMap<String, fn(&NodeProto, &[OrtValue]) -> OrtResult<OrtValue>>,
    vendor_ops: HashMap<String, fn(&[u8], &[OrtValue]) -> OrtResult<OrtValue>>,
}

// Helper function to convert OrtValue to ndarray
fn ort_to_ndarray(ort: &OrtValue) -> OrtResult<ArrayD<f32>> {
    match ort {
        OrtValue::Tensor { shape, dtype: DataType::Float, data, .. } => {
            let float_data: Vec<f32> = data
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            ArrayD::from_shape_vec(shape.clone(), float_data)
                .map_err(|_| OrtError::InvalidTensorData("Shape mismatch".into()))
        }
        _ => Err(OrtError::TypeMismatch("Expected float tensor")),
    }
}

// Helper function to convert ndarray to OrtValue
fn ndarray_to_ort(array: ArrayD<f32>, dtype: DataType) -> OrtValue {
    let shape = array.shape().to_vec();
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

        let mut engine = Self {
            model,
            node_registry: HashMap::new(),
            vendor_ops: HashMap::new(),
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
            shape,
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
                    if axis < shape.len() && shape[axis] == 1 {
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
                    new_shape.insert(axis, 1);
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
        let data: Vec<u8> = shape.iter().map(|&s| s as i64).flat_map(|s| s.to_le_bytes()).collect();
        Ok(OrtValue::Tensor {
            shape: vec![shape.len()],
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
    fn op_scatter_nd(_node: &NodeProto, _inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        Err(OrtError::UnsupportedOp("ScatterND not implemented".into()))
    }

    fn op_nonzero(_node: &NodeProto, _inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        Err(OrtError::UnsupportedOp("NonZero not implemented".into()))
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
    fn op_cumsum(_node: &NodeProto, _inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        Err(OrtError::UnsupportedOp("CumSum not implemented".into()))
    }

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
            shape: vec![range.len()],
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

    // Complex Operations (Placeholders)
    fn op_conv(_node: &NodeProto, _inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        Err(OrtError::UnsupportedOp("Conv not implemented".into()))
    }

    fn op_conv_transpose(_node: &NodeProto, _inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        Err(OrtError::UnsupportedOp("ConvTranspose not implemented".into()))
    }

    fn op_lstm(_node: &NodeProto, _inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        Err(OrtError::UnsupportedOp("LSTM not implemented".into()))
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

    fn op_stft(_node: &NodeProto, _inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        Err(OrtError::UnsupportedOp("STFT not implemented".into()))
    }

    fn op_resize(_node: &NodeProto, _inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        Err(OrtError::UnsupportedOp("Resize not implemented".into()))
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

        for node in &graph.node {
            if node.op_type.contains('.') {
                vendor_ops.insert(node.op_type.clone());
            } else {
                core_ops.insert(node.op_type.clone());
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
    OrtEngine::print_model_info("model.onnx")?;
    let engine = OrtEngine::new("model.onnx")?;
    let input_data1: Vec<i64> = (0..128).collect();
    let input_data2: Vec<i64> = (128..256).collect();

    let tokens = vec![
        OrtValue::Tensor {
            shape: vec![1, 128],
            dtype: DataType::Int64,
            data: Arc::new(
                input_data1
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
        },
        OrtValue::Tensor {
            shape: vec![1, 128],
            dtype: DataType::Int64,
            data: Arc::new(
                input_data2
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
        },
    ];
    let seq_input = OrtValue::Sequence(tokens);
    let mut inputs = HashMap::new();
    inputs.insert("token_sequence".to_string(), seq_input);
    let outputs = engine.infer(inputs)?;
    match outputs.get("predictions") {
        Some(OrtValue::Map(result_map)) => {
            for (key, value) in result_map {
                println!("Key: {:?}, Value: {:?}", key, value);
            }
        }
        Some(_) => println!("Error: 'predictions' output is not a Map"),
        None => println!("Error: 'predictions' output not found"),
    }
    Ok(())
}