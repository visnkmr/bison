// core_ops.rs - Implementation of core operations
// This file contains implementations of functions declared in main.rs

use crate::*;

// Implementations of functions will go here

impl OrtEngine{
    pub fn op_resize(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        pub fn op_stft(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        pub fn op_lstm(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        pub fn op_conv_transpose(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        pub fn op_conv(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        pub fn op_scatter_nd(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    pub fn op_nonzero(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    
        pub fn op_cumsum(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        pub fn op_add(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        pub fn op_sub(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        pub fn op_mul(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        pub fn op_div(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        pub fn op_pow(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        pub fn op_sqrt(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Sqrt requires one float tensor"))?)?;
            let result = array.mapv(|a| a.sqrt());
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        // Element-wise Operations
        pub fn op_exp(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Exp requires one float tensor"))?)?;
            let result = array.mapv(|a| a.exp());
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_sigmoid(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Sigmoid requires one float tensor"))?)?;
            let result = array.mapv(|a| 1.0 / (1.0 + (-a).exp()));
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_tanh(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Tanh requires one float tensor"))?)?;
            let result = array.mapv(|a| a.tanh());
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_sin(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Sin requires one float tensor"))?)?;
            let result = array.mapv(|a| a.sin());
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_cos(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Cos requires one float tensor"))?)?;
            let result = array.mapv(|a| a.cos());
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_atan(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Atan requires one float tensor"))?)?;
            let result = array.mapv(|a| a.atan());
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_floor(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Floor requires one float tensor"))?)?;
            let result = array.mapv(|a| a.floor());
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_round(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Round requires one float tensor"))?)?;
            let result = array.mapv(|a| a.round());
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_leaky_relu(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let alpha = node.attributes.iter().find(|a| a.name == "alpha").map(|a| a.f).unwrap_or(0.01);
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("LeakyRelu requires one float tensor"))?)?;
            let result = array.mapv(|a| if a >= 0.0 { a } else { alpha * a });
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        // Comparison Operations
        pub fn op_less(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Less requires two float tensors"))?)?;
            let array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Less requires two float tensors"))?)?;
            let result = ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| if a < b { 1.0 } else { 0.0 });
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_greater(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Greater requires two float tensors"))?)?;
            let array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Greater requires two float tensors"))?)?;
            let result = ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| if a > b { 1.0 } else { 0.0 });
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_equal(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Equal requires two float tensors"))?)?;
            let array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Equal requires two float tensors"))?)?;
            let result = ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| if a == b { 1.0 } else { 0.0 });
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_greater_or_equal(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("GreaterOrEqual requires two float tensors"))?)?;
            let array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("GreaterOrEqual requires two float tensors"))?)?;
            let result = ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| if a >= b { 1.0 } else { 0.0 });
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_and(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let array1 = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("And requires two float tensors"))?)?;
            let array2 = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("And requires two float tensors"))?)?;
            let result = ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 });
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        // Shape Manipulation Operations
        pub fn op_reshape(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        pub fn op_squeeze(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        pub fn op_unsqueeze(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        pub fn op_transpose(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let perm = node.attributes.iter().find(|a| a.name == "perm")
                .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<_>>())
                .unwrap_or_else(|| (0..inputs[0].shape().len()).collect());
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Transpose requires one float tensor"))?)?;
            let result = array.permuted_axes(perm);
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        pub fn op_concat(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        pub fn op_shape(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        pub fn op_constant_of_shape(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        pub fn op_expand(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        pub fn op_reduce_sum(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        pub fn op_reduce_mean(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        
    pub fn op_matmul(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_sequence_at(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_sequence_construct(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        // SequenceConstruct creates a sequence from input tensors
        if inputs.is_empty() {
            return Err(OrtError::InvalidTensorData("SequenceConstruct requires at least one input tensor".into()));
        }
        
        Ok(OrtValue::Sequence(inputs.to_vec()))
    }
    
    pub fn op_sequence_empty(node: &NodeProto, _inputs: &[OrtValue]) -> OrtResult<OrtValue> {
        // SequenceEmpty creates an empty sequence
        // Optional attribute 'dtype' specifies the element type (not used in this implementation)
        let _dtype = node.attributes.iter()
            .find(|a| a.name == "dtype")
            .map(|a| a.i)
            .unwrap_or(1); // Default to float
            
        Ok(OrtValue::Sequence(Vec::new()))
    }
    
    pub fn op_sequence_erase(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_sequence_insert(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_sequence_length(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    pub fn op_map_from_tensor(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_map_to_tensor(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_map_get(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_map_has_key(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_map_keys(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    pub fn op_erf(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_gelu(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_split(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_dropout(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_einsum(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_topk(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_gather_elements(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_gather_nd(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_reduce_max(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_attention(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    pub fn op_embedding(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_constant_of_shape_int64(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_layer_normalization_with_epsilon(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    
    pub fn op_position_embeddings(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_token_type_embeddings(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_bert_attention(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_bert_intermediate(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_bert_output(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_bert_pooler(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        pub fn op_gemm(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        pub fn op_gather(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_slice(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
       
    
        pub fn op_where(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let condition = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Where requires condition tensor"))?)?;
            let x = ort_to_ndarray(inputs.get(1).ok_or_else(|| OrtError::TypeMismatch("Where requires x tensor"))?)?;
            let y = ort_to_ndarray(inputs.get(2).ok_or_else(|| OrtError::TypeMismatch("Where requires y tensor"))?)?;
            let result = ndarray::Zip::from(&condition).and(&x).and(&y)
                .map_collect(|&c, &x, &y| if c != 0.0 { x } else { y });
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        // Softmax
        pub fn op_softmax(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        pub fn op_range(_node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
        pub fn op_clip(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
            let min = node.attributes.iter().find(|a| a.name == "min").map(|a| a.f).unwrap_or(f32::NEG_INFINITY);
            let max = node.attributes.iter().find(|a| a.name == "max").map(|a| a.f).unwrap_or(f32::INFINITY);
            let array = ort_to_ndarray(inputs.get(0).ok_or_else(|| OrtError::TypeMismatch("Clip requires one float tensor"))?)?;
            let result = array.mapv(|a| a.clamp(min, max));
            Ok(ndarray_to_ort(result, DataType::Float))
        }
    
        // Cast
        pub fn op_cast(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    
        pub fn op_layer_normalization(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
       pub fn op_pad(node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    pub fn op_if(&self, node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
       pub fn op_loop(&self, node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
    
        
        pub fn op_scan(&self,node: &NodeProto, inputs: &[OrtValue]) -> OrtResult<OrtValue> {
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
    
        
    
    
}