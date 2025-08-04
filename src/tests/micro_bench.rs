// Micro-benchmark and correctness scaffolding for core ops.
// IMPORTANT: This file provides benchmark harness functions but does not run them.
// Users can hook these into criterion or custom runners if desired.
//
// Scope:
// - Correctness spot checks vs ndarray reference for selected ops
// - Timing scaffolding to measure op latency with different shapes/dtypes
//
// Notes:
// - Uses only stock Rust + project deps (ndarray, rayon). No external benchmark crate wired.
// - All functions are behind #[cfg(test)], but they will not auto-run unless explicitly invoked in a test.
// - Designed so you can later plug into criterion or a binary to execute and record timings.

#![allow(dead_code)]

use std::{sync::Arc, time::Instant, collections::HashMap};
use ndarray::{ArrayD, IxDyn, Array2, Array3, Array4};
use crate::*;
use crate::convert::{ndarray_to_ort, ort_to_ndarray, ArrayDResult};

#[cfg(test)]
mod micro_bench {
    use super::*;

    fn mk_tensor_f32(shape: &[usize], fill: f32) -> OrtValue {
        let total: usize = shape.iter().product();
        let data: Vec<f32> = (0..total).map(|i| fill + i as f32 * 0.001).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        OrtValue::Tensor {
            shape: shape.iter().copied().map(Dimensions::Fixed).collect(),
            dtype: DataType::Float,
            data: Arc::new(bytes),
        }
    }

    fn mk_tensor_i32(shape: &[usize], base: i32) -> OrtValue {
        let total: usize = shape.iter().product();
        let data: Vec<i32> = (0..total).map(|i| base + i as i32).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        OrtValue::Tensor {
            shape: shape.iter().copied().map(Dimensions::Fixed).collect(),
            dtype: DataType::Int32,
            data: Arc::new(bytes),
        }
    }

    fn mk_tensor_i64(shape: &[usize], base: i64) -> OrtValue {
        let total: usize = shape.iter().product();
        let data: Vec<i64> = (0..total).map(|i| base + i as i64).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        OrtValue::Tensor {
            shape: shape.iter().copied().map(Dimensions::Fixed).collect(),
            dtype: DataType::Int64,
            data: Arc::new(bytes),
        }
    }

    fn bench_op<F>(label: &str, iters: usize, mut f: F) -> u128
    where
        F: FnMut() -> OrtResult<OrtValue>,
    {
        // warmup
        for _ in 0..(iters.min(5)) {
            let _ = f();
        }
        let start = Instant::now();
        for _ in 0..iters {
            let _ = f();
        }
        start.elapsed().as_micros()
    }

    // Helper to create a NodeProto with attribute (axis, ints vec etc.)
    fn node_with_axis(axis: i64) -> NodeProto {
        let mut n = NodeProto::default();
        n.attributes.push(AttributeProto { name: "axis".to_string(), i: axis, ..Default::default() });
        n
    }

    // -----------------------------
    // Correctness spot checks
    // -----------------------------

    #[test]
    fn correctness_add_float_small() {
        let a = mk_tensor_f32(&[2, 3], 1.0);
        let b = mk_tensor_f32(&[2, 3], 2.0);
        let node = NodeProto::default();
        let out = OrtEngine::op_add(&node, &[a, b]).expect("add failed");
        let ad = ort_to_ndarray(&out).expect("to nd failed");
        match ad {
            ArrayDResult::Float(arr) => {
                assert_eq!(arr.shape(), &[2, 3]);
                // simple invariants
                assert!(arr[[0, 0]] > 2.9 && arr[[0, 0]] < 3.1);
            }
            _ => panic!("expected float output"),
        }
    }

    #[test]
    fn correctness_mul_int32_small() {
        let a = mk_tensor_i32(&[2, 2], 1);
        let b = mk_tensor_i32(&[2, 2], 2);
        let node = NodeProto::default();
        let out = OrtEngine::op_mul(&node, &[a, b]).expect("mul failed");
        let ad = ort_to_ndarray(&out).expect("to nd failed");
        match ad {
            ArrayDResult::Int32(arr) => {
                assert_eq!(arr.shape(), &[2, 2]);
                // values should be monotonically increasing and > 0
                for v in arr.iter() { assert!(*v >= 2); }
            }
            _ => panic!("expected int32 output"),
        }
    }

    #[test]
    fn correctness_softmax_axis_last() {
        let input = mk_tensor_f32(&[3], 0.5);
        let mut node = NodeProto::default();
        node.attributes.push(AttributeProto { name: "axis".to_string(), i: -1, ..Default::default() });
        let out = OrtEngine::op_softmax(&node, &[input]).expect("softmax failed");
        if let ArrayDResult::Float(arr) = ort_to_ndarray(&out).unwrap() {
            let sum_exp: f32 = arr.iter().map(|x| x.exp()).sum();
            assert!((sum_exp - 1.0).abs() < 1e-5);
        } else { panic!("expected float"); }
    }

    #[test]
    fn correctness_concat_axis0() {
        let a = mk_tensor_f32(&[2, 4], 0.0);
        let b = mk_tensor_f32(&[3, 4], 1.0);
        let mut node = NodeProto::default();
        node.attributes.push(AttributeProto { name: "axis".to_string(), i: 0, ..Default::default() });
        let out = OrtEngine::op_concat(&node, &[a, b]).expect("concat failed");
        if let ArrayDResult::Float(arr) = ort_to_ndarray(&out).unwrap() {
            assert_eq!(arr.shape(), &[5, 4]);
        } else { panic!("expected float"); }
    }

    #[test]
    fn correctness_gather_axis0() {
        // data [3,4], indices [2]
        let mut data = OrtValue::Tensor { shape: vec![Dimensions::Fixed(3), Dimensions::Fixed(4)], dtype: DataType::Float, data: Arc::new(vec![]) };
        // fill contiguous 0..12
        {
            let vals: Vec<f32> = (0..12).map(|x| x as f32).collect();
            let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
            if let OrtValue::Tensor { data, .. } = &mut data { *Arc::make_mut(data) = bytes; }
        }
        let indices = mk_tensor_i64(&[1], 2);
        let mut node = NodeProto::default();
        node.attributes.push(AttributeProto { name: "axis".to_string(), i: 0, ..Default::default() });
        let out = OrtEngine::op_gather(&node, &[data, indices]).expect("gather failed");
        if let ArrayDResult::Float(arr) = ort_to_ndarray(&out).unwrap() {
            assert_eq!(arr.shape(), &[1, 4]);
            assert_eq!(arr[[0, 0]], 8.0);
            assert_eq!(arr[[0, 3]], 11.0);
        } else { panic!("expected float"); }
    }

    // -----------------------------
    // Timing scaffolding
    // -----------------------------

    fn bench_add_float() {
        let sizes = &[
            (vec![1, 256], 50_000usize),
            (vec![64, 64], 10_000usize),
            (vec![32, 1024], 5_000usize),
            (vec![8, 512, 512], 200usize),
        ];
        for (shape, iters) in sizes {
            let a = mk_tensor_f32(shape, 0.1);
            let b = mk_tensor_f32(shape, 0.2);
            let node = NodeProto::default();
            let micros = bench_op("Add", *iters, || OrtEngine::op_add(&node, &[a.clone(), b.clone()]));
            println!("[Add f32] shape={:?} iters={} time_us={}", shape, iters, micros);
        }
    }

    fn bench_mul_float() {
        let sizes = &[
            (vec![1, 256], 50_000usize),
            (vec![64, 64], 10_000usize),
            (vec![8, 512, 512], 200usize),
        ];
        for (shape, iters) in sizes {
            let a = mk_tensor_f32(shape, 0.1);
            let b = mk_tensor_f32(shape, 0.2);
            let node = NodeProto::default();
            let micros = bench_op("Mul", *iters, || OrtEngine::op_mul(&node, &[a.clone(), b.clone()]));
            println!("[Mul f32] shape={:?} iters={} time_us={}", shape, iters, micros);
        }
    }

    fn bench_softmax_float() {
        let sizes = &[
            (vec![1024], 10_000usize),
            (vec![32, 512], 2_000usize),
            (vec![8, 128, 128], 200usize),
        ];
        for (shape, iters) in sizes {
            let x = mk_tensor_f32(shape, 0.0);
            let mut node = NodeProto::default();
            node.attributes.push(AttributeProto { name: "axis".to_string(), i: -1, ..Default::default() });
            let micros = bench_op("Softmax", *iters, || OrtEngine::op_softmax(&node, &[x.clone()]));
            println!("[Softmax f32] shape={:?} iters={} time_us={}", shape, iters, micros);
        }
    }

    fn bench_concat_float() {
        let cases = &[
            (vec![64, 256], vec![64, 256], 1, 2_000usize),
            (vec![32, 512], vec![64, 512], 0, 2_000usize),
        ];
        for (s1, s2, axis, iters) in cases {
            let a = mk_tensor_f32(s1, 0.0);
            let b = mk_tensor_f32(s2, 1.0);
            let mut node = NodeProto::default();
            node.attributes.push(AttributeProto { name: "axis".to_string(), i: *axis, ..Default::default() });
            let micros = bench_op("Concat", *iters, || OrtEngine::op_concat(&node, &[a.clone(), b.clone()]));
            println!("[Concat f32] shapes={:?}+{:?} axis={} iters={} time_us={}", s1, s2, axis, iters, micros);
        }
    }

    fn bench_matmul_float() {
        // Simple MxK x KxN cases
        let cases = &[
            (64, 64, 64, 1_000usize),
            (128, 128, 128, 200usize),
            (256, 256, 256, 50usize),
        ];
        for (m, k, n, iters) in cases {
            // Build A [m,k], B [k,n]
            let a = mk_tensor_f32(&[*m, *k], 0.1);
            let b = mk_tensor_f32(&[*k, *n], 0.2);
            let node = NodeProto::default();
            let micros = bench_op("MatMul", *iters, || OrtEngine::op_matmul(&node, &[a.clone(), b.clone()]));
            println!("[MatMul f32] MxK={}x{} KxN={}x{} iters={} time_us={}", m, k, k, n, iters, micros);
        }
    }

    fn bench_gemm_float() {
        // Simple Gemm [M,K], [K,N], bias [N]
        let cases = &[
            (64, 64, 64, 1_000usize),
            (128, 128, 128, 200usize),
        ];
        for (m, k, n, iters) in cases {
            let a = mk_tensor_f32(&[*m, *k], 0.1);
            let b = mk_tensor_f32(&[*k, *n], 0.2);
            let bias = mk_tensor_f32(&[*n], 0.0);
            let mut node = NodeProto::default();
            // default alpha=1, beta=1, transA/B=0
            let micros = bench_op("Gemm", *iters, || OrtEngine::op_gemm(&node, &[a.clone(), b.clone(), bias.clone()]));
            println!("[Gemm f32] MxK={}x{} KxN={}x{} iters={} time_us={}", m, k, k, n, iters, micros);
        }
    }

    fn bench_conv_float() {
        // NCHW: [N,C,H,W], W: [M,C,kH,kW]
        let cases = &[
            ((1, 16, 64, 64), (16, 16, 3, 3), 1_0usize),
            ((1, 32, 64, 64), (32, 32, 3, 3), 10usize),
        ];
        for ((n,c,h,w), (m, _wc, kh, kw), iters) in cases {
            let x = mk_tensor_f32(&[*n as usize, *c as usize, *h as usize, *w as usize], 0.01);
            let w = mk_tensor_f32(&[*m as usize, *c as usize, *kh as usize, *kw as usize], 0.02);
            let mut node = NodeProto::default();
            node.attributes.push(AttributeProto { name: "strides".to_string(), ints: vec![1,1], ..Default::default() });
            node.attributes.push(AttributeProto { name: "dilations".to_string(), ints: vec![1,1], ..Default::default() });
            node.attributes.push(AttributeProto { name: "auto_pad".to_string(), s: "VALID".as_bytes().to_vec(), ..Default::default() });
            let micros = bench_op("Conv", *iters, || OrtEngine::op_conv_optimized(&node, &[x.clone(), w.clone()]));
            println!("[Conv f32] NCHW=({},{},{},{}) K=({},{},{},{}) iters={} time_us={}", n,c,h,w,m,c,kh,kw,iters,micros);
        }
    }

    fn bench_resize_float() {
        // Input [N,C,H,W], use scales mode
        let x = mk_tensor_f32(&[1, 1, 64, 64], 1.0);
        let roi = OrtValue::Tensor { shape: vec![Dimensions::Fixed(0)], dtype: DataType::Float, data: Arc::new(vec![]) };
        let scales_vals = vec![1.0f32, 1.0, 2.0, 2.0];
        let scales_bytes: Vec<u8> = scales_vals.iter().flat_map(|x| x.to_le_bytes()).collect();
        let scales = OrtValue::Tensor { shape: vec![Dimensions::Fixed(4)], dtype: DataType::Float, data: Arc::new(scales_bytes) };
        let mut node = NodeProto::default();
        node.attributes.push(AttributeProto { name: "mode".to_string(), s: "nearest".as_bytes().to_vec(), ..Default::default() });
        node.attributes.push(AttributeProto { name: "coordinate_transformation_mode".to_string(), s: "half_pixel".as_bytes().to_vec(), ..Default::default() });
        let micros = bench_op("Resize", 100, || OrtEngine::op_resize(&node, &[x.clone(), roi.clone(), scales.clone()]));
        println!("[Resize f32] 64->128 nearest half_pixel time_us={}", micros);
    }

    fn bench_lstm_float_small() {
        // Minimal LSTM to ensure path works
        let seq = 4usize; let batch = 1usize; let in_size = 8usize; let hidden = 8usize;
        let x = super::mk_tensor_f32(&[seq, batch, in_size], 0.01);
        let w = super::mk_tensor_f32(&[1, 4*hidden, in_size], 0.02);
        let r = super::mk_tensor_f32(&[1, 4*hidden, hidden], 0.03);
        let mut node = NodeProto::default();
        node.attributes.push(AttributeProto { name: "hidden_size".to_string(), i: hidden as i64, ..Default::default() });
        let micros = bench_op("LSTM", 20, || OrtEngine::op_lstm(&node, &[x.clone(), w.clone(), r.clone()]));
        println!("[LSTM f32] seq={} batch={} in={} hidden={} time_us={}", seq, batch, in_size, hidden, micros);
    }

    fn bench_stft_float_small() {
        // [B, T, C=1], frame_step=16, window len=32
        let x = super::mk_tensor_f32(&[1, 256, 1], 0.0);
        let step = OrtValue::Tensor { shape: vec![], dtype: DataType::Int64, data: Arc::new((16i64).to_le_bytes().to_vec()) };
        let window_vals: Vec<f32> = vec![1.0; 32];
        let window = OrtValue::Tensor { shape: vec![Dimensions::Fixed(32)], dtype: DataType::Float, data: Arc::new(window_vals.iter().flat_map(|v| v.to_le_bytes()).collect()) };
        let frame_len = OrtValue::Tensor { shape: vec![], dtype: DataType::Int64, data: Arc::new((32i64).to_le_bytes().to_vec()) };
        let node = NodeProto::default();
        let micros = bench_op("STFT", 10, || OrtEngine::op_stft_optimized(&node, &[x.clone(), step.clone(), window.clone(), frame_len.clone()]));
        println!("[STFT f32] B=1 T=256 C=1 frame_step=16 frame_len=32 time_us={}", micros);
    }

    // Master function to print all timings when test is executed.
    // It's a test so it can be launched via `cargo test -- --nocapture` if desired.
    #[test]
    fn print_benchmarks_summary() {
        println!("=== Micro Benchmarks (scaffolding) ===");
        bench_add_float();
        bench_mul_float();
        bench_softmax_float();
        bench_concat_float();
        bench_matmul_float();
        bench_gemm_float();
        bench_conv_float();
        bench_resize_float();
        bench_lstm_float_small();
        bench_stft_float_small();
        println!("=== End Benchmarks ===");
    }
}
