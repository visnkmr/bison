use crate::{OrtValue, DataType, Dimensions, OrtResult, OrtEngine};
use std::collections::HashMap;
use std::sync::Arc;

/// Function to directly infer using token IDs, style embedding, and speed
pub fn infer_with_tokens_and_style(
    engine: &OrtEngine,
    tokens: Vec<Vec<i64>>,
    style_embedding: Vec<Vec<f32>>,
    speed: f32,
) -> OrtResult<HashMap<String, OrtValue>> {
    // Create inputs map
    let mut inputs = HashMap::new();
    
    // Add token IDs
    let token_shape = vec![
        Dimensions::Fixed(tokens.len()),
        Dimensions::Fixed(tokens[0].len()),
    ];
    
    // Flatten tokens for storage
    let mut token_data = Vec::new();
    for token_seq in &tokens {
        token_data.extend(token_seq.iter().map(|&x| x.to_le_bytes()).flatten());
    }
    
    // Create token tensor
    let token_tensor = OrtValue::Tensor {
        shape: token_shape,
        dtype: DataType::Int64,
        data: Arc::new(token_data),
    };
    
    // Add tokens to inputs
    inputs.insert("tokens".to_string(), token_tensor);
    
    // Add style embedding
    let embedding_shape = vec![
        Dimensions::Fixed(style_embedding.len()),
        Dimensions::Fixed(style_embedding[0].len()),
    ];
    
    // Flatten style embedding for storage
    let mut embedding_data = Vec::new();
    for embedding_vec in &style_embedding {
        embedding_data.extend(embedding_vec.iter().map(|&x| x.to_le_bytes()).flatten());
    }
    
    // Create embedding tensor
    let embedding_tensor = OrtValue::Tensor {
        shape: embedding_shape,
        dtype: DataType::Float,
        data: Arc::new(embedding_data),
    };
    
    // Add embedding to inputs
    inputs.insert("styles".to_string(), embedding_tensor);
    
    // Add speed parameter
    let speed_shape = vec![Dimensions::Fixed(1)];
    let speed_data = Arc::new(speed.to_le_bytes().to_vec());
    
    let speed_tensor = OrtValue::Tensor {
        shape: speed_shape,
        dtype: DataType::Float,
        data: speed_data,
    };
    
    // Add speed to inputs
    inputs.insert("speed".to_string(), speed_tensor);
    
    // Run inference
    engine.infer(inputs)
}

/// Helper function to convert your example data to the right format
pub fn run_example_inference(model_path: &str) -> OrtResult<HashMap<String, OrtValue>> {
    // Load the model
    let engine = OrtEngine::new(model_path)?;
    
    // Your example data
    let tokens = vec![vec![0, 50, 156, 43, 102, 4, 0]];
    
    let style_embedding = vec![vec![-0.16746138, 0.106833816, -0.17197946, -0.17930198, -0.4060307, 0.11337316, -0.05904325, -0.13578473, -0.343865, -0.0030500141, -0.058186237, -0.18617716, 0.3655906, 0.1500281, 0.0323276, -0.2660883, -0.021834578, -0.18887411, 0.15604171, -0.17936222, -0.21674247, -0.08793214, -0.014403321, -0.038582608, 0.005953279, 0.30037892, -0.25818214, 0.14401352, 0.00625191, 0.18139648, 0.1905407, -0.30535796, -0.016582137, -0.06380315, 0.19268999, 0.031495668, -0.10360171, -0.07843726, 0.035174046, 0.047639426, 0.09471621, -0.059944917, 0.07799803, 0.42816967, -0.27074027, -0.059864923, 0.094025224, -0.07608084, -0.009240143, 0.2764985, -0.044961445, -0.22325265, 0.28969276, 0.021382106, 0.09409301, 0.3064245, 0.085562065, -0.018245282, -0.12442948, 0.12522374, 0.20399052, -0.07992236, -0.17870936, -0.03290955, 0.20011769, 0.23295887, -0.0011655795, 0.2106421, 0.029463217, 0.049337372, 0.07007421, 0.06657779, 0.12671578, -0.3048649, -0.17952333, -0.20896465, 0.010621702, 0.16129294, 0.24825078, -0.06730439, 0.14417285, 0.14019054, -0.16492297, 0.07709213, 0.18941414, 0.07108727, -0.16543987, -0.1864754, -0.25925547, -0.011538826, 0.12039098, 0.024524461, 0.09829027, -0.020422952, -0.19386753, -0.13779366, 0.06404631, -0.091026954, 0.1432159, -0.1445843, -0.099253185, -0.27379233, 0.07603142, -0.06384298, 0.20024501, 0.14540523, 0.010894625, 0.18515547, 0.23194641, -0.07801862, -0.03515421, 0.005198706, 0.11977995, 0.028442672, -0.26251578, 0.087687396, -0.09812868, -0.021395776, 0.17591082, 0.00079514645, -0.037736632, 0.16991898, 0.020198015, 0.29645926, 0.21168791, -0.37216398, 0.13653347, -0.06943156, -0.014739413, 0.16784102, 0.48688984, 0.10855578, -0.25430948, -0.13242087, 0.36683533, 0.0017357357, -0.3956462, -0.27680144, 0.29430857, -0.09608546, 0.10188929, -0.1437357, 0.26491192, -0.07434953, 0.2738349, 0.074040905, -0.15176898, -0.13395815, 0.3927017, -0.14603326, 0.26794004, 0.06925736, -0.111301675, 0.45458955, -0.21831812, -0.15351343, -0.14352655, 0.2463764, 0.59878033, -0.28609738, 0.21620028, 0.16584155, 0.26237804, 0.639141, 0.48741198, 0.28353006, 0.20943506, -0.005696906, 0.0027122672, -0.2647833, 0.20146331, 0.7051931, -0.33182484, 0.12572102, -0.18048556, -0.886673, -0.18763334, 0.11108457, 0.04415555, -0.4453653, 0.7829914, 0.23367575, 0.07653396, -0.058281526, 0.63499576, -0.12139675, 0.10016927, -0.24464339, -0.169406, -0.37613553, -0.0048745424, -0.05477307, 0.21715853, 0.44753513, 0.08324612, -0.34354436, 0.20547722, 0.14335431, 0.15277404, 0.137537, 0.014170506, -0.48911935, -0.35340762, 0.09423898, 0.56586313, 0.21005873, 0.3004918, 0.20001253, -0.21485168, 0.2627742, -0.077053934, -0.22529292, -0.18517601, -0.077634186, 0.13139398, -0.22406033, 0.2564357, -0.32308036, -0.49612147, 0.6047083, 0.04769512, 0.13776457, -0.45326835, -0.01648686, -0.36213535, -0.12455494, 0.23899081, -0.019421533, -0.11391652, -0.010070331, 0.17823072, 0.12505603, -0.19111425, 0.36101788, 0.35537708, -0.31394705, 0.05328191, -0.30892166, -0.11983204, 0.2664771, -0.0821251, 0.28563684, 0.13965864, 0.5084046, -0.09097928, 0.4581191, -0.094962835, 0.6173904, -0.07400553, -0.17739949, -0.12363535, 0.5033887, 0.23096946, -0.11120826, -0.12198155, -0.10832049]];
    
    let speed = 1.0;
    
    // Run inference
    infer_with_tokens_and_style(&engine, tokens, style_embedding, speed)
}

#[test]
fn tryop(){
    let results = run_example_inference("kokoro-v1.0.onnx");
    println!("{:?}",results);

}