use burn::{
    module::Module,
    nn::{
        self, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig, Conv1dRecord},
    },
    tensor::{Tensor, backend::Backend, cast::ToElement},
};

use super::*;

use burn::tensor::Shape;
use npy::{self, NpyData};
use std::error::Error;
use std::io::Read;

fn numpy_to_tensor<B: Backend, const D: usize>(numpy_data: NpyData<f32>) -> Tensor<B, D> {
    let v = numpy_data.to_vec();
    let shape: Shape = v[0..D]
        .iter()
        .map(|&v| v as usize)
        .collect::<Vec<_>>()
        .into();

    let tensor_device_ref = Default::default(); //WgpuDevice::BestAvailable;

    Tensor::<B, D>::from_floats(&v[D..], &tensor_device_ref).reshape::<D, Shape>(shape)
}

fn load_tensor<B: Backend, const D: usize>(
    name: &str,
    path: &str,
) -> Result<Tensor<B, D>, Box<dyn Error>> {
    let tensor_path = format!("{}/{}.npy", path, name);

    let mut buf = vec![];
    std::fs::File::open(tensor_path)?.read_to_end(&mut buf)?;

    let tensor_numpy: NpyData<f32> = NpyData::from_bytes(&buf)?;

    let tensor = numpy_to_tensor(tensor_numpy);

    Ok(tensor)
}

fn load_f32<B: Backend>(name: &str, path: &str) -> Result<f32, Box<dyn Error>> {
    load_tensor::<B, 1>(name, path).map(|t| t.into_scalar().to_f32())
}

fn load_usize<B: Backend>(name: &str, path: &str) -> Result<usize, Box<dyn Error>> {
    load_tensor::<B, 1>(name, path).map(|t| t.into_scalar().to_usize())
}

fn load_linear<B: Backend>(path: &str) -> Result<nn::Linear<B>, Box<dyn Error>> {
    let weight = load_tensor::<B, 2>("weight", path)?;
    let bias = load_tensor::<B, 1>("bias", path).ok();
    let tensor_device_ref = weight.device();

    let record = nn::LinearRecord {
        weight: Param::from_tensor(weight),
        bias: bias.map(|t| Param::from_tensor(t)),
    };

    let linear: nn::Linear<B> = nn::LinearConfig::new(3, 3)
        .init(&tensor_device_ref)
        .load_record(record);
    Ok(linear)
}

fn load_layer_norm<B: Backend>(path: &str) -> Result<nn::LayerNorm<B>, Box<dyn Error>> {
    let weight = Param::from_tensor(load_tensor::<B, 1>("weight", path)?);
    let bias = Param::from_tensor(load_tensor::<B, 1>("bias", path)?);
    let eps = load_f32::<B>("eps", path)? as f64;
    let tensor_device_ref = weight.device();

    let [n_state] = weight.dims();

    let record = nn::LayerNormRecord {
        gamma: weight,
        beta: bias,
        epsilon: <f64 as Module<B>>::into_record(eps),
    };

    let layer_norm: nn::LayerNorm<B> = nn::LayerNormConfig::new(n_state)
        .init(&tensor_device_ref)
        .load_record(record);

    Ok(layer_norm)
}

fn load_multihead_self_attention<B: Backend>(
    path: &str,
) -> Result<MultiHeadSelfAttention<B>, Box<dyn Error>> {
    let query = load_linear(&format!("{}/{}", path, "query"))?;
    let key = load_linear(&format!("{}/{}", path, "key"))?;
    let value = load_linear(&format!("{}/{}", path, "value"))?;
    let out = load_linear(&format!("{}/{}", path, "out"))?;
    let n_head: usize = load_usize::<B>("n_head", path)?;

    // Initializing attention block
    let attention_block = MultiHeadSelfAttention {
        n_head,
        query,
        key,
        value,
        out,
        kv_cache: None,
    };

    Ok(attention_block)
}

fn load_multihead_cross_attention<B: Backend>(
    path: &str,
    dtw_timestamps: bool,
) -> Result<MultiHeadCrossAttention<B>, Box<dyn Error>> {
    let query = load_linear(&format!("{}/{}", path, "query"))?;
    let key = load_linear(&format!("{}/{}", path, "key"))?;
    let value = load_linear(&format!("{}/{}", path, "value"))?;
    let out = load_linear(&format!("{}/{}", path, "out"))?;

    let n_head: usize = load_usize::<B>("n_head", path)?;

    // Initializing attention block
    let attention_block = MultiHeadCrossAttention {
        n_head,
        query,
        key,
        value,
        out,
        kv_cache: None,
        output_attentions: dtw_timestamps.then(|| None),
    };

    Ok(attention_block)
}

fn load_mlp<B: Backend>(path: &str) -> Result<MLP<B>, Box<dyn Error>> {
    let lin1 = load_linear(&format!("{}/{}", path, "mlp1"))?;
    let lin2 = load_linear(&format!("{}/{}", path, "mlp2"))?;

    let gelu = nn::Gelu::new();

    let mlp = MLP { lin1, lin2, gelu };

    Ok(mlp)
}

fn load_conv1d<B: Backend>(path: &str, config: Conv1dConfig) -> Result<Conv1d<B>, Box<dyn Error>> {
    let weight = Param::from_tensor(load_tensor::<B, 3>("weight", path)?);
    let bias = Param::from_tensor(load_tensor::<B, 1>("bias", path)?);
    let tensor_device_ref = weight.device();

    let record = Conv1dRecord {
        weight,
        bias: Some(bias),
        stride: <usize as Module<B>>::into_record(1),
        kernel_size: <usize as Module<B>>::into_record(1),
        dilation: <usize as Module<B>>::into_record(1),
        groups: <usize as Module<B>>::into_record(1),
        padding: <usize as Module<B>>::into_record(10),
    };

    let conv1d: Conv1d<B> = config.init(&tensor_device_ref).load_record(record);
    Ok(conv1d)
}

fn load_residual_encoder_attention_block<B: Backend>(
    path: &str,
) -> Result<ResidualEncoderAttentionBlock<B>, Box<dyn Error>> {
    let attn = load_multihead_self_attention(&format!("{}/{}", path, "attn"))?;
    let attn_ln = load_layer_norm(&format!("{}/{}", path, "attn_ln"))?;
    let mlp = load_mlp(&format!("{}/{}", path, "mlp"))?;
    let mlp_ln = load_layer_norm(&format!("{}/{}", path, "mlp_ln"))?;

    let residual_block = ResidualEncoderAttentionBlock {
        attn,
        attn_ln,
        mlp,
        mlp_ln,
    };

    Ok(residual_block)
}

fn load_residual_decoder_attention_block<B: Backend>(
    path: &str,
    dtw_timestamps: bool,
) -> Result<ResidualDecoderAttentionBlock<B>, Box<dyn Error>> {
    let attn = load_multihead_self_attention(&format!("{}/{}", path, "attn"))?;
    let attn_ln = load_layer_norm(&format!("{}/{}", path, "attn_ln"))?;
    let cross_attn =
        load_multihead_cross_attention(&format!("{}/{}", path, "cross_attn"), dtw_timestamps)?;
    let cross_attn_ln = load_layer_norm(&format!("{}/{}", path, "cross_attn_ln"))?;
    let mlp = load_mlp(&format!("{}/{}", path, "mlp"))?;
    let mlp_ln = load_layer_norm(&format!("{}/{}", path, "mlp_ln"))?;

    let residual_block = ResidualDecoderAttentionBlock {
        attn,
        attn_ln,
        cross_attn,
        cross_attn_ln,
        mlp,
        mlp_ln,
    };

    Ok(residual_block)
}

fn load_audio_encoder<B: Backend>(
    path: &str,
) -> Result<(AudioEncoder<B>, AudioEncoderConfig), Box<dyn Error>> {
    let n_mels = load_usize::<B>("n_mels", path)?;
    let n_audio_state = load_usize::<B>("n_audio_state", path)?;

    let conv1_config =
        Conv1dConfig::new(n_mels, n_audio_state, 3).with_padding(PaddingConfig1d::Explicit(1));
    let conv2_config = Conv1dConfig::new(n_audio_state, n_audio_state, 3)
        .with_padding(PaddingConfig1d::Explicit(1))
        .with_stride(2);

    let conv1 = load_conv1d(&format!("{}/{}", path, "conv1"), conv1_config)?;
    let conv2 = load_conv1d(&format!("{}/{}", path, "conv2"), conv2_config)?;

    let n_layer = load_usize::<B>("n_layer", path)?;

    let blocks: Vec<ResidualEncoderAttentionBlock<B>> = (0..n_layer)
        .map(|i| load_residual_encoder_attention_block(&format!("{}/block_{}", path, i)))
        .collect::<Result<_, _>>()?;

    let ln_post = load_layer_norm(&format!("{}/{}", path, "ln_post"))?;
    let positional_embedding =
        Param::from_tensor(load_tensor::<B, 2>("positional_embedding", path)?);

    let [n_audio_ctx, _] = positional_embedding.dims();

    let n_head = blocks[0].attn.n_head;

    let audio_encoder = AudioEncoder {
        conv1,
        gelu1: nn::Gelu::new(),
        conv2,
        gelu2: nn::Gelu::new(),
        blocks,
        ln_post,
        positional_embedding,
        n_audio_ctx,
        n_mels,
    };

    let config = AudioEncoderConfig {
        n_mels,
        n_audio_ctx,
        n_audio_state,
        n_audio_head: n_head,
        n_audio_layer: n_layer,
    };

    Ok((audio_encoder, config))
}

fn load_text_decoder<B: Backend>(
    path: &str,
    dtw_timestamps: bool,
) -> Result<(TextDecoder<B>, TextDecoderConfig), Box<dyn Error>> {
    let token_embedding = load_tensor::<B, 2>("token_embedding/weight", path)?;
    let positional_embedding = load_tensor::<B, 2>("positional_embedding", path)?;
    let tensor_device_ref = token_embedding.device();

    let n_layer = load_usize::<B>("n_layer", path)?;
    let blocks: Vec<ResidualDecoderAttentionBlock<B>> = (0..n_layer)
        .map(|i| {
            load_residual_decoder_attention_block(&format!("{}/block_{}", path, i), dtw_timestamps)
        })
        .collect::<Result<_, _>>()?;

    let n_text_head = blocks[0].attn.n_head;

    let ln = load_layer_norm(&format!("{}/{}", path, "ln"))?;

    let [n_text_ctx, n_text_state] = positional_embedding.dims();

    let mask: Vec<_> = (0..MAX_TARGET_POSITIONS)
        .flat_map(|i| {
            (0..MAX_TARGET_POSITIONS).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 })
        })
        .collect();
    let mask = Tensor::<B, 1>::from_floats(mask.as_slice(), &tensor_device_ref)
        .reshape([MAX_TARGET_POSITIONS, MAX_TARGET_POSITIONS]);

    let [n_vocab, _] = token_embedding.dims();

    let text_decoder = TextDecoder {
        token_embedding: Param::from_tensor(token_embedding),
        positional_embedding: Param::from_tensor(positional_embedding),
        blocks,
        ln,
        mask: Param::from_tensor(mask),
        n_text_ctx,
        n_vocab,
    };

    let config = TextDecoderConfig {
        n_vocab,
        n_text_ctx,
        n_text_state,
        n_text_head,
        n_text_layer: n_layer,
    };

    Ok((text_decoder, config))
}

pub fn load_whisper<B: Backend>(
    path: &str,
    dtw_timestamps: bool,
) -> Result<(Whisper<B>, WhisperConfig), Box<dyn Error>> {
    let (encoder, encoder_config) = load_audio_encoder(&format!("{}/{}", path, "encoder"))?;
    let (decoder, decoder_config) =
        load_text_decoder(&format!("{}/{}", path, "decoder"), dtw_timestamps)?;
    let whisper = Whisper { encoder, decoder };

    let config = WhisperConfig {
        audio_encoder_config: encoder_config,
        text_decoder_config: decoder_config,
    };

    Ok((whisper, config))
}
