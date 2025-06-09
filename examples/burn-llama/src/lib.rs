pub(crate) mod cache;
pub mod llama;
pub mod pretrained;
pub mod sampling;
pub mod shape;
pub mod tokenizer;
mod transformer;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Glowstick error: {0}")]
    Glowstick(#[from] glowstick_burn::Error),
}
