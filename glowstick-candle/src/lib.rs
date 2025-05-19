pub mod op;
pub mod prelude;
pub mod tensor;

pub use tensor::{Error, Tensor};

#[doc = include_str!("../../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;
