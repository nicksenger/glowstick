use std::ops::{Add, Div, Mul};

use burn::tensor::{Int, Tensor as BurnTensor};
use glowstick::num::{U1, U1000, U128, U296};
use glowstick::{dyndims, Shape1, Shape2, Shape3, Shape4};
use glowstick_burn::Tensor;

dyndims! {
    N: SequenceLength
}

pub type Rank1Tensor<D1, B> = Tensor<BurnTensor<B, 1>, Shape1<D1>>;
pub type Rank2Tensor<D1, D2, B> = Tensor<BurnTensor<B, 2>, Shape2<D1, D2>>;
pub type Rank3Tensor<D1, D2, D3, B> = Tensor<BurnTensor<B, 3>, Shape3<D1, D2, D3>>;
pub type Rank4Tensor<D1, D2, D3, D4, B> = Tensor<BurnTensor<B, 4>, Shape4<D1, D2, D3, D4>>;
pub type Rank1IntTensor<D1, B> = Tensor<BurnTensor<B, 1, Int>, Shape1<D1>>;
pub type Rank2IntTensor<D1, D2, B> = Tensor<BurnTensor<B, 2, Int>, Shape2<D1, D2>>;

// TODO: support batched inference here like in the candle example
pub type B = U1;

pub type C = <<U128 as Mul<U1000>>::Output as Add<U296>>::Output;
pub type H = <S as Div<A>>::Output; // Head-dim
pub type Q = S;
pub type KV = <<S as Div<A>>::Output as Mul<K>>::Output;
pub type R = <A as Div<K>>::Output;

#[cfg(not(feature = "3b"))]
mod config_dims {
    use glowstick::num::{U2048, U32, U8, U8192};

    pub type A = U32; // Attention Heads
    pub type K = U8; // Key-Value Heads
    pub type S = U2048; // Hidden Size
    pub type F = U8192; // Feed-Forward Length
    pub const NUM_HIDDEN_LAYERS: usize = 16;
    pub const ROPE_THETA: f32 = 500000.;
}

#[cfg(feature = "3b")]
mod config_dims {
    use glowstick::num::{U10, U24, U300, U72, U8, U8192};

    type U3072 = <<U300 as std::ops::Mul<U10>>::Output as std::ops::Add<U72>>::Output;
    pub type A = U24; // Attention Heads
    pub type K = U8; // Key-Value Heads
    pub type S = U3072; // Hidden Size
    pub type F = U8192; // Feed-Forward Length
    pub const NUM_HIDDEN_LAYERS: usize = 28;
    pub const ROPE_THETA: f32 = 500000.;
}

pub use config_dims::*;
