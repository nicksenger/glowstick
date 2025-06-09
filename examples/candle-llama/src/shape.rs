use std::ops::{Add, Div, Mul};

use glowstick::dyndims;
use glowstick::num::{U1000, U152, U49};

dyndims! {
    B: BatchSize,
    N: SequenceLength
}

type U49152 = <<U49 as Mul<U1000>>::Output as Add<U152>>::Output;
pub type C = U49152; // Vocabulary
pub type H = <S as Div<A>>::Output; // Head-dim
pub type Q = S;
pub type KV = <<S as Div<A>>::Output as Mul<K>>::Output;

#[cfg(all(not(feature = "smaller"), not(feature = "smallest")))]
mod config_dims {
    use glowstick::num::{U2048, U32};

    pub type A = U32; // Attention Heads
    pub type K = U32; // Key-Value Heads
    pub type S = U2048; // Hidden Size
}

#[cfg(all(feature = "smaller", not(feature = "smallest")))]
mod config_dims {
    use glowstick::num::{U15, U5, U960};

    pub type A = U15; // Attention Heads
    pub type K = U5; // Key-Value Heads
    pub type S = U960; // Hidden Size
}

#[cfg(feature = "smallest")]
mod config_dims {
    use glowstick::num::{U3, U576, U9};

    pub type A = U9; // Attention Heads
    pub type K = U3; // Key-Value Heads
    pub type S = U576; // Hidden Size
}

pub use config_dims::*;
