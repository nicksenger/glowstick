use std::ops::{Add, Mul};

use glowstick::dynamic::{DynMul, Wild};
use glowstick::num::{U1000, U151, U936};
use glowstick::{dyndim, Dyn};

dyndim!(N <- NumOutputs);
dyndim!(L <- SequenceLength);
dyndim!(NL <- NxL);

impl DynMul<SequenceLength> for NumOutputs {
    type Out = NxL;
}
impl DynMul<NumOutputs> for SequenceLength {
    type Out = NxL;
}

pub type Bhl = Dyn<Wild>;
pub type U151936 = <<U151 as Mul<U1000>>::Output as Add<U936>>::Output;
