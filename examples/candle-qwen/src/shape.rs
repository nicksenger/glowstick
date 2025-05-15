use std::ops::{Add, Mul};

use glowstick::num::{U1000, U151, U936};
use glowstick::tuple::{self, Tuplify};
use glowstick::Dyn;

pub struct SequenceLength;
impl Tuplify for SequenceLength {
    type Out = ();
}
impl tuple::Value for SequenceLength {
    type Out = ();
    fn value() {}
}
impl Tuplify for BatchHLen {
    type Out = ();
}
impl tuple::Value for BatchHLen {
    type Out = ();
    fn value() {}
}

pub struct BatchHLen;
pub type L = Dyn<SequenceLength>;
pub type Bhl = Dyn<BatchHLen>;
pub type U151936 = <<U151 as Mul<U1000>>::Output as Add<U936>>::Output;
