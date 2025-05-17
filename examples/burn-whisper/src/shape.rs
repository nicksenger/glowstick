use std::marker::PhantomData;
use std::ops::Add;

use glowstick::cmp::{And, False, IsEqual, Max, True};
use glowstick::dynamic::{Any, DynMax, DynMul, IsDynEqual, IsDynGreater, IsDynLess, Term};
use glowstick::num::{Mul, UInt, U1, U10, U1000, U10000, U150, U240, U300, U5, U865};
use glowstick::{Arrayify, Dyn, Shape1, Shape2, Shape3, Shape4};

use crate::tensor::Tensor;
use burn::tensor::{Int, Tensor as BurnTensor};

pub type Rank1Tensor<D1, B> = Tensor<BurnTensor<B, 1>, Shape1<D1>>;
pub type Rank2Tensor<D1, D2, B> = Tensor<BurnTensor<B, 2>, Shape2<D1, D2>>;
pub type Rank3Tensor<D1, D2, D3, B> = Tensor<BurnTensor<B, 3>, Shape3<D1, D2, D3>>;
pub type Rank4Tensor<D1, D2, D3, D4, B> = Tensor<BurnTensor<B, 4>, Shape4<D1, D2, D3, D4>>;
pub type Rank2IntTensor<D1, D2, B> = Tensor<BurnTensor<B, 2, Int>, Shape2<D1, D2>>;

pub type U240000 = <U240 as std::ops::Mul<U1000>>::Output;
pub type U51865 =
    <<<U10000 as std::ops::Mul<U5>>::Output as std::ops::Add<U1000>>::Output as std::ops::Add<
        U865,
    >>::Output;
pub type U1500 = <U150 as std::ops::Mul<U10>>::Output;
pub type U3000 = <U300 as std::ops::Mul<U10>>::Output;

pub type C = Dyn<Any>;
pub type St = Dyn<Any>;
pub type SH = Dyn<Any>;
pub type X = Dyn<Any>;
pub type Y = Dyn<Any>;
pub type Z = Dyn<Any>;

pub struct BatchSize;
pub type BB = Dyn<Term<U1, BatchSize>>;
impl IsDynEqual<BatchSize> for BatchSize {
    type Out = True;
}
impl DynMax<BatchSize> for BatchSize {
    type Out = BatchSize;
}

pub struct SequenceLength;
pub type L = Dyn<Term<U1, SequenceLength>>;
impl IsDynEqual<SequenceLength> for SequenceLength {
    type Out = True;
}
impl DynMax<SequenceLength> for SequenceLength {
    type Out = SequenceLength;
}
