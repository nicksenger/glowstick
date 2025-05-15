use std::ops::{Add, Mul};

use glowstick::num::{U5, U10, U150, U240, U300, U865, U1000, U10000};
use glowstick::{Dyn, Shape1, Shape2, Shape3, Shape4};

use crate::tensor::Tensor;
use burn::tensor::{Int, Tensor as BurnTensor};

pub type Rank1Tensor<D1, B> = Tensor<BurnTensor<B, 1>, Shape1<D1>>;
pub type Rank2Tensor<D1, D2, B> = Tensor<BurnTensor<B, 2>, Shape2<D1, D2>>;
pub type Rank3Tensor<D1, D2, D3, B> = Tensor<BurnTensor<B, 3>, Shape3<D1, D2, D3>>;
pub type Rank4Tensor<D1, D2, D3, D4, B> = Tensor<BurnTensor<B, 4>, Shape4<D1, D2, D3, D4>>;
pub type Rank2IntTensor<D1, D2, B> = Tensor<BurnTensor<B, 2, Int>, Shape2<D1, D2>>;

pub type U240000 = <U240 as std::ops::Mul<U1000>>::Output;
pub type U51865 = <<<U10000 as Mul<U5>>::Output as Add<U1000>>::Output as Add<U865>>::Output;
pub type U1500 = <U150 as Mul<U10>>::Output;
pub type U3000 = <U300 as Mul<U10>>::Output;

pub struct SequenceLength;
pub type L = Dyn<SequenceLength>;

pub struct BatchBeams;
pub type BB = Dyn<BatchBeams>;

pub struct Ctx;
pub type C = Dyn<Ctx>;

pub struct State;
pub type St = Dyn<State>;

pub struct StateDivHead;
pub type SH = Dyn<StateDivHead>;

pub type X = Dyn<()>;
pub type Y = Dyn<()>;
pub type Z = Dyn<()>;
