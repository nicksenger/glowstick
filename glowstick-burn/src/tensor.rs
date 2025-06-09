use std::marker::PhantomData;
use std::ops::Range;

use burn::tensor::{
    BasicOps, Bool, DType, ElementConversion, Int, Numeric, Tensor as BTensor, TensorData,
};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::cmp::Equal;
use glowstick::{
    num::{U0, U1},
    Dimension, Dimensioned, Shape, TensorShape,
};
use glowstick::{Arrayify, IsFragEqual, ShapeFragment};

pub const fn rank<B: Backend, const N: usize>(_t: &BTensor<B, N>) -> usize {
    N
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "Rank mismatch: the const generic rank provided to burn does not match the type-level rank associated with the glowstick shape. Const: ({const_level}) Type-level: ({type_level})"
    )]
    RankMismatch {
        const_level: usize,
        type_level: usize,
    },

    #[error("Dimension mismatch: expected {type_level} for dim {dim} but received {runtime}")]
    DimensionMismatch {
        dim: usize,
        runtime: usize,
        type_level: usize,
    },
}

pub struct Tensor<T, S: Shape>(pub(crate) T, pub(crate) PhantomData<S>);
impl<T: Clone, S: Shape> Clone for Tensor<T, S> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<B, S, Dtype, const D: usize> glowstick::Tensor for Tensor<BTensor<B, D, Dtype>, S>
where
    B: Backend,
    S: Shape,
    Dtype: TensorKind<B>,
{
    type Shape = S;
}

impl<B, S, Dtype, const D: usize> TryFrom<BTensor<B, D, Dtype>> for Tensor<BTensor<B, D, Dtype>, S>
where
    B: Backend,
    S: Shape,
    Dtype: TensorKind<B> + BasicOps<B>,
{
    type Error = Error;
    fn try_from(x: BTensor<B, D, Dtype>) -> Result<Self, Self::Error> {
        if S::RANK != D {
            return Err(Error::RankMismatch {
                const_level: D,
                type_level: S::RANK,
            });
        }

        for (dim, (a, b)) in x.dims().iter().copied().zip(S::iter()).enumerate() {
            if a != b {
                return Err(Error::DimensionMismatch {
                    dim,
                    runtime: a,
                    type_level: b,
                });
            }
        }

        Ok(Self(x, PhantomData))
    }
}

impl<B, S1, S2, const N: usize> std::ops::Add<Tensor<BTensor<B, N>, TensorShape<S1>>>
    for Tensor<BTensor<B, N>, TensorShape<S2>>
where
    B: Backend,
    S1: ShapeFragment,
    S2: ShapeFragment,
    (S1, S2): IsFragEqual,
{
    type Output = Self;
    fn add(self, rhs: Tensor<BTensor<B, N>, TensorShape<S1>>) -> Self {
        Tensor(self.0 + rhs.0, PhantomData)
    }
}

impl<B, S, D, const N: usize> std::ops::Add<i32> for Tensor<BTensor<B, N, D>, TensorShape<S>>
where
    B: Backend,
    D: TensorKind<B> + BasicOps<B> + Numeric<B>,
    S: ShapeFragment,
{
    type Output = Self;
    fn add(self, rhs: i32) -> Self {
        Tensor(self.0 + rhs, PhantomData)
    }
}

impl<B, S1, S2, const N: usize> std::ops::Sub<Tensor<BTensor<B, N>, TensorShape<S1>>>
    for Tensor<BTensor<B, N>, TensorShape<S2>>
where
    B: Backend,
    S1: ShapeFragment,
    S2: ShapeFragment,
    (S1, S2): IsFragEqual,
{
    type Output = Self;
    fn sub(self, rhs: Tensor<BTensor<B, N>, TensorShape<S1>>) -> Self {
        Tensor(self.0 - rhs.0, PhantomData)
    }
}

impl<B, S1, S2, const N: usize> std::ops::Div<Tensor<BTensor<B, N>, TensorShape<S1>>>
    for Tensor<BTensor<B, N>, TensorShape<S2>>
where
    B: Backend,
    S1: ShapeFragment,
    S2: ShapeFragment,
    (S1, S2): IsFragEqual,
{
    type Output = Self;
    fn div(self, rhs: Tensor<BTensor<B, N>, TensorShape<S1>>) -> Self {
        Tensor(self.0 / rhs.0, PhantomData)
    }
}
impl<B, S2, const N: usize> std::ops::Div<f32> for Tensor<BTensor<B, N>, TensorShape<S2>>
where
    B: Backend,
    S2: ShapeFragment,
{
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Tensor(self.0.div_scalar(rhs), PhantomData)
    }
}
impl<B, S2, const N: usize> std::ops::Div<f64> for Tensor<BTensor<B, N>, TensorShape<S2>>
where
    B: Backend,
    S2: ShapeFragment,
{
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Tensor(self.0.div_scalar(rhs), PhantomData)
    }
}

impl<B, S, const D: usize> std::ops::Mul<f64> for Tensor<BTensor<B, D>, S>
where
    B: Backend,
    S: Shape,
{
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0 * rhs, PhantomData)
    }
}

impl<B, S, Dtype, const D: usize> Tensor<BTensor<B, D, Dtype>, S>
where
    B: Backend,
    S: Shape,
    Dtype: TensorKind<B>,
{
    pub const fn rank() -> usize {
        S::RANK
    }

    pub fn into_inner(self) -> BTensor<B, D, Dtype> {
        self.0
    }

    pub fn inner(&self) -> &BTensor<B, D, Dtype> {
        &self.0
    }
}

impl<B, S, Dtype> Tensor<BTensor<B, 1, Dtype>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
    Dtype: TensorKind<B> + BasicOps<B>,
{
    fn check_dim(inner: &BTensor<B, 1, Dtype>) -> Result<(), Error> {
        if inner.dims()[0] != <S::Dim<U0> as Dimension>::USIZE {
            return Err(Error::DimensionMismatch {
                dim: 0,
                runtime: inner.dims()[0],
                type_level: <S::Dim<U0> as Dimension>::USIZE,
            });
        }

        Ok(())
    }
}

impl<B, S> Tensor<BTensor<B, 1, Int>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
{
    pub fn arange(range: Range<i64>, device: &B::Device) -> Self {
        Self(BTensor::<B, 1, Int>::arange(range, device), PhantomData)
    }

    pub fn float(self) -> Tensor<BTensor<B, 1>, S> {
        Tensor(self.0.float(), PhantomData)
    }
}

impl<B, S> Tensor<BTensor<B, 1>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
{
    pub fn from_floats(f: &[f32], d: &B::Device) -> Result<Self, Error> {
        Self::check_rank()?;
        let inner: BTensor<B, 1> = BTensor::from_floats(f, d);
        Self::check_dim(&inner)?;

        Ok(Self(inner, PhantomData))
    }
}

impl<B, S, const N: usize> Tensor<BTensor<B, N, Int>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
    (<S as Shape>::Rank, U1): Equal,
{
    pub fn from_ints<T: Into<TensorData>>(n: T, d: &B::Device) -> Self {
        Self(BTensor::from_ints(n, d), PhantomData)
    }
}

impl<B, S, Dtype, const N: usize> Tensor<BTensor<B, N, Dtype>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
    Dtype: TensorKind<B>,
{
    fn check_rank() -> Result<(), Error> {
        if S::RANK != N {
            return Err(Error::RankMismatch {
                const_level: 1,
                type_level: S::RANK,
            });
        }

        Ok(())
    }
}

impl<B, S, const N: usize> Tensor<BTensor<B, N>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
{
    pub fn cast(self, dtype: DType) -> Tensor<BTensor<B, N>, S> {
        Self(self.0.cast(dtype), PhantomData)
    }

    pub fn cos(self) -> Self {
        Self(self.0.cos(), PhantomData)
    }

    pub fn sin(self) -> Self {
        Self(self.0.cos(), PhantomData)
    }
}

impl<B, S, Dtype, const N: usize> Tensor<BTensor<B, N, Dtype>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
    Dtype: TensorKind<B> + BasicOps<B>,
{
    pub fn dims(&self) -> [usize; N] {
        self.0.dims()
    }

    pub fn device(&self) -> <B as Backend>::Device {
        self.0.device()
    }
}

impl<B, S, Dtype, const N: usize> Tensor<BTensor<B, N, Dtype>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
    Dtype: TensorKind<B> + BasicOps<B> + Numeric<B>,
{
    pub fn mask_fill<E>(self, mask: Tensor<BTensor<B, N, Bool>, S>, value: E) -> Self
    where
        E: ElementConversion,
    {
        Self(self.0.mask_fill(mask.into_inner(), value), PhantomData)
    }

    pub fn to_data(&self) -> TensorData {
        self.0.to_data()
    }

    pub fn into_data(self) -> TensorData {
        self.0.into_data()
    }
}

impl<B, S, Dtype, const N: usize> Tensor<BTensor<B, N, Dtype>, S>
where
    B: Backend,
    S: Shape,
    <S as Shape>::Fragment: Arrayify<usize, Out = [usize; N]>,
    (S, U0): Dimensioned,
    Dtype: TensorKind<B> + BasicOps<B> + Numeric<B>,
{
    pub fn ones(device: &B::Device) -> Self {
        Self(
            BTensor::ones(
                <<S as Shape>::Fragment as glowstick::Arrayify<usize>>::value(),
                device,
            ),
            PhantomData,
        )
    }

    pub fn zeros(device: &B::Device) -> Self {
        Self(
            BTensor::zeros(
                <<S as Shape>::Fragment as glowstick::Arrayify<usize>>::value(),
                device,
            ),
            PhantomData,
        )
    }

    pub fn from_data<E: burn::tensor::Element>(data: Vec<E>, device: &B::Device) -> Self {
        Self(
            BTensor::from_data(
                TensorData::new(
                    data,
                    <<S as Shape>::Fragment as glowstick::Arrayify<usize>>::value(),
                ),
                device,
            ),
            PhantomData,
        )
    }
}

impl<B, S, const N: usize> Tensor<BTensor<B, N>, S>
where
    B: Backend,
    S: Shape,
{
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt(), PhantomData)
    }
}
