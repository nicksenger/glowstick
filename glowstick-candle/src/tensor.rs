use std::marker::PhantomData;

use candle::{DType, Device};

use glowstick::Shape;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Rank mismatch: runtime ({runtime}) vs type-level ({type_level})")]
    RankMismatch { runtime: usize, type_level: usize },

    #[error("Dimension mismatch: expected {type_level} for dim {dim} but received {runtime}")]
    DimensionMismatch {
        dim: usize,
        runtime: usize,
        type_level: usize,
    },

    #[error("{0}")]
    Candle(#[from] candle::Error),
}

#[allow(unused)]
#[derive(Debug)]
pub struct Tensor<S: Shape>(pub(crate) candle::Tensor, pub(crate) PhantomData<S>);
impl<S: Shape> AsRef<candle::Tensor> for Tensor<S> {
    fn as_ref(&self) -> &candle::Tensor {
        self.inner()
    }
}

impl<S: Shape> Clone for Tensor<S> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<S> glowstick::Tensor for Tensor<S>
where
    S: Shape,
{
    type Shape = S;
}
impl<S> glowstick::Tensor for &Tensor<S>
where
    S: Shape,
{
    type Shape = S;
}

impl<S> TryFrom<candle::Tensor> for Tensor<S>
where
    S: Shape,
{
    type Error = Error;
    fn try_from(x: candle::Tensor) -> Result<Self, Self::Error> {
        if S::RANK != x.rank() {
            return Err(Error::RankMismatch {
                runtime: x.rank(),
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

impl<S> std::ops::Add<Tensor<S>> for &Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn add(self, rhs: Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 + rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Add<Tensor<S>> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn add(self, rhs: Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((self.0 + rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Add<&Tensor<S>> for &Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn add(self, rhs: &Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 + &rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Add<&Tensor<S>> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn add(self, rhs: &Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((self.0 + &rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Sub<Tensor<S>> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn sub(self, rhs: Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 - rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Mul<Tensor<S>> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn mul(self, rhs: Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 * rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Mul<&Tensor<S>> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn mul(self, rhs: &Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 * &rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Mul<&Tensor<S>> for &Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn mul(self, rhs: &Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 * &rhs.0)?, PhantomData))
    }
}

impl<S> std::ops::Mul<f64> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn mul(self, rhs: f64) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 * rhs)?, PhantomData))
    }
}

impl<S> std::ops::Div<Tensor<S>> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn div(self, rhs: Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 / &rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Div<f64> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn div(self, rhs: f64) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 / rhs)?, PhantomData))
    }
}

impl<S> Tensor<S>
where
    S: Shape,
{
    pub fn inner(&self) -> &candle::Tensor {
        &self.0
    }

    pub fn shape() -> impl Into<candle::Shape> {
        <S as Shape>::iter().collect::<Vec<_>>()
    }

    pub fn dims(&self) -> &[usize] {
        self.inner().dims()
    }

    pub fn from_vec<D: candle::WithDType>(v: Vec<D>, device: &Device) -> Result<Self, Error> {
        candle::Tensor::from_vec(v, Self::shape(), device)
            .map(|t| Self(t, PhantomData))
            .map_err(Into::into)
    }

    pub fn zeros(dtype: DType, device: &Device) -> Result<Self, Error> {
        candle::Tensor::zeros(Self::shape(), dtype, device)
            .map(|t| Self(t, PhantomData))
            .map_err(Into::into)
    }

    pub fn ones(dtype: DType, device: &Device) -> Result<Self, Error> {
        candle::Tensor::ones(Self::shape(), dtype, device)
            .map(|t| Self(t, PhantomData))
            .map_err(Into::into)
    }

    pub fn zeros_like(&self) -> Result<Self, Error> {
        Ok(Self(self.0.zeros_like()?, PhantomData))
    }

    /// Return the candle tensor, discarding type information
    pub fn into_inner(self) -> candle::Tensor {
        self.0
    }

    pub fn to_dtype(&self, dtype: candle::DType) -> Result<Self, Error> {
        Ok(Self(self.0.to_dtype(dtype)?, PhantomData))
    }

    pub fn dtype(&self) -> candle::DType {
        self.0.dtype()
    }

    pub fn contiguous(&self) -> Result<Self, Error> {
        Ok(Self(self.0.contiguous()?, PhantomData))
    }

    pub fn exp(&self) -> Result<Self, Error> {
        Ok(Self(self.0.exp()?, PhantomData))
    }

    pub fn clamp(&self, a: f32, b: f32) -> Result<Self, Error> {
        Ok(Self(self.0.clamp(a, b)?, PhantomData))
    }

    pub fn neg(&self) -> Result<Self, Error> {
        Ok(Self(self.0.neg()?, PhantomData))
    }

    pub fn to_device(&self, device: &Device) -> Result<Self, Error> {
        Ok(Self(self.0.to_device(device)?, PhantomData))
    }

    pub fn log(&self) -> Result<Self, Error> {
        Ok(Self(self.0.log()?, PhantomData))
    }

    pub fn minimum(&self, other: &Self) -> Result<Self, Error> {
        Ok(Self(self.0.minimum(other.inner())?, PhantomData))
    }

    pub fn maximum(&self, other: &Self) -> Result<Self, Error> {
        Ok(Self(self.0.maximum(other.inner())?, PhantomData))
    }

    pub fn detach(self) -> Self {
        Self(self.0.detach(), PhantomData)
    }

    pub fn abs(&self) -> Result<Self, Error> {
        Ok(Self(self.0.abs()?, PhantomData))
    }
}
