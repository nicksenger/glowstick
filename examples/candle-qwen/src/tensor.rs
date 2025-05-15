use std::marker::PhantomData;

use candle::{shape::ShapeWithOneHole, DType, Device};

use glowstick::{
    num::Unsigned,
    op::{broadcast, flatten, matmul, narrow, reshape, squeeze, transpose, unsqueeze},
    Shape, TensorShape,
};

use crate::Error;

#[allow(unused)]
pub struct Tensor<S: Shape>(candle::Tensor, PhantomData<S>);

impl<S> glowstick::Tensor for Tensor<S>
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

impl<S> std::ops::Mul<f64> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn mul(self, rhs: f64) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 * rhs)?, PhantomData))
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

    pub fn zeros(dtype: DType, device: &Device) -> Result<Self, Error> {
        candle::Tensor::zeros(Self::shape(), dtype, device)
            .map(|t| Self(t, PhantomData))
            .map_err(Into::into)
    }

    /// Return the candle tensor, discarding type information
    pub fn into_inner(self) -> candle::Tensor {
        self.0
    }

    pub fn to_dtype(&self, dtype: candle::DType) -> Result<Self, Error> {
        Ok(Self(self.0.to_dtype(dtype)?, PhantomData))
    }

    pub fn contiguous(&self) -> Result<Self, Error> {
        Ok(Self(self.0.contiguous()?, PhantomData))
    }
}

pub trait Squeeze {
    type Out;
    fn squeeze(&self) -> Self::Out;
}
impl<S, Dim> Squeeze for (&Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): squeeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as squeeze::Compatible>::Out>, Error>;
    fn squeeze(&self) -> Self::Out {
        self.0.inner().squeeze(<Dim as Unsigned>::USIZE)?.try_into()
    }
}

#[allow(unused)]
pub trait Unsqueeze {
    type Out;
    fn unsqueeze(&self) -> Self::Out;
}
impl<S, Dim> Unsqueeze for (&Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): unsqueeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as unsqueeze::Compatible>::Out>, Error>;
    fn unsqueeze(&self) -> Self::Out {
        self.0
            .inner()
            .unsqueeze(<Dim as Unsigned>::USIZE)?
            .try_into()
    }
}

pub trait Narrow {
    type Out;
    fn narrow(&self) -> Self::Out;
}
impl<S, Dim, Len> Narrow for (&Tensor<S>, PhantomData<Dim>, usize, PhantomData<Len>)
where
    S: Shape,
    Dim: Unsigned,
    Len: Unsigned,
    (S, Dim, Len): narrow::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, Len) as narrow::Compatible>::Out>, Error>;
    fn narrow(&self) -> Self::Out {
        self.0
            .inner()
            .narrow(
                <Dim as glowstick::num::Unsigned>::USIZE,
                self.2,
                <Len as glowstick::num::Unsigned>::USIZE,
            )?
            .try_into()
    }
}

pub trait Reshape {
    type Out;
    fn reshape(&self) -> Self::Out;
}
impl<S1, S2> Reshape for (&Tensor<S1>, PhantomData<S2>)
where
    S1: Shape,
    S2: Shape + glowstick::tuple::Value,
    <S2 as glowstick::tuple::Value>::Out: ShapeWithOneHole,
    (S1, S2): reshape::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as reshape::Compatible>::Out>, Error>;
    fn reshape(&self) -> Self::Out {
        self.0
            .inner()
            .reshape(<S2 as glowstick::tuple::Value>::value())?
            .try_into()
    }
}

pub trait Transpose {
    type Out;
    fn transpose(&self) -> Self::Out;
}
impl<S, Dim1, Dim2> Transpose for (&Tensor<S>, PhantomData<Dim1>, PhantomData<Dim2>)
where
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): transpose::Compatible,
{
    type Out = Result<Tensor<<(S, Dim1, Dim2) as transpose::Compatible>::Out>, Error>;
    fn transpose(&self) -> Self::Out {
        self.0
            .inner()
            .transpose(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            )?
            .try_into()
    }
}

pub trait Flatten {
    type Out;
    fn flatten(&self) -> Self::Out;
}
impl<S, Dim1, Dim2> Flatten for (&Tensor<S>, PhantomData<Dim1>, PhantomData<Dim2>)
where
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): flatten::Compatible,
{
    type Out = Result<Tensor<<(S, Dim1, Dim2) as flatten::Compatible>::Out>, Error>;
    fn flatten(&self) -> Self::Out {
        self.0
            .inner()
            .flatten(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            )?
            .try_into()
    }
}

pub trait BroadcastAdd {
    type Out;
    fn broadcast_add(&self) -> Self::Out;
}
impl<S1, S2> BroadcastAdd for (&Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0.inner().broadcast_add(self.1.inner())?.try_into()
    }
}

pub trait Matmul {
    type Out;
    fn matmul(self) -> Self::Out;
}
impl<S1, S2> Matmul for (Tensor<S1>, &Tensor<S2>)
where
    S1: Shape + matmul::Operand,
    S2: Shape + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type Out = Result<Tensor<TensorShape<<(S1, S2) as matmul::Compatible>::Out>>, Error>;
    fn matmul(self) -> Self::Out {
        self.0.into_inner().matmul(self.1.inner())?.try_into()
    }
}

#[macro_export]
macro_rules! squeeze {
    [$t:expr,$i:ty] => {{
        use $crate::tensor::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {
        $crate::squeeze![&$crate::squeeze![$t, $i]?, $($is),+]
    };
}

#[macro_export]
macro_rules! unsqueeze {
    [$t:expr,$i:ty] => {{
        use $crate::tensor::Unsqueeze;
        ($t, std::marker::PhantomData::<$i>).unsqueeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {
        $crate::unsqueeze![&$crate::unsqueeze![$t, $i]?, $($is),+]
    };
}

#[macro_export]
macro_rules! narrow {
    ($t:expr,$d:ty,$s:expr,$l:ty) => {{
        use $crate::tensor::Narrow;
        (
            $t,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow()
    }};
}

#[macro_export]
macro_rules! reshape {
    ($t:expr,$s:ty) => {{
        use $crate::tensor::Reshape;
        ($t, std::marker::PhantomData::<$s>).reshape()
    }};
}

#[macro_export]
macro_rules! transpose {
    ($t:expr,$d1:ty,$d2:ty) => {{
        use $crate::tensor::Transpose;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
}

#[macro_export]
macro_rules! flatten {
    ($t:expr,$d1:ty,$d2:ty) => {{
        use $crate::tensor::Flatten;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten()
    }};
}

#[macro_export]
macro_rules! broadcast_add {
    ($t1:expr,$t2:expr) => {{
        use $crate::tensor::BroadcastAdd;
        ($t1, $t2).broadcast_add()
    }};
}

#[macro_export]
macro_rules! matmul {
    ($t1:expr,$t2:expr) => {{
        use $crate::tensor::Matmul;
        ($t1, $t2).matmul()
    }};
}
