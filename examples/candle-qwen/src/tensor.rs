use std::{borrow::Borrow, marker::PhantomData};

use candle::{shape::ShapeWithOneHole, DType, Device};

use glowstick::{
    num::Unsigned,
    op::{
        broadcast, flatten, matmul, narrow, narrow_dyn, narrow_dyn_start, reshape, squeeze,
        transpose, unsqueeze,
    },
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
impl<T, S, Dim> Squeeze for (T, PhantomData<S>, PhantomData<Dim>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    (S, Dim): squeeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as squeeze::Compatible>::Out>, Error>;
    fn squeeze(&self) -> Self::Out {
        self.0.borrow().inner().squeeze(<Dim as Unsigned>::USIZE)?.try_into()
    }
}

#[allow(unused)]
pub trait Unsqueeze {
    type Out;
    fn unsqueeze(&self) -> Self::Out;
}
impl<T, S, Dim> Unsqueeze for (T, PhantomData<S>, PhantomData<Dim>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    (S, Dim): unsqueeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as unsqueeze::Compatible>::Out>, Error>;
    fn unsqueeze(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .unsqueeze(<Dim as Unsigned>::USIZE)?
            .try_into()
    }
}

#[allow(unused)]
pub trait Narrow {
    type Out;
    fn narrow(&self) -> Self::Out;
}
impl<T, S, Dim, Start, Len> Narrow
    for (
        T,
        PhantomData<S>,
        PhantomData<Dim>,
        PhantomData<Start>,
        PhantomData<Len>,
    )
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    Start: Unsigned,
    Len: Unsigned,
    (S, Dim, Start, Len): narrow::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, Start, Len) as narrow::Compatible>::Out>, Error>;
    fn narrow(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .narrow(
                <Dim as Unsigned>::USIZE,
                <Start as Unsigned>::USIZE,
                <Len as Unsigned>::USIZE,
            )?
            .try_into()
    }
}

pub trait NarrowDynStart {
    type Out;
    fn narrow_dyn_start(&self) -> Self::Out;
}
impl<T, S, Dim, Len> NarrowDynStart for (T, PhantomData<S>, PhantomData<Dim>, usize, PhantomData<Len>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    Len: Unsigned,
    (S, Dim, Len): narrow_dyn_start::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, Len) as narrow_dyn_start::Compatible>::Out>, Error>;
    fn narrow_dyn_start(&self) -> Self::Out {
        self.0.borrow()
            .inner()
            .narrow(<Dim as Unsigned>::USIZE, self.3, <Len as Unsigned>::USIZE)?
            .try_into()
    }
}

#[allow(unused)]
pub trait NarrowDyn {
    type Out;
    fn narrow_dyn(&self) -> Self::Out;
}
impl<T, S, Dim, DynDim> NarrowDyn
    for (
        T,
        PhantomData<S>,
        PhantomData<Dim>,
        PhantomData<DynDim>,
        usize,
        usize,
    )
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    (S, Dim, DynDim): narrow_dyn::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, DynDim) as narrow_dyn::Compatible>::Out>, Error>;
    fn narrow_dyn(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .narrow(<Dim as Unsigned>::USIZE, self.4, self.5)?
            .try_into()
    }
}

pub trait Reshape {
    type Out;
    fn reshape<Args: ShapeWithOneHole>(&self, args: Args) -> Self::Out;
}
impl<T, S1, S2> Reshape for (T, PhantomData<S1>, PhantomData<S2>)
where
    T: Borrow<Tensor<S1>>,
    S1: Shape,
    S2: Shape,
    (S1, S2): reshape::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as reshape::Compatible>::Out>, Error>;
    fn reshape<Args: ShapeWithOneHole>(&self, args: Args) -> Self::Out {
        self.0.borrow().inner().reshape(args)?.try_into()
    }
}

pub trait Transpose {
    type Out;
    fn transpose(&self) -> Self::Out;
}
impl<T, S, Dim1, Dim2> Transpose for (T, PhantomData<S>, PhantomData<Dim1>, PhantomData<Dim2>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): transpose::Compatible,
{
    type Out = Result<Tensor<<(S, Dim1, Dim2) as transpose::Compatible>::Out>, Error>;
    fn transpose(&self) -> Self::Out {
        self.0
            .borrow()
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
impl<S, Dim1, Dim2> Flatten for (Tensor<S>, PhantomData<Dim1>, PhantomData<Dim2>)
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
impl<S1, S2> BroadcastAdd for (Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0.inner().broadcast_add(self.1.borrow().inner())?.try_into()
    }
}
impl<S1, S2> BroadcastAdd for (Tensor<S1>, &Tensor<S2>)
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
impl<S1, S2> BroadcastAdd for (&Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0.inner().broadcast_add(self.1.borrow().inner())?.try_into()
    }
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
impl<S1, U, S2> Matmul for (Tensor<S1>, U, PhantomData<S2>)
where
    U: Borrow<Tensor<S2>>,
    S1: Shape + matmul::Operand,
    S2: Shape + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type Out = Result<Tensor<TensorShape<<(S1, S2) as matmul::Compatible>::Out>>, Error>;
    fn matmul(self) -> Self::Out {
        self.0.into_inner().matmul(self.1.borrow().inner())?.try_into()
    }
}
impl<S1, U, S2> Matmul for (&Tensor<S1>, U, PhantomData<S2>)
where
    U: Borrow<Tensor<S2>>,
    S1: Shape + matmul::Operand,
    S2: Shape + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type Out = Result<Tensor<TensorShape<<(S1, S2) as matmul::Compatible>::Out>>, Error>;
    fn matmul(self) -> Self::Out {
        self.0.inner().matmul(self.1.borrow().inner())?.try_into()
    }
}

#[macro_export]
macro_rules! squeeze {
    [$t:expr,$i:ty] => {{
        use $crate::tensor::Squeeze;
        ($t, std::marker::PhantomData, std::marker::PhantomData::<$i>).squeeze()
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
    ($t:expr,$d:ty:[$s:ty,$l:ty]) => {{
        use $crate::tensor::Narrow;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$s>,
            std::marker::PhantomData::<$l>
        ).narrow()
    }};
    ($t:expr,$d:ty:[$s:expr,$l:ty]) => {{
        use $crate::tensor::NarrowDynStart;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow_dyn_start()
    }};
    ($t:expr,$d:ty:[$s:expr,$l:expr] => $y:ty) => {{
        use $crate::tensor::NarrowDyn;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$y>,
            $s,
            $l,
        )
            .narrow_dyn()
    }};
    ($t:expr,$d:ty:[$s:ty,$l:ty],$($ds:tt)+) => {{
        use $crate::tensor::Narrow;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$s>,
            std::marker::PhantomData::<$l>,
        )
            .narrow().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
    ($t:expr,$d:ty:[$s:expr,$l:ty],$($ds:tt)+) => {{
        use $crate::tensor::NarrowDynStart;
        (
            $t,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow_dyn_start().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
    ($t:expr,$d:ty:[$s:expr,$l:expr] => $y:ty,$($ds:tt)+) => {{
        use $crate::tensor::NarrowDyn;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$y>,
            $s,
            $l,
        )
            .narrow_dyn().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
}

#[macro_export]
macro_rules! reshape {
    ($t:expr,[$e:expr => $d:ty,$($ds:tt)*]) => {{
        use $crate::tensor::Reshape;
        (
            $t,
            std::marker::PhantomData::<glowstick::TensorShape<$crate::reshape_tys!($e => $d,$($ds)+)>>,
        )
            .reshape($crate::reshape_val!($d => $e,$($ds)+).tuplify())
    }};
    ($t:expr,[$($ds:tt)+]) => {{
        use $crate::tensor::Reshape;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>>,
        )
            .reshape($crate::reshape_val!($($ds)+).tuplify())
    }};
}
#[macro_export]
macro_rules! reshape_tys {
    ($e:expr => $d:ty) => {
        glowstick::Shp<(<$d as glowstick::dynamic::Dim>::Id, glowstick::Empty)>
    };
    ($e:expr => $d:ty,$($ds:tt)+) => {
        glowstick::Shp<(<$d as glowstick::dynamic::Dim>::Id, $crate::reshape_tys!($($ds)+))>
    };
    ($d:ty) => {
        glowstick::Shp<($d, glowstick::Empty)>
    };
    ($d:ty,$($ds:tt)+) => {
        glowstick::Shp<($d, $crate::reshape_tys!($($ds)+))>
    };
}
#[macro_export]
macro_rules! reshape_val {
    ($e:expr => $d:ty) => {
        glowstick::ValueList(($e, glowstick::ValueList(())))
    };
    ($d:ty) => {
        glowstick::ValueList((<$d as glowstick::num::Unsigned>::USIZE,glowstick::ValueList(())))
    };
    ($e:expr => $d:ty,$($ds:tt)+) => {
        glowstick::ValueList(($e,$crate::reshape_val!($($ds)+)))
    };
    ($d:ty,$($ds:tt)+) => {
        glowstick::ValueList((<$d as glowstick::num::Unsigned>::USIZE,$crate::reshape_val!($($ds)+)))
    };
}

#[macro_export]
macro_rules! transpose {
    ($t:expr,$d1:ty:$d2:ty) => {{
        use $crate::tensor::Transpose;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
    ($t:expr,$d1:ty:$d2:ty,$($d1s:ty:$d2s:ty),+) => {{
        use $crate::tensor::Transpose;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose().and_then(|t| $crate::transpose!(&t, $($d1s:$d2s),+))
    }};
}

#[macro_export]
macro_rules! flatten {
    ($t:expr,[$d1:ty,$d2:ty]) => {{
        use $crate::tensor::Flatten;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten()
    }};
    ($t:expr,[$d1:ty,$d2:ty],$([$d1s:ty,$d2s:ty]),+) => {{
        use $crate::tensor::Flatten;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten().and_then(|t| $crate::flatten!(&t, $([$d1s,$d2s]),+))
    }};
}

#[macro_export]
macro_rules! broadcast_add {
    ($t1:expr,$t2:expr) => {{
        use $crate::tensor::BroadcastAdd;
        ($t1, $t2).broadcast_add()
    }};
    ($t1:expr,$t2:expr,$($t2s:expr),+) => {{
        use $crate::tensor::BroadcastAdd;
        ($t1, $t2)
            .broadcast_add()
            .and_then(|t| $crate::broadcast_add!(&t, $t2s))
    }};
}

#[macro_export]
macro_rules! matmul {
    ($t1:expr,$t2:expr) => {{
        use $crate::tensor::Matmul;
        ($t1, $t2, std::marker::PhantomData).matmul()
    }};
    ($t1:expr,$t2:expr,$($t2s:expr),+) => {{
        use $crate::tensor::Matmul;
        ($t1, $t2, std::marker::PhantomData).matmul().and_then(|t| $crate::matmul!(&t, $t2s))
    }};
}
