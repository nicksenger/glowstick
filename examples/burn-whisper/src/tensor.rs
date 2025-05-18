use std::marker::PhantomData;

use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{BasicOps, Int, RangesArg, ReshapeArgs, Tensor as BTensor, TensorData};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::cmp::{Equal, Greater};
use glowstick::{
    Dimension, Dimensioned, Shape, TensorShape,
    num::{U0, U1, Unsigned},
    op::{
        broadcast, flatten, matmul, narrow, narrow_dyn, narrow_dyn_start, reshape, squeeze,
        transmute, transpose, unsqueeze,
    },
};
use glowstick::{IsFragEqual, ShapeFragment};

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

pub struct Tensor<T, S: Shape>(T, PhantomData<S>);
impl<T, S: Shape> Clone for Tensor<T, S>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Tensor::<T, S>(self.0.clone(), PhantomData)
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
{
    pub fn from_ints(n: TensorData, d: &B::Device) -> Result<Self, Error> {
        let bt = BTensor::from_ints(n, d);
        bt.try_into()
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

    pub fn transmute<S2>(self) -> Tensor<BTensor<B, N, Dtype>, S2>
    where
        S2: Shape,
        (S, S2): transmute::Compatible,
    {
        Tensor(self.into_inner(), PhantomData)
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

    pub fn try_slice<S2, R, const M: usize>(
        self,
        ranges: R,
    ) -> Result<Tensor<BTensor<B, N>, S2>, Error>
    where
        S2: Shape,
        (<S as Shape>::Rank, <S2 as Shape>::Rank): Equal,
        R: RangesArg<M>,
    {
        self.0.slice(ranges).try_into()
    }
}

pub trait Softmax {
    type Out;
    fn softmax(self) -> Self::Out;
}
impl<B, S, const N: usize, Dim> Softmax for (Tensor<BTensor<B, N>, S>, PhantomData<Dim>)
where
    B: Backend,
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = Tensor<BTensor<B, N>, S>;
    fn softmax(self) -> Self::Out {
        Tensor(
            softmax(self.0.into_inner(), <Dim as Unsigned>::USIZE),
            PhantomData,
        )
    }
}

pub trait LogSoftmax {
    type Out;
    fn log_softmax(self) -> Self::Out;
}
impl<B, S, const N: usize, Dim> LogSoftmax for (Tensor<BTensor<B, N>, S>, PhantomData<Dim>)
where
    B: Backend,
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = Tensor<BTensor<B, N>, S>;
    fn log_softmax(self) -> Self::Out {
        Tensor(
            log_softmax(self.0.into_inner(), <Dim as Unsigned>::USIZE),
            PhantomData,
        )
    }
}

pub trait VarMean {
    type Out;
    fn var_mean(self) -> Self::Out;
}
impl<B, S, const N: usize, Dim> VarMean for (Tensor<BTensor<B, N>, S>, PhantomData<Dim>)
where
    B: Backend,
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = (Tensor<BTensor<B, N>, S>, Tensor<BTensor<B, N>, S>);
    fn var_mean(self) -> Self::Out {
        let (var, mean) = self.0.into_inner().var_mean(<Dim as Unsigned>::USIZE);
        (Tensor(var, PhantomData), Tensor(mean, PhantomData))
    }
}

pub trait MeanDim {
    type Out;
    fn mean_dim(self) -> Self::Out;
}
impl<B, S, const N: usize, Dim> MeanDim for (Tensor<BTensor<B, N>, S>, PhantomData<Dim>)
where
    B: Backend,
    S: Shape,
    Dim: Unsigned,
    (S, Dim, U0, U1): narrow::Compatible,
{
    type Out = Tensor<BTensor<B, N>, <(S, Dim, U0, U1) as narrow::Compatible>::Out>;
    fn mean_dim(self) -> Self::Out {
        Tensor(
            self.0.into_inner().mean_dim(<Dim as Unsigned>::USIZE),
            PhantomData,
        )
    }
}

pub trait Squeeze<const M: usize> {
    type Out;
    fn squeeze(self) -> Self::Out;
}
macro_rules! squeeze_impl {
    ($in:literal => $out:literal) => {
        impl<B, S, Dim> Squeeze<$out> for (Tensor<BTensor<B, $in>, S>, PhantomData<Dim>)
        where
            B: Backend,
            S: Shape,
            Dim: Unsigned,
            (S, Dim): squeeze::Compatible,
        {
            type Out = Tensor<BTensor<B, $out>, <(S, Dim) as squeeze::Compatible>::Out>;
            fn squeeze(self) -> Self::Out {
                Tensor::<BTensor<B, $out>, <(S, Dim) as squeeze::Compatible>::Out>(
                    self.0.into_inner().squeeze(<Dim as Unsigned>::USIZE),
                    PhantomData,
                )
            }
        }
    };
}
squeeze_impl!(3 => 2);
squeeze_impl!(2 => 1);

pub trait Unsqueeze<const M: usize> {
    type Out;
    fn unsqueeze(self) -> Self::Out;
}
macro_rules! unsqueeze_impl {
    ($in:literal => $out:literal) => {
        impl<B, S, Dim> Unsqueeze<$out> for (Tensor<BTensor<B, $in>, S>, PhantomData<Dim>)
        where
            B: Backend,
            S: Shape,
            Dim: Unsigned,
            (S, Dim): unsqueeze::Compatible,
        {
            type Out = Tensor<BTensor<B, $out>, <(S, Dim) as unsqueeze::Compatible>::Out>;
            fn unsqueeze(self) -> Self::Out {
                Tensor::<BTensor<B, $out>, <(S, Dim) as unsqueeze::Compatible>::Out>(
                    self.0.into_inner().unsqueeze_dim(<Dim as Unsigned>::USIZE),
                    PhantomData,
                )
            }
        }
    };
}
unsqueeze_impl!(3 => 4);
unsqueeze_impl!(2 => 3);
unsqueeze_impl!(1 => 2);

pub trait Narrow {
    type Out;
    fn narrow(self) -> Self::Out;
}
impl<B, Dtype, S, Dim, Start, Len, const N: usize> Narrow
    for (
        Tensor<BTensor<B, N, Dtype>, S>,
        PhantomData<Dim>,
        PhantomData<Start>,
        PhantomData<Len>,
    )
where
    B: Backend,
    Dtype: TensorKind<B> + BasicOps<B>,
    S: Shape,
    Dim: Unsigned,
    Len: Unsigned,
    Start: Unsigned,
    (S, Dim, Start, Len): narrow::Compatible,
{
    type Out = Tensor<BTensor<B, N, Dtype>, <(S, Dim, Start, Len) as narrow::Compatible>::Out>;
    fn narrow(self) -> Self::Out {
        Tensor(
            self.0.into_inner().narrow(
                <Dim as glowstick::num::Unsigned>::USIZE,
                <Start as glowstick::num::Unsigned>::USIZE,
                <Len as glowstick::num::Unsigned>::USIZE,
            ),
            PhantomData,
        )
    }
}

pub trait NarrowDynStart<const N: usize> {
    type Out;
    fn narrow_dyn_start(self) -> Self::Out;
}
impl<B, Dtype, S, Dim, Len, const N: usize> NarrowDynStart<N>
    for (
        Tensor<BTensor<B, N, Dtype>, S>,
        PhantomData<Dim>,
        usize,
        PhantomData<Len>,
    )
where
    S: Shape,
    B: Backend,
    Dtype: TensorKind<B> + BasicOps<B>,
    Dim: Unsigned,
    Len: Unsigned,
    (S, Dim, Len): narrow_dyn_start::Compatible,
{
    type Out = Tensor<BTensor<B, N, Dtype>, <(S, Dim, Len) as narrow_dyn_start::Compatible>::Out>;
    fn narrow_dyn_start(self) -> Self::Out {
        Tensor(
            self.0.into_inner().narrow(
                <Dim as glowstick::num::Unsigned>::USIZE,
                self.2,
                <Len as glowstick::num::Unsigned>::USIZE,
            ),
            PhantomData,
        )
    }
}

pub trait NarrowDyn {
    type Out;
    fn narrow_dyn(self) -> Self::Out;
}
impl<B, Dtype, S, Dim, DynDim, const N: usize> NarrowDyn
    for (
        Tensor<BTensor<B, N, Dtype>, S>,
        PhantomData<Dim>,
        PhantomData<DynDim>,
        usize,
        usize,
    )
where
    B: Backend,
    S: Shape,
    Dtype: TensorKind<B> + BasicOps<B>,
    Dim: Unsigned,
    (S, Dim, DynDim): narrow_dyn::Compatible,
{
    type Out = Tensor<BTensor<B, N, Dtype>, <(S, Dim, DynDim) as narrow_dyn::Compatible>::Out>;
    fn narrow_dyn(self) -> Self::Out {
        Tensor(
            self.0
                .into_inner()
                .narrow(<Dim as glowstick::num::Unsigned>::USIZE, self.3, self.4),
            PhantomData,
        )
    }
}

pub trait Reshape<Args, const M: usize> {
    type Out;
    fn reshape(self, args: Args) -> Self::Out;
}
impl<B, S1, S2, Args, const N: usize, const M: usize> Reshape<Args, M>
    for (Tensor<BTensor<B, N>, S1>, PhantomData<TensorShape<S2>>)
where
    Args: ReshapeArgs<M>,
    B: Backend,
    S1: Shape,
    TensorShape<S2>: Shape,
    S2: ShapeFragment,
    (S1, TensorShape<S2>): reshape::Compatible,
{
    type Out = Result<Tensor<BTensor<B, M>, TensorShape<S2>>, Error>;
    fn reshape(self, args: Args) -> Self::Out {
        self.0.into_inner().reshape(args).try_into()
    }
}

pub trait Transpose {
    type Out;
    fn transpose(self) -> Self::Out;
}
impl<B, S, Dim1, Dim2, const N: usize> Transpose
    for (
        Tensor<BTensor<B, N>, S>,
        PhantomData<Dim1>,
        PhantomData<Dim2>,
    )
where
    B: Backend,
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): transpose::Compatible,
{
    type Out = Tensor<BTensor<B, N>, <(S, Dim1, Dim2) as transpose::Compatible>::Out>;
    fn transpose(self) -> Self::Out {
        Tensor(
            self.0.into_inner().swap_dims(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            ),
            PhantomData,
        )
    }
}

pub trait Flatten<const M: usize> {
    type Out;
    fn flatten(self) -> Self::Out;
}
impl<B, S, Dim1, Dim2, const N: usize, const M: usize> Flatten<M>
    for (
        Tensor<BTensor<B, N>, S>,
        PhantomData<Dim1>,
        PhantomData<Dim2>,
    )
where
    B: Backend,
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): flatten::Compatible,
{
    type Out = Tensor<BTensor<B, M>, <(S, Dim1, Dim2) as flatten::Compatible>::Out>;
    fn flatten(self) -> Self::Out {
        Tensor(
            self.0.into_inner().flatten(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            ),
            PhantomData,
        )
    }
}

pub trait Expand<const M: usize> {
    type Out;
    fn expand(self) -> Self::Out;
}
impl<B, S1, S2, const N: usize, const M: usize> Expand<M>
    for (Tensor<BTensor<B, N>, S1>, &Tensor<BTensor<B, M>, S2>)
where
    B: Backend,
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Tensor<BTensor<B, M>, <(S2, S1) as broadcast::Compatible>::Out>;
    fn expand(self) -> Self::Out {
        Tensor(
            self.0.into_inner().expand(self.1.inner().dims()),
            PhantomData,
        )
    }
}

pub trait Matmul {
    type Out;
    fn matmul(self) -> Self::Out;
}
impl<B, S1, S2, const N: usize> Matmul for (Tensor<BTensor<B, N>, S1>, Tensor<BTensor<B, N>, S2>)
where
    B: Backend,
    S1: Shape + matmul::Operand,
    S2: Shape + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type Out = Tensor<BTensor<B, N>, TensorShape<<(S1, S2) as matmul::Compatible>::Out>>;
    fn matmul(self) -> Self::Out {
        Tensor(self.0.into_inner().matmul(self.1.into_inner()), PhantomData)
    }
}

#[macro_export]
macro_rules! log_softmax {
    [$t:expr,$i:ty] => {{
        use $crate::tensor::LogSoftmax;
        ($t, std::marker::PhantomData::<$i>).log_softmax()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::log_softmax![$crate::log_softmax![$t,$i],$($is),+]
    }};
}

#[macro_export]
macro_rules! softmax {
    [$t:expr,$i:ty] => {{
        use $crate::tensor::Softmax;
        ($t, std::marker::PhantomData::<$i>).softmax()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::softmax![$crate::softmax![$t,$i],$($is),+]
    }};
}

#[macro_export]
macro_rules! var_mean {
    [$t:expr,$i:ty] => {{
        use $crate::tensor::VarMean;
        ($t, std::marker::PhantomData::<$i>).var_mean()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::var_mean![$crate::var_mean![$t,$i],$($is),+]
    }};
}

#[macro_export]
macro_rules! mean_dim {
    [$t:expr,$i:ty] => {{
        use $crate::tensor::MeanDim;
        ($t, std::marker::PhantomData::<$i>).mean_dim()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::mean_dim![$crate::mean_dim![$t,$i],$($is),+]
    }};
}

#[macro_export]
macro_rules! squeeze {
    [$t:expr,$i:ty] => {{
        use $crate::tensor::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::squeeze![$crate::squeeze![$t,$i],$($is),+]
    }};
}

#[macro_export]
macro_rules! unsqueeze {
    [$t:expr,$i:ty] => {{
        use $crate::tensor::Unsqueeze;
        ($t, std::marker::PhantomData::<$i>).unsqueeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::unsqueeze![$crate::unsqueeze![$t,$i],$($is),+]
    }};
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
            .reshape($crate::reshape_val!($d => $e,$($ds)+).into_array())
    }};
    ($t:expr,[$($ds:tt)+]) => {{
        use $crate::tensor::Reshape;
        (
            $t,
            std::marker::PhantomData::<glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>>,
        )
            .reshape($crate::reshape_val!($($ds)+).into_array())
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
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
    ($t:expr,$d1:ty:$d2:ty,$($d1s:ty:$d2s:ty),+) => {{
        use $crate::tensor::Transpose;
        (
            $t,
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
        let t = (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten();

        $crate::flatten!(&t, $([$d1s,$d2s]),+)
    }};
}

#[macro_export]
macro_rules! expand {
    ($t1:expr,$t2:expr) => {{
        use $crate::tensor::Expand;
        ($t1, $t2).expand()
    }};
    ($t1:expr,$t2:expr,$($t2s:expr),+) => {{
        $crate::expand![$crate::expand!($t1, $t2),$($t2s),+] 
    }};
}

#[macro_export]
macro_rules! matmul {
    ($t1:expr,$t2:expr) => {{
        use $crate::tensor::Matmul;
        ($t1, $t2).matmul()
    }};
    ($t1:expr,$t2:expr,$($t2s:expr),+) => {{
        $crate::matmul![$crate::matmul!($t1, $t2),$($t2s),+] 
    }};
}
