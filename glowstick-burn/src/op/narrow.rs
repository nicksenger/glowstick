use std::marker::PhantomData;

use burn::tensor::{BasicOps, Tensor as BTensor};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::{
    num::Unsigned,
    op::{narrow, narrow_dyn, narrow_dyn_start},
    Shape,
};

use crate::Tensor;

/// Narrows a tensor at the specified dimension from start index to length.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{narrow, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let narrowed = narrow!(a.clone(), U0: [U1, U1]);
///
/// assert_eq!(narrowed.dims(), [1, 3, 4]);
/// # Ok(())
/// # }
/// ```
///
/// When using dynamic start and length, the resulting tensor's shape will be determined by the provided expressions.
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{narrow, Tensor};
/// use glowstick::{Shape3, num::{U0, U1, U2, U3, U4}, dyndims};
///
/// dyndims! {
///     N: SequenceLength
/// }
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let [start, len] = [1, 2];
/// let narrowed = narrow!(a.clone(), U1: [{ start }, { len }] => N);
///
/// assert_eq!(narrowed.dims(), [2, 2, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! narrow {
    ($t:expr,$d:ty:[$s:ty,$l:ty]) => {{
        glowstick::op::narrow::check::<_, _, $d, $s, $l>(&$t);
        use $crate::op::narrow::Narrow;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$s>,
            std::marker::PhantomData::<$l>
        ).narrow()
    }};
    ($t:expr,$d:ty:[$s:expr,$l:ty]) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $l>(&$t);
        use $crate::op::narrow::NarrowDynStart;
        (
            $t,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow_dyn_start()
    }};
    ($t:expr,$d:ty:[$s:expr,$l:expr] => $y:ty) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $y>(&$t);
        use $crate::op::narrow::NarrowDyn;
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
        glowstick::op::narrow::check::<_, _, $d, $s, $l>(&$t);
        use $crate::op::narrow::Narrow;
        let narrowed = (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$s>,
            std::marker::PhantomData::<$l>,
        )
            .narrow();
        $crate::narrow!(narrowed,$($ds)+)
    }};
    ($t:expr,$d:ty:[$s:expr,$l:ty],$($ds:tt)+) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $l>(&$t);
        use $crate::op::narrow::NarrowDynStart;
        (
            $t,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow_dyn_start().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
    ($t:expr,$d:ty:[$s:expr,$l:expr] => $y:ty,$($ds:tt)+) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $y>(&$t);
        use $crate::op::narrow::NarrowDyn;
        let narrowed = (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$y>,
            $s,
            $l,
        )
            .narrow_dyn();
        $crate::narrow!(narrowed,$($ds)+)
    }};
}

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
