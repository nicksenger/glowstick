use std::marker::PhantomData;

use burn::{
    prelude::Backend,
    tensor::{BasicOps, Tensor as BTensor, TensorKind},
};
use glowstick::{num::Unsigned, op::cat_dyn, Shape};

use crate::Tensor;

/// Concatenates the given tensors along a specified dimension.
/// A dynamic dimension must be provided for the return type.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{cat, Tensor};
/// use glowstick::{Shape4, num::*, dyndims};
///
/// dyndims! {
///     B: BatchSize
/// }
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 4>, Shape4<U1, U4, U3, U2>>::ones(&device);
/// let b = Tensor::<BurnTensor<Backend, 4>, Shape4<U1, U4, U3, U2>>::ones(&device);
/// let concatenated = cat!(vec![a, b], U0 => B);
///
/// assert_eq!(concatenated.dims(), [2, 4, 3, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! cat {
    ($ts:expr,$i:ty => $d:ty) => {{
        use $crate::op::cat::Cat;
        (
            $ts,
            std::marker::PhantomData::<$i>,
            std::marker::PhantomData::<$d>,
        )
            .cat()
    }};
}

pub trait Cat {
    type Out;
    fn cat(self) -> Self::Out;
}
impl<B, Dt, S, I, D, const N: usize> Cat
    for (
        Vec<Tensor<BTensor<B, N, Dt>, S>>,
        PhantomData<I>,
        PhantomData<glowstick::Dyn<D>>,
    )
where
    B: Backend,
    Dt: TensorKind<B> + BasicOps<B>,
    S: Shape,
    (S, I, glowstick::Dyn<D>): cat_dyn::Compatible,
    I: Unsigned,
{
    type Out = Tensor<BTensor<B, N, Dt>, <(S, I, glowstick::Dyn<D>) as cat_dyn::Compatible>::Out>;
    fn cat(self) -> Self::Out {
        Tensor(
            BTensor::cat(
                self.0.into_iter().map(Tensor::into_inner).collect(),
                <I as Unsigned>::USIZE,
            ),
            PhantomData,
        )
    }
}
