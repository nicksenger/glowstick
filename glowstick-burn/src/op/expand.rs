use std::marker::PhantomData;

use burn::tensor::{BasicOps, BroadcastArgs, Tensor as BTensor};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::{op::broadcast, Shape};

use crate::Tensor;

/// Expands the lefthand tensor to the shape of the provided righthand tensor
/// or shape.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{expand, Tensor};
/// use glowstick::{Shape2, Shape4, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 2>, Shape2<U1, U2>>::ones(&device);
/// let b = Tensor::<BurnTensor<Backend, 4>, Shape4<U1, U4, U3, U2>>::ones(&device);
/// let c = expand!(a.clone(), &b);
/// let d = expand!(a, [U1, U4, U3, U2]);
///
/// assert_eq!(c.dims(), [1, 4, 3, 2]);
/// assert_eq!(d.dims(), [1, 4, 3, 2]);
/// # Ok(())
/// # }
/// ```
///
/// When expanding to a shape, a combination of type-level integers and
/// expressions bound to dynamic dimensions may be provided.
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{expand, Tensor};
/// use glowstick::{Shape2, Shape4, num::{U1, U2, U3, U4}, dyndims};
///
/// dyndims! {
///     B: BatchSize,
///     N: SequenceLength
/// }
///
/// let device = NdArrayDevice::Cpu;
/// let [batch_size, seq_len] = [4, 12];
/// let a = Tensor::<BurnTensor<Backend, 2>, Shape2<U1, U2>>::ones(&device);
/// let b = expand!(a, [{ batch_size } => B, { seq_len } => N, U3, U2]);
///
/// assert_eq!(b.dims(), [4, 12, 3, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! expand {
    ($t:expr,[$($ds:tt)+]) => {{
        type S = glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>;
        use $crate::op::expand::Expand;
        (
            $t,
            std::marker::PhantomData::<S>,
        )
            .expand($crate::reshape_val!($($ds)+).into_array())
    }};
    ($t1:expr,$t2:expr) => {{
        use $crate::op::expand::Expand;
        (
            $t1,
            $t2,
        )
            .expand($t2.inner().shape().dims())
    }}
}

pub trait Expand<A, const N: usize, const M: usize>
where
    A: BroadcastArgs<N, M>,
{
    type Out;
    fn expand(self, shape: A) -> Self::Out;
}
impl<B, S1, S2, D1, D2, const N: usize, const M: usize> Expand<[usize; M], N, M>
    for (
        Tensor<BTensor<B, N, D1>, S1>,
        &Tensor<BTensor<B, M, D2>, S2>,
    )
where
    B: Backend,
    S1: Shape,
    S2: Shape,
    D1: TensorKind<B> + BasicOps<B>,
    D2: TensorKind<B>,
    (S2, S1): broadcast::Compatible,
{
    type Out = Tensor<BTensor<B, M, D1>, <(S2, S1) as broadcast::Compatible>::Out>;
    fn expand(self, shape: [usize; M]) -> Self::Out {
        Tensor(self.0.into_inner().expand(shape), PhantomData)
    }
}
impl<B, S1, S2, const N: usize, const M: usize> Expand<[i32; M], N, M>
    for (Tensor<BTensor<B, N>, S1>, PhantomData<S2>)
where
    B: Backend,
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Tensor<BTensor<B, M>, <(S2, S1) as broadcast::Compatible>::Out>;
    fn expand(self, shape: [i32; M]) -> Self::Out {
        Tensor(self.0.into_inner().expand(shape), PhantomData)
    }
}
