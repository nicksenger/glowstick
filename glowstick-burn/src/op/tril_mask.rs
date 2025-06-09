use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::{Bool, Tensor as BTensor};

use glowstick::Shape;

use crate::Tensor;

/// Creates a lower triangular mask of the specified shape.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{tril_mask, Tensor};
/// use glowstick::{Shape2, num::*, dyndims};
///
/// dyndims! {
///     B: BatchSize,
///     N: SequenceLength
/// }
///
/// let device = NdArrayDevice::Cpu;
/// let mask = tril_mask!(0, &device, Backend, [{ 2 } => B, { 3 } => N]);
///
/// assert_eq!(
///     mask.inner().to_data().to_vec::<bool>().unwrap(),
///     &[
///         false, true, true,
///         false, false, true
///     ]
/// );
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! tril_mask {
    ($o:expr,$d:expr,$b:ty,[$($ds:tt)+]) => {{
        use $crate::op::tril_mask::TrilMask;
        (
            std::marker::PhantomData::<$b>,
            std::marker::PhantomData::<glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>>,
            $o,
            $d,
        )
            .tril_mask($crate::reshape_val!($($ds)+).into_array())
    }};
}

pub trait TrilMask<const M: usize> {
    type Out;
    fn tril_mask(self, shape: [usize; M]) -> Self::Out;
}
impl<B, S, const M: usize> TrilMask<M>
    for (PhantomData<B>, PhantomData<S>, i64, &<B as Backend>::Device)
where
    B: Backend,
    S: Shape,
{
    type Out = Tensor<BTensor<B, M, Bool>, S>;
    fn tril_mask(self, shape: [usize; M]) -> Self::Out {
        Tensor(
            BTensor::<B, M, Bool>::tril_mask(shape, self.2, self.3),
            PhantomData,
        )
    }
}
