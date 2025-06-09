use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::activation::log_softmax;
use burn::tensor::Tensor as BTensor;

use glowstick::cmp::Greater;
use glowstick::{num::Unsigned, Shape};

use crate::Tensor;

/// Applies the log softmax function to a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{log_softmax, Tensor};
/// use glowstick::{Shape3, num::*, dyndims};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let logsoftmaxed = log_softmax!(a.clone(), U1);
///
/// assert_eq!(logsoftmaxed.dims(), [2, 3, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! log_softmax {
    [$t:expr,$i:ty] => {{
        use $crate::op::log_softmax::LogSoftmax;
        ($t, std::marker::PhantomData::<$i>).log_softmax()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::log_softmax![$crate::log_softmax![$t,$i],$($is),+]
    }};
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
