use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::activation::softmax;
use burn::tensor::Tensor as BTensor;
use glowstick::cmp::Greater;
use glowstick::{num::Unsigned, Shape};

use crate::Tensor;

/// Applies the softmax function to a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{softmax, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let softmaxed = softmax!(a, U1);
///
/// assert_eq!(softmaxed.dims(), [2, 3, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! softmax {
    [$t:expr,$i:ty] => {{
        use $crate::op::softmax::Softmax;
        ($t, std::marker::PhantomData::<$i>).softmax()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::softmax![$crate::softmax![$t,$i],$($is),+]
    }};
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
