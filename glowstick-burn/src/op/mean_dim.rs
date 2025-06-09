use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::Tensor as BTensor;

use glowstick::{
    num::{Unsigned, U0, U1},
    op::narrow,
    Shape,
};

use crate::Tensor;

/// Computes the mean of a tensor along a specified dimension, resulting in a tensor with size `U1` at that dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{mean_dim, Tensor};
/// use glowstick::{Shape4, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 4>, Shape4<U2, U3, U4, U5>>::ones(&device);
/// let meaned = mean_dim!(a, U1);
///
/// assert_eq!(meaned.dims(), [2, 1, 4, 5]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! mean_dim {
    [$t:expr,$i:ty] => {{
        use $crate::op::mean_dim::MeanDim;
        ($t, std::marker::PhantomData::<$i>).mean_dim()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::mean_dim![$crate::mean_dim![$t,$i],$($is),+]
    }};
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
