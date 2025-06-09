use std::marker::PhantomData;

use burn::tensor::{BasicOps, Int, Numeric, Tensor as BTensor};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::cmp::Greater;
use glowstick::{
    num::{Unsigned, U0, U1},
    op::narrow,
    Shape,
};

use crate::Tensor;

/// Returns the indices of the maximum values along a specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{argmax, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let argmaxed = argmax!(a, U1);
///
/// assert_eq!(argmaxed.dims(), [2, 1, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! argmax {
    [$t:expr,$i:ty] => {{
        use $crate::op::argmax::ArgMax;
        ($t, std::marker::PhantomData::<$i>).argmax()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::argmax![$crate::argmax![$t,$i],$($is),+]
    }};
}

pub trait ArgMax {
    type Out;
    fn argmax(self) -> Self::Out;
}
impl<B, D, S, const N: usize, Dim> ArgMax for (Tensor<BTensor<B, N, D>, S>, PhantomData<Dim>)
where
    B: Backend,
    D: TensorKind<B> + BasicOps<B> + Numeric<B>,
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
    (S, Dim, U0, U1): narrow::Compatible,
{
    type Out = Tensor<BTensor<B, N, Int>, <(S, Dim, U0, U1) as narrow::Compatible>::Out>;
    fn argmax(self) -> Self::Out {
        Tensor(
            self.0.into_inner().argmax(<Dim as Unsigned>::USIZE),
            PhantomData,
        )
    }
}
