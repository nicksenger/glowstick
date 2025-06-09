use std::marker::PhantomData;

use burn::tensor::Tensor as BTensor;
use burn::prelude::Backend;

use glowstick::cmp::Greater;
use glowstick::{
    num::Unsigned, Shape,
};

use crate::Tensor;

/// Computes the variance and mean of a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{var_mean, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let (variance, mean) = var_mean!(a, U1);
///
/// assert_eq!(variance.dims(), [2, 1, 4]);
/// assert_eq!(mean.dims(), [2, 1, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! var_mean {
    [$t:expr,$i:ty] => {{
        use $crate::op::var_mean::VarMean;
        ($t, std::marker::PhantomData::<$i>).var_mean()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::var_mean![$crate::var_mean![$t,$i],$($is),+]
    }};
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
