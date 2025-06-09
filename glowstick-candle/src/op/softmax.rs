use std::marker::PhantomData;

use glowstick::cmp::Greater;
use glowstick::num::Unsigned;
use glowstick::Shape;

use crate::Tensor;

/// Applies the softmax function to a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{softmax, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let softmaxed = softmax!(a, U1)?;
///
/// assert_eq!(softmaxed.dims(), &[2, 3, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! softmax {
    ($t:expr,$i:ty) => {{
        use $crate::op::softmax::Softmax;
        ($t, std::marker::PhantomData::<$i>).softmax()
    }};
    ($t:expr,$i:ty,$($is:ty),+) => {{
        $crate::softmax!($crate::softmax!($t,$i),$($is),+)
    }};
}

pub trait Softmax {
    type Out;
    fn softmax(self) -> Self::Out;
}
impl<S, Dim> Softmax for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn softmax(self) -> Self::Out {
        Ok(Tensor(
            candle_nn::ops::softmax(self.0.inner(), <Dim as Unsigned>::USIZE)?,
            PhantomData,
        ))
    }
}
