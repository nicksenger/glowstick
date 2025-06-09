use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::squeeze, Shape};

use crate::{Error, Tensor};

/// Squeezes the specified dimensions from a tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{squeeze, Tensor};
/// use glowstick::{Shape4, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape4<U1, U2, U3, U1>>::ones(DType::F32, &device)?;
/// let squeezed = squeeze![a, U0, U3]?; // Squeezes dimensions 0 and 3
///
/// assert_eq!(squeezed.dims(), &[2, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! squeeze {
    [$t:expr,$i:ty] => {{
        glowstick::op::squeeze::check::<_, _, $i>(&$t);
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
            .and_then(|t| $crate::squeeze_next![t, $($is),+])
    }};
}
#[macro_export]
macro_rules! squeeze_next {
    [$t:expr,$i:ty] => {{
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<<$i as std::ops::Sub<glowstick::num::U1>>::Output>).squeeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
            .and_then(|t| $crate::squeeze_next![t, $($is),+])
    }};
}

pub trait Squeeze {
    type Out;
    fn squeeze(&self) -> Self::Out;
}
impl<S, Dim> Squeeze for (&Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): squeeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as squeeze::Compatible>::Out>, Error>;
    fn squeeze(&self) -> Self::Out {
        self.0.inner().squeeze(<Dim as Unsigned>::USIZE)?.try_into()
    }
}
impl<S, Dim> Squeeze for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): squeeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as squeeze::Compatible>::Out>, Error>;
    fn squeeze(&self) -> Self::Out {
        self.0.inner().squeeze(<Dim as Unsigned>::USIZE)?.try_into()
    }
}
