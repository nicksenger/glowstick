use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::unsqueeze, Shape};

use crate::{Error, Tensor};

/// Unsqueezes a tensor at the specified dimension(s).
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{unsqueeze, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U1, U4>>::ones(DType::F32, &device)?;
/// let unsqueezed = unsqueeze![a, U0, U4]?;
///
/// assert_eq!(unsqueezed.dims(), &[1, 2, 1, 4, 1]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! unsqueeze {
    [$t:expr,$i:ty] => {{
        use $crate::op::unsqueeze::Unsqueeze;
        ($t, std::marker::PhantomData::<$i>).unsqueeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        use $crate::op::unsqueeze::Unsqueeze;
        ($t, std::marker::PhantomData::<$i>).unsqueeze()
            .and_then(|t| $crate::unsqueeze![t, $($is),+])
    }};
}

#[allow(unused)]
pub trait Unsqueeze {
    type Out;
    fn unsqueeze(&self) -> Self::Out;
}
impl<S, Dim> Unsqueeze for (&Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): unsqueeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as unsqueeze::Compatible>::Out>, Error>;
    fn unsqueeze(&self) -> Self::Out {
        self.0
            .inner()
            .unsqueeze(<Dim as Unsigned>::USIZE)?
            .try_into()
    }
}
impl<S, Dim> Unsqueeze for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): unsqueeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as unsqueeze::Compatible>::Out>, Error>;
    fn unsqueeze(&self) -> Self::Out {
        self.0
            .inner()
            .unsqueeze(<Dim as Unsigned>::USIZE)?
            .try_into()
    }
}
