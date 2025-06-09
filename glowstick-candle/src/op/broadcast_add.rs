use std::borrow::Borrow;

use glowstick::{op::broadcast, Shape};

use crate::{Error, Tensor};

/// Performs addition of the lefthand tensor and righthand tensor(s). The righthand
/// tensor(s) must be compatible for broadcast to the shape of the lefthand tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{broadcast_add, Tensor};
/// use glowstick::{Shape1, Shape2, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U1, U2>>::ones(DType::F32, &device)?;
/// let b = Tensor::<Shape2<U2, U2>>::ones(DType::F32, &device)?;
/// let c = broadcast_add!(a, b)?;
///
/// assert_eq!(
///     c.inner().to_vec2::<f32>()?,
///     vec![
///         vec![2., 2.],
///         vec![2., 2.]
///     ]
/// );
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! broadcast_add {
    ($t1:expr,$t2:expr) => {{
        use $crate::op::broadcast_add::BroadcastAdd;
        ($t1, $t2).broadcast_add()
    }};
    ($t1:expr,$t2:expr,$($t2s:expr),+) => {{
        use $crate::op::broadcast_add::BroadcastAdd;
        ($t1, $t2)
            .broadcast_add()
            .and_then(|t| $crate::broadcast_add!(&t, $t2s))
    }};
}

pub trait BroadcastAdd {
    type Out;
    fn broadcast_add(&self) -> Self::Out;
}
impl<S1, S2> BroadcastAdd for (Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0
            .inner()
            .broadcast_add(self.1.borrow().inner())?
            .try_into()
    }
}
impl<S1, S2> BroadcastAdd for (Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0.inner().broadcast_add(self.1.inner())?.try_into()
    }
}
impl<S1, S2> BroadcastAdd for (&Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0
            .inner()
            .broadcast_add(self.1.borrow().inner())?
            .try_into()
    }
}
impl<S1, S2> BroadcastAdd for (&Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0.inner().broadcast_add(self.1.inner())?.try_into()
    }
}
