use std::marker::PhantomData;

use glowstick::{op::broadcast, Shape};

use crate::{Error, Tensor};

/// Broadcasts the lefthand tensor to the shape of the provided righthand tensor
/// or shape.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{expand, Tensor};
/// use glowstick::{Shape2, Shape4, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U1, U2>>::ones(DType::F32, &device)?;
/// let b = Tensor::<Shape4<U1, U4, U3, U2>>::ones(DType::F32, &device)?;
/// let c = expand!(&a, &b)?;
/// let d = expand!(&a, [U1, U4, U3, U2])?;
///
/// assert_eq!(c.dims(), &[1, 4, 3, 2]);
/// assert_eq!(d.dims(), &[1, 4, 3, 2]);
/// # Ok(())
/// # }
/// ```
///
/// When broadcasting to a shape, a combination of type-level integers and
/// expressions bound to dynamic dimensions may be provided.
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{expand, Tensor};
/// use glowstick::{Shape2, Shape4, num::{U1, U2, U3, U4}, dyndims};
///
/// dyndims! {
///     B: BatchSize,
///     N: SequenceLength
/// }
///
/// let device = Device::Cpu;
/// let [batch_size, seq_len] = [4, 12];
/// let a = Tensor::<Shape2<U1, U2>>::ones(DType::F32, &device)?;
/// let b = expand!(&a, [{ batch_size } => B, { seq_len } => N, U3, U2])?;
///
/// assert_eq!(b.dims(), &[4, 12, 3, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! expand {
    ($t:expr,[$($ds:tt)+]) => {{
        use $crate::op::expand::BroadcastAs;
        (
            $t,
            std::marker::PhantomData::<glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>>,
        )
            .expand(&candle::Shape::from_dims(&$crate::reshape_val!($($ds)+).into_array()))
    }};
    ($t1:expr,$t2:expr) => {{
        use $crate::op::expand::BroadcastAs;
        (
            $t1,
            $t2,
        )
            .expand($t2.inner().shape())
    }}
}

pub trait BroadcastAs {
    type Out;
    fn expand(&self, shape: &candle::Shape) -> Self::Out;
}

impl<S1, S2> BroadcastAs for (Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &candle::Shape) -> Self::Out {
        self.0.inner().expand(shape)?.try_into()
    }
}

impl<S1, S2> BroadcastAs for (Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &candle::Shape) -> Self::Out {
        self.0.inner().expand(shape)?.try_into()
    }
}

impl<S1, S2> BroadcastAs for (&Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &candle::Shape) -> Self::Out {
        self.0.inner().expand(shape)?.try_into()
    }
}

impl<S1, S2> BroadcastAs for (&Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &candle::Shape) -> Self::Out {
        self.0.inner().expand(shape)?.try_into()
    }
}

impl<S1, S2> BroadcastAs for (&Tensor<S1>, PhantomData<S2>)
where
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &candle::Shape) -> Self::Out {
        self.0.inner().expand(shape)?.try_into()
    }
}
