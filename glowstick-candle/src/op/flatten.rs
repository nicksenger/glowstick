use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::flatten, Shape};

use crate::{Error, Tensor};

/// Flattens the given tensor from the specified start dimension to the end
/// dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{flatten, Tensor};
/// use glowstick::{Shape4, num::*, dyndims};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape4<U1, U4, U3, U2>>::ones(DType::F32, &device)?;
/// let flattened = flatten!(a, [U0, U2])?;
///
/// assert_eq!(flattened.dims(), &[12, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! flatten {
    ($t:expr,[$d1:ty,$d2:ty]) => {{
        use $crate::op::flatten::Flatten;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten()
    }};
    ($t:expr,[$d1:ty,$d2:ty],$([$d1s:ty,$d2s:ty]),+) => {{
        use $crate::op::flatten::Flatten;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten().and_then(|t| $crate::flatten!(&t, $([$d1s,$d2s]),+))
    }};
}

pub trait Flatten {
    type Out;
    fn flatten(&self) -> Self::Out;
}
impl<S, Dim1, Dim2> Flatten for (Tensor<S>, PhantomData<Dim1>, PhantomData<Dim2>)
where
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): flatten::Compatible,
{
    type Out = Result<Tensor<<(S, Dim1, Dim2) as flatten::Compatible>::Out>, Error>;
    fn flatten(&self) -> Self::Out {
        self.0
            .inner()
            .flatten(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            )?
            .try_into()
    }
}
impl<S, Dim1, Dim2> Flatten for (&Tensor<S>, PhantomData<Dim1>, PhantomData<Dim2>)
where
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): flatten::Compatible,
{
    type Out = Result<Tensor<<(S, Dim1, Dim2) as flatten::Compatible>::Out>, Error>;
    fn flatten(&self) -> Self::Out {
        self.0
            .inner()
            .flatten(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            )?
            .try_into()
    }
}
