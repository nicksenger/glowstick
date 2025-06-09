use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{num::Unsigned, op::transpose, Shape};

use crate::{Error, Tensor};

/// Swaps the dimensions of a tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{transpose, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let transposed = transpose!(a, U1, U2)?;
///
/// assert_eq!(transposed.dims(), &[2, 4, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! transpose {
    ($t:expr,$d1:ty,$d2:ty) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
    ($t:expr,$d1:ty:$d2:ty) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
    ($t:expr,$d1:ty:$d2:ty,$($d1s:ty:$d2s:ty),+) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose().and_then(|t| $crate::transpose!(&t, $($d1s:$d2s),+))
    }};
}

pub trait Transpose {
    type Out;
    fn transpose(&self) -> Self::Out;
}
impl<T, S, Dim1, Dim2> Transpose for (T, PhantomData<S>, PhantomData<Dim1>, PhantomData<Dim2>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): transpose::Compatible,
{
    type Out = Result<Tensor<<(S, Dim1, Dim2) as transpose::Compatible>::Out>, Error>;
    fn transpose(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .transpose(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            )?
            .try_into()
    }
}
