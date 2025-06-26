use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{
    num::{Unsigned, U0, U1},
    op::narrow,
    Shape,
};

use crate::{Error, Tensor};

/// Computes the argmax of a tensor along a specified dimension, resulting in a tensor with size `U1` at that dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{argmax_dim, Tensor};
/// use glowstick::{Shape4, num::{U1, U2, U3, U4, U5}, dyndims};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape4<U2, U3, U4, U5>>::ones(DType::F32, &device)?;
/// let argmaxed = argmax_dim!(a, U1)?;
///
/// assert_eq!(argmaxed.dims(), &[2, 1, 4, 5]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! argmax_dim {
    [$t:expr,$i:ty] => {{
        use $crate::op::argmax_dim::ArgMaxDim;
        ($t, std::marker::PhantomData, std::marker::PhantomData::<$i>).argmax_dim()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::argmax_dim![$crate::argmax_dim![$t,$i],$($is),+]
    }};
}

pub trait ArgMaxDim {
    type Out;
    fn argmax_dim(self) -> Self::Out;
}
impl<T, S, Dim> ArgMaxDim for (T, PhantomData<S>, PhantomData<Dim>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    (S, Dim, U0, U1): narrow::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, U0, U1) as narrow::Compatible>::Out>, Error>;
    fn argmax_dim(self) -> Self::Out {
        Ok(Tensor(
            self.0
                .borrow()
                .inner()
                .argmax_keepdim(<Dim as Unsigned>::USIZE)?,
            PhantomData,
        ))
    }
}
