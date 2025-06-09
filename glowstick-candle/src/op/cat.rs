use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::cat_dyn, Shape};

use crate::{Error, Tensor};

/// Concatenates the given tensors along a specified dimension.
/// A dynamic dimension must be provided for the return type.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{cat, Tensor};
/// use glowstick::{Shape4, num::*, dyndims};
///
/// dyndims! {
///     B: BatchSize
/// }
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape4<U1, U4, U3, U2>>::ones(DType::F32, &device)?;
/// let b = Tensor::<Shape4<U1, U4, U3, U2>>::ones(DType::F32, &device)?;
/// let concatenated = cat!(vec![a, b].as_slice(), U0 => B)?;
///
/// assert_eq!(concatenated.dims(), &[2, 4, 3, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! cat {
    ($ts:expr,$i:ty => $d:ty) => {{
        use $crate::op::cat::Cat;
        (
            $ts,
            std::marker::PhantomData::<$i>,
            std::marker::PhantomData::<$d>,
        )
            .cat()
    }};
}

pub trait Cat {
    type Out;
    fn cat(self) -> Self::Out;
}
impl<S, I, D> Cat for (&[Tensor<S>], PhantomData<I>, PhantomData<glowstick::Dyn<D>>)
where
    S: Shape,
    (S, I, glowstick::Dyn<D>): cat_dyn::Compatible,
    I: Unsigned,
{
    type Out = Result<Tensor<<(S, I, glowstick::Dyn<D>) as cat_dyn::Compatible>::Out>, Error>;
    fn cat(self) -> Self::Out {
        candle::Tensor::cat(self.0, <I as Unsigned>::USIZE)?.try_into()
    }
}
impl<S, I, D> Cat
    for (
        &[&Tensor<S>],
        PhantomData<I>,
        PhantomData<glowstick::Dyn<D>>,
    )
where
    S: Shape,
    (S, I, glowstick::Dyn<D>): cat_dyn::Compatible,
    I: Unsigned,
{
    type Out = Result<Tensor<<(S, I, glowstick::Dyn<D>) as cat_dyn::Compatible>::Out>, Error>;
    fn cat(self) -> Self::Out {
        candle::Tensor::cat(self.0, <I as Unsigned>::USIZE)?.try_into()
    }
}
