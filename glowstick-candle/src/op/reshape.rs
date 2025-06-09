use std::{borrow::Borrow, marker::PhantomData};

use candle::shape::ShapeWithOneHole;
use glowstick::{op::reshape, Shape};

use crate::{Error, Tensor};

/// Reshapes a tensor to the specified dimensions.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{reshape, Tensor};
/// use glowstick::{Shape2, Shape4, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U2, U3>>::ones(DType::F32, &device)?;
/// let reshaped = reshape!(a, [U1, U6])?;
///
/// assert_eq!(reshaped.dims(), &[1, 6]);
/// # Ok(())
/// # }
/// ```
///
/// When using dynamic dimensions, the resulting tensor's shape will be determined by the provided expressions.
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{reshape, Tensor};
/// use glowstick::{Shape2, num::*, dyndims};
///
/// dyndims! {
///     A: Rows,
///     B: Cols
/// }
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U1, U4>>::ones(DType::F32, &device)?;
/// let [rows, cols] = [2, 2];
/// let reshaped = reshape!(a, [{ rows } => A, { cols } => B])?;
///
/// assert_eq!(reshaped.dims(), &[2, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! reshape {
    ($t:expr,[$($ds:tt)+]) => {{
        type TS = glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>;
        glowstick::op::reshape::check::<_, _, TS>(&$t);
        use $crate::op::reshape::Reshape;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<TS>,
        )
            .reshape($crate::reshape_val!($($ds)+).tuplify())
    }};
}

pub trait Reshape {
    type Out;
    fn reshape<Args: ShapeWithOneHole>(&self, args: Args) -> Self::Out;
}

impl<T, S1, S2> Reshape for (T, PhantomData<S1>, PhantomData<S2>)
where
    T: Borrow<Tensor<S1>>,
    S1: Shape,
    S2: Shape,
    (S1, S2): reshape::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as reshape::Compatible>::Out>, Error>;
    fn reshape<Args: ShapeWithOneHole>(&self, args: Args) -> Self::Out {
        self.0.borrow().inner().reshape(args)?.try_into()
    }
}

#[macro_export]
macro_rules! reshape_tys {
    ($e:expr => $d:ty) => {
        glowstick::Shp<(<$d as glowstick::dynamic::Dim>::Id, glowstick::Empty)>
    };
    ($e:expr => $d:ty,$($ds:tt)+) => {
        glowstick::Shp<(<$d as glowstick::dynamic::Dim>::Id, $crate::reshape_tys!($($ds)+))>
    };
    ($d:ty) => {
        glowstick::Shp<($d, glowstick::Empty)>
    };
    ($d:ty,$($ds:tt)+) => {
        glowstick::Shp<($d, $crate::reshape_tys!($($ds)+))>
    };
}
#[macro_export]
macro_rules! reshape_val {
    ($e:expr => $d:ty) => {
        glowstick::ValueList(($e, glowstick::ValueList(())))
    };
    ($d:ty) => {
        glowstick::ValueList((<$d as glowstick::num::Unsigned>::USIZE,glowstick::ValueList(())))
    };
    ($e:expr => $d:ty,$($ds:tt)+) => {
        glowstick::ValueList(($e,$crate::reshape_val!($($ds)+)))
    };
    ($d:ty,$($ds:tt)+) => {
        glowstick::ValueList((<$d as glowstick::num::Unsigned>::USIZE,$crate::reshape_val!($($ds)+)))
    };
}
