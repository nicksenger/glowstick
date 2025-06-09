use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{num::Unsigned, op::narrow, Shape};

use crate::{Error, Tensor};

/// Narrows a tensor at the specified dimension from start index to length.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{narrow, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let narrowed = narrow!(a, U0: [U1, U1])?;
///
/// assert_eq!(narrowed.dims(), &[1, 3, 4]);
/// # Ok(())
/// # }
/// ```
///
/// When using dynamic start and length, the resulting tensor's shape will be determined by the provided expressions.
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{narrow, Tensor};
/// use glowstick::{Shape3, num::{U0, U1, U2, U3, U4}, dyndims};
///
/// dyndims! {
///     N: SequenceLength
/// }
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let [start, len] = [1, 2];
/// let narrowed = narrow!(a, U1: [{ start }, { len }] => N)?;
///
/// assert_eq!(narrowed.dims(), &[2, 2, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! narrow {
    ($t:expr,$d:ty:[$s:ty,$l:ty]) => {{
        glowstick::op::narrow::check::<_, _, $d, $s, $l>(&$t);
        use $crate::op::narrow::Narrow;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$s>,
            std::marker::PhantomData::<$l>
        ).narrow()
    }};
    ($t:expr,$d:ty:[$s:expr,$l:ty]) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $l>(&$t);
        use $crate::op::narrow_dyn_start::NarrowDynStart;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow_dyn_start()
    }};
    ($t:expr,$d:ty:[$s:expr,$l:expr] => $y:ty) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $y>(&$t);
        use $crate::op::narrow_dyn::NarrowDyn;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$y>,
            $s,
            $l,
        )
            .narrow_dyn()
    }};
    ($t:expr,$d:ty:[$s:ty,$l:ty],$($ds:tt)+) => {{
        glowstick::op::narrow::check::<_, _, $d, $s, $l>(&$t);
        use $crate::op::narrow::Narrow;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$s>,
            std::marker::PhantomData::<$l>,
        )
            .narrow().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
    ($t:expr,$d:ty:[$s:ty,$l:ty],$($ds:tt)+) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $l>(&$t);
        use $crate::op::narrow_dyn_start::NarrowDynStart;
        (
            $t,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow_dyn_start().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
    ($t:expr,$d:ty:[$s:expr,$l:expr] => $y:ty,$($ds:tt)+) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $y>(&$t);
        use $crate::op::narrow_dyn::NarrowDyn;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$y>,
            $s,
            $l,
        )
            .narrow_dyn().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
}

#[allow(unused)]
pub trait Narrow {
    type Out;
    fn narrow(&self) -> Self::Out;
}
impl<T, S, Dim, Start, Len> Narrow
    for (
        T,
        PhantomData<S>,
        PhantomData<Dim>,
        PhantomData<Start>,
        PhantomData<Len>,
    )
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    Start: Unsigned,
    Len: Unsigned,
    (S, Dim, Start, Len): narrow::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, Start, Len) as narrow::Compatible>::Out>, Error>;
    fn narrow(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .narrow(
                <Dim as Unsigned>::USIZE,
                <Start as Unsigned>::USIZE,
                <Len as Unsigned>::USIZE,
            )?
            .try_into()
    }
}
