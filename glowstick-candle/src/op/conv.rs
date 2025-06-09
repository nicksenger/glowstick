use std::marker::PhantomData;

use crate::{Error, Tensor};
use glowstick::op::convolution::IsCompatible;
use glowstick::{
    num::{Unsigned, U0},
    op::convolution,
    Indexed, Shape, ShapeDiagnostic, ShapeFragment,
};

/// Applies a 2D convolution over the input tensor with the provided kernel, padding,
/// dilation, stride and groups.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{conv2d, Tensor};
/// use glowstick::{Shape4, num::*};
///
/// let device = Device::Cpu;
/// let input = Tensor::<Shape4<U2, U2, U5, U5>>::ones(DType::F32, &device)?;
/// let kernel = Tensor::<Shape4<U4, U2, U3, U3>>::ones(DType::F32, &device)?;
/// let convolved = conv2d!(input, kernel, U0, U1, U1, 1)?;
///
/// assert_eq!(convolved.dims(), &[2, 4, 3, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! conv2d {
    ($t:expr,$kernel:expr,$padding:ty,$dilation:ty,$stride:ty,$groups:expr) => {{
        use std::marker::PhantomData;
        use $crate::op::conv::Conv2d;
        type Pad = glowstick::list![$padding, $padding];
        type Dilation = glowstick::list![$dilation, $dilation];
        type Stride = glowstick::list![$stride, $stride];
        (
            $t,
            $kernel,
            PhantomData::<Pad>,
            PhantomData::<Pad>,
            PhantomData::<Dilation>,
            PhantomData::<Stride>,
            $groups,
        )
            .conv2d()
    }};
}

pub trait Conv2d {
    type Out;
    fn conv2d(self) -> Self::Out;
}

use convolution::Kernel;
use glowstick::num::U1;
use glowstick::num::{Sub, ZipSubOneMul};
use glowstick::{Container, Empty, List, Map, Mappend, TakeFragment};

impl<T, K, P1, P2, S, D> Conv2d
    for (
        Tensor<T>,
        Tensor<K>,
        PhantomData<P1>,
        PhantomData<P2>,
        PhantomData<D>,
        PhantomData<S>,
        usize,
    )
where
    (T, K, P1, P2, S, D): convolution::IsCompatible,
    (P1, U0): Indexed,
    (S, U0): Indexed,
    (D, U0): Indexed,
    <(P1, U0) as Indexed>::Out: Unsigned,
    <(S, U0) as Indexed>::Out: Unsigned,
    <(D, U0) as Indexed>::Out: Unsigned,
    T: Shape + ShapeDiagnostic,
    K: Kernel<D> + ShapeDiagnostic,
    (T, K, P1, P2, S, D): IsCompatible,
    (
        <T as Shape>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): Sub,
    (
        <(
            <T as Shape>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as Sub>::Out,
        U1,
    ): Sub,
    <K as Kernel<D>>::DilateZipped: Container,
    (<K as Kernel<D>>::DilateZipped, ZipSubOneMul):
        Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>,
    (
        T,
        <(
            <(
                <T as Shape>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as Sub>::Out,
            U1,
        ) as Sub>::Out,
    ): TakeFragment,
    (
        <(
            T,
            <(
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
                U1,
            ) as Sub>::Out,
        ) as TakeFragment>::Out,
        List<(<K as Kernel<D>>::M, Empty)>,
    ): Mappend,
    (T, K, P1, P2, S, D): convolution::Compatible,
{
    type Out = Result<Tensor<<(T, K, P1, P2, S, D) as convolution::Compatible>::Out>, Error>;

    fn conv2d(self) -> Self::Out {
        let p = <<(P1, U0) as Indexed>::Out as glowstick::num::Unsigned>::USIZE;
        let s = <<(S, U0) as Indexed>::Out as glowstick::num::Unsigned>::USIZE;
        let d = <<(D, U0) as Indexed>::Out as glowstick::num::Unsigned>::USIZE;
        self.0
            .inner()
            .conv2d(self.1.inner(), p, s, d, self.6)?
            .try_into()
    }
}
