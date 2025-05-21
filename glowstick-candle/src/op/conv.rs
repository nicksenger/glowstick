use std::marker::PhantomData;

use crate::{Error, Tensor};
use glowstick::op::convolution::IsCompatible;
use glowstick::{
    Indexed, Shape, ShapeDiagnostic, ShapeFragment,
    num::{U0, Unsigned},
    op::convolution,
};

#[macro_export]
macro_rules! conv2d {
    [$t:expr,$k:expr,$p:ty,$d:ty,$s:ty,$g:expr] => {{
        use $crate::op::conv::Conv2d;
        use std::marker::PhantomData;
        type Pad = glowstick::list![$p, $p];
        type Dilation = glowstick::list![$d, $d];
        type Stride = glowstick::list![$s, $s];
        ($t, $k, PhantomData::<Pad>, PhantomData::<Pad>, PhantomData::<Dilation>, PhantomData::<Stride>, $g)
            .conv2d()
    }}
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
