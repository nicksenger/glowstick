use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::narrow_dyn, Shape};

use crate::{Error, Tensor};

#[allow(unused)]
pub trait NarrowDyn {
    type Out;
    fn narrow_dyn(&self) -> Self::Out;
}
impl<S, Dim, DynDim> NarrowDyn
    for (
        Tensor<S>,
        PhantomData<Dim>,
        PhantomData<DynDim>,
        usize,
        usize,
    )
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim, DynDim): narrow_dyn::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, DynDim) as narrow_dyn::Compatible>::Out>, Error>;
    fn narrow_dyn(&self) -> Self::Out {
        self.0
            .inner()
            .narrow(<Dim as Unsigned>::USIZE, self.3, self.4)?
            .try_into()
    }
}
impl<S, Dim, DynDim> NarrowDyn
    for (
        &Tensor<S>,
        PhantomData<Dim>,
        PhantomData<DynDim>,
        usize,
        usize,
    )
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim, DynDim): narrow_dyn::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, DynDim) as narrow_dyn::Compatible>::Out>, Error>;
    fn narrow_dyn(&self) -> Self::Out {
        self.0
            .inner()
            .narrow(<Dim as Unsigned>::USIZE, self.3, self.4)?
            .try_into()
    }
}
