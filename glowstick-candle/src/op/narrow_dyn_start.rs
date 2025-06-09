use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{Shape, num::Unsigned, op::narrow_dyn_start};

use crate::{Error, Tensor};

pub trait NarrowDynStart {
    type Out;
    fn narrow_dyn_start(&self) -> Self::Out;
}
impl<T, S, Dim, Len> NarrowDynStart
    for (T, PhantomData<S>, PhantomData<Dim>, usize, PhantomData<Len>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    Len: Unsigned,
    (S, Dim, Len): narrow_dyn_start::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, Len) as narrow_dyn_start::Compatible>::Out>, Error>;
    fn narrow_dyn_start(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .narrow(<Dim as Unsigned>::USIZE, self.3, <Len as Unsigned>::USIZE)?
            .try_into()
    }
}
