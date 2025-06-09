use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::Tensor as BTensor;

use glowstick::{num::Unsigned, op::flatten, Shape};

use crate::Tensor;

/// Flattens the given tensor from the specified start dimension to the end
/// dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{flatten, Tensor};
/// use glowstick::{Shape4, num::*, dyndims};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 4>, Shape4<U1, U4, U3, U2>>::ones(&device);
/// let flattened = flatten!(a.clone(), [U0, U2]);
///
/// assert_eq!(flattened.dims(), [12, 2]);
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
        let t = (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten();

        $crate::flatten!(&t, $([$d1s,$d2s]),+)
    }};
}

pub trait Flatten<const M: usize> {
    type Out;
    fn flatten(self) -> Self::Out;
}
impl<B, S, Dim1, Dim2, const N: usize, const M: usize> Flatten<M>
    for (
        Tensor<BTensor<B, N>, S>,
        PhantomData<Dim1>,
        PhantomData<Dim2>,
    )
where
    B: Backend,
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): flatten::Compatible,
{
    type Out = Tensor<BTensor<B, M>, <(S, Dim1, Dim2) as flatten::Compatible>::Out>;
    fn flatten(self) -> Self::Out {
        Tensor(
            self.0.into_inner().flatten(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            ),
            PhantomData,
        )
    }
}
