use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::Tensor as BTensor;

use glowstick::{num::Unsigned, op::transpose, Shape};

use crate::Tensor;

/// Swaps the dimensions of a tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{transpose, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let transposed = transpose!(a, U1, U2);
///
/// assert_eq!(transposed.dims(), [2, 4, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! transpose {
    ($t:expr,$d1:ty,$d2:ty) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
    ($t:expr,$d1:ty:$d2:ty) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
    ($t:expr,$d1:ty:$d2:ty,$($d1s:ty:$d2s:ty),+) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose().and_then(|t| $crate::transpose!(&t, $($d1s:$d2s),+))
    }};
}

pub trait Transpose {
    type Out;
    fn transpose(self) -> Self::Out;
}
impl<B, S, Dim1, Dim2, const N: usize> Transpose
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
    (S, Dim1, Dim2): transpose::Compatible,
{
    type Out = Tensor<BTensor<B, N>, <(S, Dim1, Dim2) as transpose::Compatible>::Out>;
    fn transpose(self) -> Self::Out {
        Tensor(
            self.0.into_inner().swap_dims(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            ),
            PhantomData,
        )
    }
}
