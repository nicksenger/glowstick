use std::marker::PhantomData;

use burn::{
    prelude::Backend,
    tensor::{BasicOps, Int, Numeric, Tensor as BTensor, TensorKind},
};
use glowstick::{num::Unsigned, op::gather, Shape};

use crate::Tensor;

/// Gathers the elements from a tensor at the provided indices along a specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor, Int};
/// use glowstick_burn::{gather, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U1, U1, U4>>::from_data(vec![1., 2., 3., 4.], &device);
/// let b = Tensor::<BurnTensor<Backend, 3, Int>, Shape3<U1, U1, U2>>::from_data(vec![1, 2], &device);
/// let gathered = gather!(a, b, U2);
///
/// assert_eq!(gathered.inner().to_data().to_vec::<f32>().unwrap(), vec![2., 3.]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! gather {
    ($t1:expr,$t2:expr,$d:ty) => {{
        use $crate::op::gather::Gather;
        (
            $t1,
            std::marker::PhantomData,
            $t2,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d>,
        )
            .gather()
    }};
}

pub trait Gather {
    type Out;
    fn gather(self) -> Self::Out;
}
impl<B, D1, S1, S2, Dim, const N: usize> Gather
    for (
        Tensor<BTensor<B, N, D1>, S1>,
        PhantomData<S1>,
        Tensor<BTensor<B, N, Int>, S2>,
        PhantomData<S2>,
        PhantomData<Dim>,
    )
where
    B: Backend,
    D1: TensorKind<B> + BasicOps<B> + Numeric<B>,
    S1: Shape,
    S2: Shape,
    Dim: Unsigned,
    (S1, S2, Dim): gather::Compatible,
{
    type Out = Tensor<BTensor<B, N, D1>, <(S1, S2, Dim) as gather::Compatible>::Out>;
    fn gather(self) -> Self::Out {
        Tensor(
            self.0
                .into_inner()
                .gather(<Dim as Unsigned>::USIZE, self.2.into_inner()),
            PhantomData,
        )
    }
}
