use std::marker::PhantomData;

use burn::tensor::{
    BasicOps, Int, Numeric,
    Tensor as BTensor,
};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::cmp::Greater;
use glowstick::{
    num::Unsigned, Shape,
};

use crate::Tensor;

/// Applies the sort-descending operation with indices to a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{sort_descending_with_indices, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let (sorted, indices) = sort_descending_with_indices!(a, U1);
///
/// assert_eq!(sorted.dims(), [2, 3, 4]);
/// assert_eq!(indices.dims(), [2, 3, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! sort_descending_with_indices {
    [$t:expr,$i:ty] => {{
        use $crate::op::sort_descending_with_indices::SortDescendingWithIndices;
        ($t, std::marker::PhantomData::<$i>).sort_descending_with_indices()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::sort_descending_with_indices![$crate::sort_descending_with_indices![$t,$i],$($is),+]
    }};
}

pub trait SortDescendingWithIndices {
    type Out;
    fn sort_descending_with_indices(self) -> Self::Out;
}
impl<B, D, S, const N: usize, Dim> SortDescendingWithIndices
    for (Tensor<BTensor<B, N, D>, S>, PhantomData<Dim>)
where
    B: Backend,
    D: TensorKind<B> + BasicOps<B> + Numeric<B>,
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = (Tensor<BTensor<B, N, D>, S>, Tensor<BTensor<B, N, Int>, S>);
    fn sort_descending_with_indices(self) -> Self::Out {
        let (t, i) = self
            .0
            .into_inner()
            .sort_descending_with_indices(<Dim as Unsigned>::USIZE);
        (Tensor(t, PhantomData), Tensor(i, PhantomData))
    }
}
