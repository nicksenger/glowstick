use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::Tensor as BTensor;

use glowstick::{op::matmul, Shape, TensorShape};

use crate::Tensor;

/// Performs matrix multiplication of the lefthand tensor and righthand tensor(s).
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
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
macro_rules! matmul {
    ($t1:expr,$t2:expr) => {{
        use $crate::op::matmul::Matmul;
        ($t1, $t2).matmul()
    }};
    ($t1:expr,$t2:expr,$($t2s:expr),+) => {{
        $crate::matmul![$crate::matmul!($t1, $t2),$($t2s),+]
    }};
}

pub trait Matmul {
    type Out;
    fn matmul(self) -> Self::Out;
}
impl<B, S1, S2, const N: usize> Matmul for (Tensor<BTensor<B, N>, S1>, Tensor<BTensor<B, N>, S2>)
where
    B: Backend,
    S1: Shape + matmul::Operand,
    S2: Shape + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type Out = Tensor<BTensor<B, N>, TensorShape<<(S1, S2) as matmul::Compatible>::Out>>;
    fn matmul(self) -> Self::Out {
        Tensor(self.0.into_inner().matmul(self.1.into_inner()), PhantomData)
    }
}
