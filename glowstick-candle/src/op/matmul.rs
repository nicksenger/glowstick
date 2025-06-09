use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{op::matmul, Shape, TensorShape};

use crate::{Error, Tensor};

/// Performs matrix multiplication of the lefthand tensor and righthand tensor(s).
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{matmul, Tensor};
/// use glowstick::{Shape2, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U2, U1>>::from_vec(vec![4f32, 5.], &device)?;
/// let b = Tensor::<Shape2<U1, U2>>::from_vec(vec![5f32, 4.], &device)?;
/// let c = matmul!(a, b)?;
///
/// assert_eq!(
///     c.inner().to_vec2::<f32>()?,
///     vec![
///         vec![20., 16.],
///         vec![25., 20.]
///     ]
/// );
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! matmul {
    ($t1:expr,$t2:expr) => {{
        use $crate::op::matmul::Matmul;
        ($t1, $t2, std::marker::PhantomData).matmul()
    }};
    ($t1:expr,$t2:expr,$($t2s:expr),+) => {{
        use $crate::op::matmul::Matmul;
        ($t1, $t2, std::marker::PhantomData)
            .matmul()
            .and_then(|t| $crate::matmul!(&t, $t2s))
    }};
}

pub trait Matmul {
    type Out;
    fn matmul(self) -> Self::Out;
}
impl<S1, U, S2> Matmul for (Tensor<S1>, U, PhantomData<S2>)
where
    U: Borrow<Tensor<S2>>,
    S1: Shape + matmul::Operand,
    S2: Shape + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type Out = Result<Tensor<TensorShape<<(S1, S2) as matmul::Compatible>::Out>>, Error>;
    fn matmul(self) -> Self::Out {
        self.0
            .into_inner()
            .matmul(self.1.borrow().inner())?
            .try_into()
    }
}
impl<S1, U, S2> Matmul for (&Tensor<S1>, U, PhantomData<S2>)
where
    U: Borrow<Tensor<S2>>,
    S1: Shape + matmul::Operand,
    S2: Shape + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type Out = Result<Tensor<TensorShape<<(S1, S2) as matmul::Compatible>::Out>>, Error>;
    fn matmul(self) -> Self::Out {
        self.0.inner().matmul(self.1.borrow().inner())?.try_into()
    }
}
