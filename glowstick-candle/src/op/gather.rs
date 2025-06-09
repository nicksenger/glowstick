use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{num::Unsigned, op::gather, Shape};

use crate::{Error, Tensor};

/// Gathers the elements from a tensor at the provided indices along a specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{gather, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U1, U1, U4>>::from_vec(vec![1f32, 2., 3., 4.], &device)?;
/// let b = Tensor::<Shape3<U1, U1, U2>>::from_vec(vec![1u32, 2], &device)?;
/// let gathered = gather!(a, b, U2)?;
///
/// assert_eq!(gathered.inner().to_vec3::<f32>()?, vec![vec![vec![2., 3.]]]);
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
    fn gather(&self) -> Self::Out;
}
impl<T1, S1, T2, S2, Dim> Gather for (T1, PhantomData<S1>, T2, PhantomData<S2>, PhantomData<Dim>)
where
    T1: Borrow<Tensor<S1>>,
    S1: Shape,
    T2: Borrow<Tensor<S2>>,
    S2: Shape,
    Dim: Unsigned,
    (S1, S2, Dim): gather::Compatible,
{
    type Out = Result<Tensor<<(S1, S2, Dim) as gather::Compatible>::Out>, Error>;
    fn gather(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .gather(self.2.borrow().inner(), <Dim as Unsigned>::USIZE)?
            .try_into()
    }
}

#[cfg(test)]
mod test_gather {
    #[test]
    fn gather() {
        use crate::gather;
        use crate::Tensor;
        use glowstick::num::{U1, U2, U4};
        type A = glowstick::shape![U1, U1, U4];
        type B = glowstick::shape![U1, U1, U2];
        let a: Tensor<A> = Tensor::from_vec(vec![1f32, 2., 3., 4.], &candle::Device::Cpu).unwrap();
        let b: Tensor<B> = Tensor::from_vec(vec![1u32, 2], &candle::Device::Cpu).unwrap();
        let gathered = gather!(a, b, U2).unwrap();
        let v = gathered.into_inner().to_vec3::<f32>().unwrap();
        assert_eq!(v, vec![vec![vec![2., 3.]]]);
    }
}
