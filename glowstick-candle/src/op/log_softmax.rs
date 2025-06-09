use std::marker::PhantomData;

use glowstick::cmp::Greater;
use glowstick::num::Unsigned;
use glowstick::Shape;

use crate::Tensor;

/// Applies the log softmax function to a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{log_softmax, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let logsoftmaxed = log_softmax!(a, U1)?;
///
/// assert_eq!(logsoftmaxed.dims(), &[2, 3, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! log_softmax {
    ($t:expr,$i:ty) => {{
        use $crate::op::log_softmax::LogSoftmax;
        ($t, std::marker::PhantomData::<$i>).log_softmax()
    }};
    ($t:expr,$i:ty,$($is:ty),+) => {{
        $crate::log_softmax!($crate::log_softmax!($t,$i),$($is),+)
    }};
}

pub trait LogSoftmax {
    type Out;
    fn log_softmax(self) -> Self::Out;
}
impl<S, Dim> LogSoftmax for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn log_softmax(self) -> Self::Out {
        Ok(Tensor(
            candle_nn::ops::log_softmax(self.0.inner(), <Dim as Unsigned>::USIZE)?,
            PhantomData,
        ))
    }
}

#[cfg(test)]
mod test_logsoft {
    #[test]
    fn logsoft() {
        use crate::log_softmax;
        use crate::Tensor;
        use glowstick::num::{U0, U4};
        type TestShape = glowstick::shape![U4];
        let ct = candle::Tensor::from_vec(vec![0., 1., 2., 3.], 4, &candle::Device::Cpu).unwrap();
        let gt: Tensor<TestShape> = ct.clone().try_into().unwrap();
        let c_softmaxed: Vec<f64> = candle_nn::ops::log_softmax(&ct, 0)
            .unwrap()
            .to_vec1()
            .unwrap();
        let g_softmaxed: Vec<f64> = log_softmax!(gt, U0)
            .unwrap()
            .into_inner()
            .to_vec1()
            .unwrap();
        assert_eq!(c_softmaxed, g_softmaxed);
    }
}
