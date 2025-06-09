use std::marker::PhantomData;

use burn::tensor::{BasicOps, Tensor as BTensor};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::{num::Unsigned, op::squeeze, Shape};

use crate::Tensor;

/// Squeezes the specified dimensions from a tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{squeeze, Tensor};
/// use glowstick::{Shape4, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 4>, Shape4<U1, U2, U3, U1>>::ones(&device);
/// let squeezed = squeeze![a, U0, U3]; // Squeezes dimensions 0 and 3
///
/// assert_eq!(squeezed.dims(), [2, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! squeeze {
    [$t:expr,$i:ty] => {{
        glowstick::op::squeeze::check::<_, _, $i>(&$t);
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::squeeze_next![$crate::squeeze![$t,$i],$($is),+]
    }};
}
#[macro_export]
macro_rules! squeeze_next {
    [$t:expr,$i:ty] => {{
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<<$i as std::ops::Sub<glowstick::num::U1>>::Output>).squeeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::squeeze_next![$crate::squeeze_next![$t,$i],$($is),+]
    }};
}

pub trait Squeeze<const M: usize> {
    type Out;
    fn squeeze(self) -> Self::Out;
}
macro_rules! squeeze_impl {
    ($in:literal => $out:literal) => {
        impl<B, D, S, Dim> Squeeze<$out> for (Tensor<BTensor<B, $in, D>, S>, PhantomData<Dim>)
        where
            B: Backend,
            D: TensorKind<B> + BasicOps<B>,
            S: Shape,
            Dim: Unsigned,
            (S, Dim): squeeze::Compatible,
        {
            type Out = Tensor<BTensor<B, $out, D>, <(S, Dim) as squeeze::Compatible>::Out>;
            fn squeeze(self) -> Self::Out {
                Tensor::<BTensor<B, $out, D>, <(S, Dim) as squeeze::Compatible>::Out>(
                    self.0.into_inner().squeeze(<Dim as Unsigned>::USIZE),
                    PhantomData,
                )
            }
        }
    };
}
squeeze_impl!(8 => 7);
squeeze_impl!(7 => 6);
squeeze_impl!(6 => 5);
squeeze_impl!(5 => 4);
squeeze_impl!(4 => 3);
squeeze_impl!(3 => 2);
squeeze_impl!(2 => 1);
