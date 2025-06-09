use std::marker::PhantomData;

use burn::tensor::{BasicOps, Tensor as BTensor};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::{num::Unsigned, op::unsqueeze, Shape};

use crate::Tensor;

/// Unsqueezes a tensor at the specified dimension(s).
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{unsqueeze, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U1, U4>>::ones(&device);
/// let unsqueezed = unsqueeze![a.clone(), U0, U4];
///
/// assert_eq!(unsqueezed.dims(), [1, 2, 1, 4, 1]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! unsqueeze {
    [$t:expr,$i:ty] => {{
        use $crate::op::unsqueeze::Unsqueeze;
        ($t, std::marker::PhantomData::<$i>).unsqueeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::unsqueeze![$crate::unsqueeze![$t,$i],$($is),+]
    }};
}

pub trait Unsqueeze<const M: usize> {
    type Out;
    fn unsqueeze(self) -> Self::Out;
}
macro_rules! unsqueeze_impl {
    ($in:literal => $out:literal) => {
        impl<B, S, D, Dim> Unsqueeze<$out> for (Tensor<BTensor<B, $in, D>, S>, PhantomData<Dim>)
        where
            B: Backend,
            S: Shape,
            D: TensorKind<B> + BasicOps<B>,
            Dim: Unsigned,
            (S, Dim): unsqueeze::Compatible,
        {
            type Out = Tensor<BTensor<B, $out, D>, <(S, Dim) as unsqueeze::Compatible>::Out>;
            fn unsqueeze(self) -> Self::Out {
                Tensor::<BTensor<B, $out, D>, <(S, Dim) as unsqueeze::Compatible>::Out>(
                    self.0.into_inner().unsqueeze_dim(<Dim as Unsigned>::USIZE),
                    PhantomData,
                )
            }
        }
    };
}
unsqueeze_impl!(7 => 8);
unsqueeze_impl!(6 => 7);
unsqueeze_impl!(5 => 6);
unsqueeze_impl!(4 => 5);
unsqueeze_impl!(3 => 4);
unsqueeze_impl!(2 => 3);
unsqueeze_impl!(1 => 2);
