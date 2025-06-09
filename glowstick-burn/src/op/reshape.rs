use std::marker::PhantomData;

use burn::tensor::{BasicOps, ReshapeArgs, Tensor as BTensor};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::ShapeFragment;
use glowstick::{op::reshape, Shape, TensorShape};

use crate::Tensor;

/// Reshapes a tensor to the specified dimensions.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{reshape, Tensor};
/// use glowstick::{Shape2, Shape4, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 2>, Shape2<U2, U3>>::ones(&device);
/// let reshaped = reshape!(a.clone(), [U1, U6]);
///
/// assert_eq!(reshaped.dims(), [1, 6]);
/// # Ok(())
/// # }
/// ```
///
/// When using dynamic dimensions, the resulting tensor's shape will be determined by the provided expressions.
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{reshape, Tensor};
/// use glowstick::{Shape2, num::*, dyndims};
///
/// dyndims! {
///     A: Rows,
///     B: Cols
/// }
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 2>, Shape2<U1, U4>>::ones(&device);
/// let [rows, cols] = [2, 2];
/// let reshaped = reshape!(a.clone(), [{ rows } => A, { cols } => B]);
///
/// assert_eq!(reshaped.dims(), [2, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! reshape {
    ($t:expr,[$($ds:tt)+]) => {{
        type TS = glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>;
        glowstick::op::reshape::check::<_, _, TS>(&$t);
        use $crate::op::reshape::Reshape;
        (
            $t,
            std::marker::PhantomData::<TS>,
        )
            .reshape($crate::reshape_val!($($ds)+).into_array())
    }};
}
#[macro_export]
macro_rules! reshape_tys {
    ($e:expr => $d:ty) => {
        glowstick::Shp<(<$d as glowstick::dynamic::Dim>::Id, glowstick::Empty)>
    };
    ($e:expr => $d:ty,$($ds:tt)+) => {
        glowstick::Shp<(<$d as glowstick::dynamic::Dim>::Id, $crate::reshape_tys!($($ds)+))>
    };
    ($d:ty) => {
        glowstick::Shp<($d, glowstick::Empty)>
    };
    ($d:ty,$($ds:tt)+) => {
        glowstick::Shp<($d, $crate::reshape_tys!($($ds)+))>
    };
}
#[macro_export]
macro_rules! reshape_val {
    ($e:expr => $d:ty) => {
        glowstick::ValueList(($e, glowstick::ValueList(())))
    };
    ($d:ty) => {
        glowstick::ValueList((<$d as glowstick::num::Unsigned>::I32,glowstick::ValueList(())))
    };
    ($e:expr => $d:ty,$($ds:tt)+) => {
        glowstick::ValueList(($e,$crate::reshape_val!($($ds)+)))
    };
    ($d:ty,$($ds:tt)+) => {
        glowstick::ValueList((<$d as glowstick::num::Unsigned>::I32,$crate::reshape_val!($($ds)+)))
    };
}

pub trait Reshape<Args, const M: usize> {
    type Out;
    fn reshape(self, args: Args) -> Self::Out;
}
impl<B, D, S1, S2, Args, const N: usize, const M: usize> Reshape<Args, M>
    for (Tensor<BTensor<B, N, D>, S1>, PhantomData<TensorShape<S2>>)
where
    Args: ReshapeArgs<M>,
    B: Backend,
    D: TensorKind<B> + BasicOps<B>,
    S1: Shape,
    TensorShape<S2>: Shape,
    S2: ShapeFragment,
    (S1, TensorShape<S2>): reshape::Compatible,
{
    type Out = Tensor<BTensor<B, M, D>, TensorShape<S2>>;
    fn reshape(self, args: Args) -> Self::Out {
        Tensor(self.0.into_inner().reshape(args), PhantomData)
    }
}
