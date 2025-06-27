use glowstick::{num::*, Empty, ShapeFragment};
use glowstick::{Dimension, Mappend, Shape, Shape1, Shp, TensorShape};

pub trait Content {}
impl Content for f32 {}
impl<T: Arr> Content for T
where
    T: Arr,
    <T as Arr>::Content: Content,
{
}

pub trait Arr {
    type Content: Content;
    type Size: Dimension;

    fn get(&self, idx: usize) -> &Self::Content;
}

macro_rules! arr_impl {
    [$t:ty:$n:expr] => {
        impl<T: Content> Arr for [T; <$t as Dimension>::USIZE] {
            type Content = T;
            type Size = $t;

            fn get(&self, idx: usize) -> &Self::Content {
                &self[idx]
            }
        }
    };
    [$t:ty:$n:expr,$($ts:ty:$ns:expr),+] => {
        arr_impl![$t:$n];
        arr_impl![$($ts:$ns),+];
    };
}
arr_impl![U1: 1, U2: 2, U3: 3, U4: 4, U5: 5, U6: 6, U7: 7, U8: 8, U9: 9, U10: 10, U11: 11, U12: 12];

pub trait Tensor: Arr {
    type Shape: Shape;
}

impl<const N: usize> Tensor for [f32; N]
where
    Self: Arr,
    <Self as Arr>::Size: NonZero,
{
    type Shape = Shape1<<Self as Arr>::Size>;
}

impl<T, const N: usize> Tensor for [T; N]
where
    T: Tensor,
    Self: Arr,
    <Self as Arr>::Size: NonZero,
    (
        Shp<(<Self as Arr>::Size, Empty)>,
        <<T as Tensor>::Shape as Shape>::Fragment,
    ): Mappend,
    <(
        Shp<(<Self as Arr>::Size, Empty)>,
        <<T as Tensor>::Shape as Shape>::Fragment,
    ) as Mappend>::Out: ShapeFragment,
{
    type Shape = TensorShape<
        <(
            Shp<(<Self as Arr>::Size, Empty)>,
            <<T as Tensor>::Shape as Shape>::Fragment,
        ) as Mappend>::Out,
    >;
}
