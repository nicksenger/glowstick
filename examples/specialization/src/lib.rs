use glowstick::cmp::IsEqual;
use glowstick::num::U2;
use glowstick::{Shape, Shape1, Shape2};

mod inversions;
mod tensor;

use inversions::{gauss_jordan::InvertGaussJordan, two_by_two::Invert2x2Matrix};
use tensor::{Arr, Tensor};

pub trait InvertMatrix {
    type Output;
    fn invert(&self) -> Self::Output;
}

impl<T> InvertMatrix for &&T
where
    T: Tensor + InvertGaussJordan<Output = T>,
    (<<T as Tensor>::Shape as Shape>::Rank, U2): IsEqual,
{
    type Output = T;
    fn invert(&self) -> Self::Output {
        println!("just using Gauss-Jordan");
        self.invert_gauss_jordan()
    }
}

pub trait Invert2x2 {
    type Output;
    fn invert(&self) -> Self::Output;
}

impl<T> Invert2x2 for &T
where
    T: Tensor<Shape = Shape2<U2, U2>> + Invert2x2Matrix<Output = T>,
    <T as Arr>::Content: Tensor<Shape = Shape1<U2>>,
{
    type Output = T;
    fn invert(&self) -> Self::Output {
        println!("using special 2x2 inversion!");
        self.invert_2x2_matrix()
    }
}

#[allow(unused)]
macro_rules! invert {
    ($t:expr) => {{
        #[allow(unused)]
        use $crate::Invert2x2;
        #[allow(unused)]
        use $crate::InvertMatrix;
        (&&$t).invert()
    }};
}

#[cfg(test)]
mod test {
    use crate::{t, Tensor};

    #[test]
    fn test_invert_standard() {
        let my_matrix = t![t![1f32, 0., 5.], t![2f32, 1., 6.], t![3f32, 4., 0.]];
        let inverted = invert!(my_matrix);
        assert_eq!(
            round(inverted, 3),
            t![t![-24f32, 20., -5.], t![18f32, -15., 4.], t![5f32, -4., 1.]]
        );
    }

    #[test]
    fn test_invert_2x2() {
        let my_matrix = t![t![4f32, 3.], t![1., 1.]];
        let inverted = invert!(my_matrix);
        assert_eq!(inverted, t![t![1f32, -3.], t![-1., 4.]]);
    }

    fn round<T: Tensor>(mut tensor: T, digits: i32) -> T {
        let b = 10f32.powi(digits);
        tensor.iter_mut().for_each(|t| {
            *t = f32::round(*t * b) / b;
        });

        tensor
    }
}
