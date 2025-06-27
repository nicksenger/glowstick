use glowstick::cmp::IsEqual;
use glowstick::num::{U2, U4};
use glowstick::{Shape, Shape2};

mod tensor;

pub use tensor::{Arr, Tensor};

pub trait Invert4x4Matrix {
    fn invert(&self) -> Self;
}

impl<T> Invert4x4Matrix for &T
where
    T: Tensor<Shape = Shape2<U4, U4>>,
{
    fn invert(&self) -> Self {
        todo!("special 4x4 matrix inversion!")
    }
}

pub trait InvertMatrix {
    fn invert(&self) -> Self;
}

impl<T> InvertMatrix for &&T
where
    T: Tensor,
    (<<T as Tensor>::Shape as Shape>::Rank, U2): IsEqual,
{
    fn invert(&self) -> Self {
        todo!("standard matrix inversion",)
    }
}

#[allow(unused)]
macro_rules! invert {
    ($t:expr) => {{
        #[allow(unused)]
        use $crate::Invert4x4Matrix;
        #[allow(unused)]
        use $crate::InvertMatrix;
        (&&$t).invert()
    }};
}

#[cfg(test)]
mod test {
    #[test]
    #[should_panic(expected = "special 4x4 matrix inversion!")]
    fn test_invert_special() {
        let my_matrix = [
            [1f32, 2., 3., 4.],
            [1., 2., 3., 4.],
            [1., 2., 3., 4.],
            [1., 2., 3., 4.],
        ];
        invert!(my_matrix);
    }

    #[test]
    #[should_panic(expected = "standard matrix inversion")]
    fn test_invert_standard() {
        let my_matrix = [[1f32, 2.], [3., 4.]];
        invert!(my_matrix);
    }
}
