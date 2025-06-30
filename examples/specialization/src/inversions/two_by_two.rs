use glowstick::{num::U2, Shape2};

use crate::{
    t,
    tensor::{Arr, TVec, Tensor},
};

pub trait Invert2x2Matrix {
    type Output;
    fn invert_2x2_matrix(&self) -> Self::Output;
}

impl<T> Invert2x2Matrix for T
where
    T: Tensor<Shape = Shape2<U2, U2>> + Arr,
    <T as Arr>::Content: Arr<Content = f32>,
{
    type Output = TVec<TVec<f32, U2>, U2>;

    fn invert_2x2_matrix(&self) -> Self::Output {
        let a = *self.get(0).get(0);
        let b = *self.get(0).get(1);
        let c = *self.get(1).get(0);
        let d = *self.get(1).get(1);

        let determinant = a * d - b * c;

        if determinant == 0.0 {
            panic!("inversion failed!");
        }

        t![
            t![d / determinant, -b / determinant],
            t![-c / determinant, a / determinant]
        ]
    }
}
