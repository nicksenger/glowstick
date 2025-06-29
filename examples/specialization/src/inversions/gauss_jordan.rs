use glowstick::num::{U0, U1};
use glowstick::{Dimension, Dimensioned, Shape};

use crate::tensor::{Arr, TVec, Tensor};

pub trait InvertGaussJordan {
    type Output;
    fn invert_gauss_jordan(&self) -> Self::Output;
}

impl<T> InvertGaussJordan for T
where
    T: Tensor + Arr + Clone,
    <T as Arr>::Content: Arr<Content = f32>,
    (<T as Tensor>::Shape, U0): Dimensioned,
    (<T as Tensor>::Shape, U1): Dimensioned,
{
    type Output = TVec<
        TVec<f32, <<T as Tensor>::Shape as Shape>::Dim<U0>>,
        <<T as Tensor>::Shape as Shape>::Dim<U0>,
    >;

    fn invert_gauss_jordan(&self) -> Self::Output {
        let n = <<<T as Tensor>::Shape as Shape>::Dim<U0> as Dimension>::USIZE;
        let mut a = self.clone();
        let mut b = identity::<<<T as Tensor>::Shape as Shape>::Dim<U0>>();

        for i in 0..n {
            let mut max_row = i;
            for j in (i + 1)..n {
                if a.get(j).get(i).abs() > a.get(max_row).get(i).abs() {
                    max_row = j;
                }
            }

            a.swap(i, max_row);
            b.swap(i, max_row);

            let divisor = *a.get(i).get(i);
            for j in 0..n {
                *a.get_mut(i).get_mut(j) /= divisor;
                *b.get_mut(i).get_mut(j) /= divisor;
            }

            for k in (0..n).filter(|&k| k != i) {
                let factor = *a.get(k).get(i);
                for j in 0..n {
                    *a.get_mut(k).get_mut(j) -= factor * a.get(i).get(j);
                    *b.get_mut(k).get_mut(j) -= factor * b.get(i).get(j);
                }
            }
        }

        b
    }
}

fn identity<N: Dimension>() -> TVec<TVec<f32, N>, N> {
    let n = <N as Dimension>::USIZE;
    let row = TVec::zeros();
    let mut matrix = TVec::with(row);

    for i in 0..n {
        *matrix.get_mut(i).get_mut(i) = 1.;
    }
    matrix
}
