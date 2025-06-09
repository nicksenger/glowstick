use core::ops::Add;

use typosaurus::{bool::And, num::consts::U1};

use crate::{
    DecimalDiagnostic, Dimensioned, IDX, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, TensorShape,
    cmp::{IsEqual, IsGreater},
    diagnostic::{self, Truthy},
};

struct Gather;
impl diagnostic::Operation for Gather {}

/// Boolean type operator for `Gather` compatibility.
///
/// If shape `T` may be gathered at dim `I` with indices `U`,
/// then the `Out` associated type of this trait for
/// `(T, I, U)` is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U, I> IsCompatible for (TensorShape<T>, TensorShape<U>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    TensorShape<U>: Shape + ShapeDiagnostic,
    U: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank): IsEqual,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank) as IsEqual>::Out,
    ): And,
{
    type Out = <(
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank) as IsEqual>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Gather`-compatible shapes.
///
/// If shape `T` may be gathered on dim `I` using indices from shape `U`,
/// then the `Out` associated type of this trait for `(T, I, U)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, U, I> Compatible for (TensorShape<T>, TensorShape<U>, I)
where
    (TensorShape<T>, TensorShape<U>, I): IsCompatible,
    <(TensorShape<T>, TensorShape<U>, I) as IsCompatible>::Out: Truthy<Gather, <TensorShape<T> as ShapeDiagnostic>::Out, IDX<<I as DecimalDiagnostic>::Out>>,
    I: DecimalDiagnostic,
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    U: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
{
    type Out = TensorShape<U>;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        num::consts::{U1, U2, U3, U6, U42, U420},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U3, U42, U2];
        type Another = shape![U6, U6, U2];
        assert_type_eq!(<(MyShape, Another, U1) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, Another, U1) as Compatible>::Out,
            shape![U6, U6, U2]
        );
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U2, U42, B, U2];
        type Another = shape![U2, U42, B, U1];
        assert_type_eq!(<(MyShape, Another, U3) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, Another, U2) as Compatible>::Out,
            shape![U2, U42, B, U1]
        );

        type Invalid = shape![U420, U420];
        assert_type_eq!(<(MyShape, Invalid, U1) as IsCompatible>::Out, False);
    }
}
