use typosaurus::{
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    DecimalDiagnostic, Dimension, Dimensioned, IDX, Shape, ShapeDiagnostic, ShapeFragment,
    SkipFragment, TakeFragment, TensorShape,
    cmp::IsGreater,
    diagnostic::{self, Truthy},
    num::Add,
};

struct Pad;
impl diagnostic::Operation for Pad {}

/// Boolean type operator for `Narrow` compatibility.
///
/// If shape `T` may be narrowed at dim `I` to length `L` starting
/// from element `S`, then the `Out` associated type of this trait for
/// `(T, I, S, L) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, I, A, B> IsCompatible for (TensorShape<T>, I, A, B)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    (I, U1): Add,
    (A, B): Add,
{
    type Out = <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out;
    crate::private_impl!();
}

/// Type operator for `Narrow`-compatible shapes.
///
/// If shape `T` may be narrowed on dim `I` to length `L` starting from element
/// `S`, then the `Out` associated type of this trait for `(T, I, S, L)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, I, A, B> Compatible for (TensorShape<T>, I, A, B)
where
    (TensorShape<T>, I, A, B): IsCompatible,
    <(TensorShape<T>, I, A, B) as IsCompatible>::Out:
        Truthy<Pad, <TensorShape<T> as ShapeDiagnostic>::Out, IDX<<I as DecimalDiagnostic>::Out>>,
    I: DecimalDiagnostic,
    A: Dimension,
    B: Dimension,
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    (I, U1): Add,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <(I, U1) as Add>::Out): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (A, B): Add,
    (
        <(TensorShape<T>, I) as Dimensioned>::Out,
        <(A, B) as Add>::Out,
    ): Add,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        List<(
            <(
                <(TensorShape<T>, I) as Dimensioned>::Out,
                <(A, B) as Add>::Out,
            ) as Add>::Out,
            Empty,
        )>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(
                <(
                    <(TensorShape<T>, I) as Dimensioned>::Out,
                    <(A, B) as Add>::Out,
                ) as Add>::Out,
                Empty,
            )>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(
                <(
                    <(TensorShape<T>, I) as Dimensioned>::Out,
                    <(A, B) as Add>::Out,
                ) as Add>::Out,
                Empty,
            )>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                List<(
                    <(
                        <(TensorShape<T>, I) as Dimensioned>::Out,
                        <(A, B) as Add>::Out,
                    ) as Add>::Out,
                    Empty,
                )>,
            ) as Mappend>::Out,
            <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        num::consts::{U0, U1, U2, U4, U6, U8},
    };

    use super::*;

    use crate::shape;

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U2, U4, U6, U8];
        assert_type_eq!(<(MyShape, U1, U2, U2) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U1, U2, U2) as Compatible>::Out,
            shape![U2, U8, U6, U8]
        );
        assert_type_eq!(<(MyShape, U0, U4, U0) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U0, U4, U0) as Compatible>::Out,
            shape![U6, U4, U6, U8]
        );
        assert_type_eq!(<(MyShape, U2, U0, U2) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U2, U0, U2) as Compatible>::Out,
            shape![U2, U4, U8, U8]
        );

        assert_type_eq!(<(MyShape, U6, U2, U2) as IsCompatible>::Out, False);
        assert_type_eq!(<(MyShape, U8, U0, U0) as IsCompatible>::Out, False);
    }
}
