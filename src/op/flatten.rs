use typosaurus::{
    bool::And,
    collections::list::{Empty, List},
    num::consts::U1,
    traits::{fold::Foldable, semigroup::Mappend},
};

use crate::{
    cmp::IsGreater,
    diagnostic::{self, Truthy},
    num::{monoid::Multiplication, Add, Sub},
    DecimalDiagnostic, Dimensioned, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, TensorShape,
};

struct Flatten;
impl diagnostic::Operation for Flatten {}

/// Boolean type operator for `Flatten` compatibility.
///
/// If shape `T` may be flattened from dimensions `D1` (inclusive) to `D2` (inclusive),
/// then the `Out` associated type of this trait for `(T, D1, D2) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, D1, D2> IsCompatible for (TensorShape<T>, D1, D2)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, D2): IsGreater,
    (D2, D1): IsGreater,
    (
        <(<T as ShapeFragment>::Rank, D2) as IsGreater>::Out,
        <(D2, D1) as IsGreater>::Out,
    ): And,
{
    type Out = <(
        <(<T as ShapeFragment>::Rank, D2) as IsGreater>::Out,
        <(D2, D1) as IsGreater>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Flatten` compatible arguments.
///
/// If shape `T` may be flattened from dimensions `D1` (inclusive) to `D2` (inclusive),
/// then the `Out` associated type of this trait for `(T, D1, D2) is the resulting
/// shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, D1, D2> Compatible for (TensorShape<T>, D1, D2)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, D2): IsGreater,
    (D2, D1): IsGreater,
    (
        <(<T as ShapeFragment>::Rank, D2) as IsGreater>::Out,
        <(D2, D1) as IsGreater>::Out,
    ): And,
    D1: DecimalDiagnostic,
    <(TensorShape<T>, D1, D2) as IsCompatible>::Out:
        Truthy<Flatten, <TensorShape<T> as ShapeDiagnostic>::Out, <D1 as DecimalDiagnostic>::Out>,
    (TensorShape<T>, D1): Dimensioned,
    (TensorShape<T>, D2): Dimensioned,
    (TensorShape<T>, D1): TakeFragment,
    (TensorShape<T>, D1): SkipFragment,
    (D2, U1): Add,
    (TensorShape<T>, <(D2, U1) as Add>::Out): SkipFragment,
    (<(D2, U1) as Add>::Out, D1): Sub,
    <(TensorShape<T>, D1) as SkipFragment>::Out: ShapeFragment,
    (
        TensorShape<<(TensorShape<T>, D1) as SkipFragment>::Out>,
        <(<(D2, U1) as Add>::Out, D1) as Sub>::Out,
    ): TakeFragment,
    <(
        TensorShape<<(TensorShape<T>, D1) as SkipFragment>::Out>,
        <(<(D2, U1) as Add>::Out, D1) as Sub>::Out,
    ) as TakeFragment>::Out: Foldable<Multiplication>,
    (
        <(TensorShape<T>, D1) as TakeFragment>::Out,
        List<(
            <<(
                TensorShape<<(TensorShape<T>, D1) as SkipFragment>::Out>,
                <(<(D2, U1) as Add>::Out, D1) as Sub>::Out,
            ) as TakeFragment>::Out as Foldable<Multiplication>>::Out,
            Empty,
        )>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, D1) as TakeFragment>::Out,
            List<(
                <<(
                    TensorShape<<(TensorShape<T>, D1) as SkipFragment>::Out>,
                    <(<(D2, U1) as Add>::Out, D1) as Sub>::Out,
                ) as TakeFragment>::Out as Foldable<Multiplication>>::Out,
                Empty,
            )>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(D2, U1) as Add>::Out) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(TensorShape<T>, D1) as TakeFragment>::Out,
            List<(
                <<(
                    TensorShape<<(TensorShape<T>, D1) as SkipFragment>::Out>,
                    <(<(D2, U1) as Add>::Out, D1) as Sub>::Out,
                ) as TakeFragment>::Out as Foldable<Multiplication>>::Out,
                Empty,
            )>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(D2, U1) as Add>::Out) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(TensorShape<T>, D1) as TakeFragment>::Out,
                List<(
                    <<(
                        TensorShape<<(TensorShape<T>, D1) as SkipFragment>::Out>,
                        <(<(D2, U1) as Add>::Out, D1) as Sub>::Out,
                    ) as TakeFragment>::Out as Foldable<Multiplication>>::Out,
                    Empty,
                )>,
            ) as Mappend>::Out,
            <(TensorShape<T>, <(D2, U1) as Add>::Out) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::True,
        num::consts::{U0, U1, U12, U2, U24, U3, U4, U6},
    };

    use super::*;

    use crate::{shape, Dyn};

    #[allow(unused)]
    #[test]
    fn valid() {
        type MyShape = shape![U3, U4, U2];
        assert_type_eq!(<(MyShape, U0, U2) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0, U2) as Compatible>::Out, shape![U24]);
        assert_type_eq!(<(MyShape, U0, U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0, U1) as Compatible>::Out, shape![U12, U2]);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        struct BatchSize;
        type B = Dyn<BatchSize>;
        type MyShape = shape![U3, U2, B];

        assert_type_eq!(<(MyShape, U0, U1) as Compatible>::Out, shape![U6, B]);
        assert_type_eq!(<(MyShape, U0, U2) as Compatible>::Out, shape![B]);
        assert_type_eq!(<(MyShape, U1, U2) as Compatible>::Out, shape![U3, B]);
    }
}
