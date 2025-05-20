use typosaurus::{
    bool::And,
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    DecimalDiagnostic, Dimensioned, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, TensorShape,
    cmp::IsGreater,
    diagnostic::{self, Truthy},
    num::{Add, Sub},
};

struct Transpose;
impl diagnostic::Operation for Transpose {}

/// Boolean type operator for `Transpose` compatibility.
///
/// If shape `T` may be transposed at dims `D1` and `D2`,
/// then the `Out` associated type of this trait for
/// `(T, D1, D2) is `True`.
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

/// Type operator for `Transpose` compatible shapes.
///
/// If dimensions `D1` and `D2` of shape `T` may be transposed, then the
/// `Out` associated type of this trait for `(T, D1, D2)` is the
/// resulting shape.
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
        Truthy<Transpose, <TensorShape<T> as ShapeDiagnostic>::Out, <D1 as DecimalDiagnostic>::Out>,
    (TensorShape<T>, D1): Dimensioned,
    (TensorShape<T>, D2): Dimensioned,
    (D1, U1): Add,
    (TensorShape<T>, D1): TakeFragment,
    (TensorShape<T>, <(D1, U1) as Add>::Out): SkipFragment,
    (D2, U1): Add,
    (TensorShape<T>, <(D2, U1) as Add>::Out): SkipFragment,
    (D2, <(D1, U1) as Add>::Out): Sub,
    <(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out: ShapeFragment,
    (
        TensorShape<<(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out>,
        <(D2, <(D1, U1) as Add>::Out) as Sub>::Out,
    ): TakeFragment,
    (
        <(TensorShape<T>, D1) as TakeFragment>::Out,
        List<(<(TensorShape<T>, D2) as Dimensioned>::Out, Empty)>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, D1) as TakeFragment>::Out,
            List<(<(TensorShape<T>, D2) as Dimensioned>::Out, Empty)>,
        ) as Mappend>::Out,
        <(
            TensorShape<<(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out>,
            <(D2, <(D1, U1) as Add>::Out) as Sub>::Out,
        ) as TakeFragment>::Out,
    ): Mappend,
    (
        <(
            <(
                <(TensorShape<T>, D1) as TakeFragment>::Out,
                List<(<(TensorShape<T>, D2) as Dimensioned>::Out, Empty)>,
            ) as Mappend>::Out,
            <(
                TensorShape<<(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out>,
                <(D2, <(D1, U1) as Add>::Out) as Sub>::Out,
            ) as TakeFragment>::Out,
        ) as Mappend>::Out,
        List<(<(TensorShape<T>, D1) as Dimensioned>::Out, Empty)>,
    ): Mappend,
    (
        <(
            <(
                <(
                    <(TensorShape<T>, D1) as TakeFragment>::Out,
                    List<(<(TensorShape<T>, D2) as Dimensioned>::Out, Empty)>,
                ) as Mappend>::Out,
                <(
                    TensorShape<<(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out>,
                    <(D2, <(D1, U1) as Add>::Out) as Sub>::Out,
                ) as TakeFragment>::Out,
            ) as Mappend>::Out,
            List<(<(TensorShape<T>, D1) as Dimensioned>::Out, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(D2, U1) as Add>::Out) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(
                <(
                    <(TensorShape<T>, D1) as TakeFragment>::Out,
                    List<(<(TensorShape<T>, D2) as Dimensioned>::Out, Empty)>,
                ) as Mappend>::Out,
                <(
                    TensorShape<<(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out>,
                    <(D2, <(D1, U1) as Add>::Out) as Sub>::Out,
                ) as TakeFragment>::Out,
            ) as Mappend>::Out,
            List<(<(TensorShape<T>, D1) as Dimensioned>::Out, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(D2, U1) as Add>::Out) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(
                    <(
                        <(TensorShape<T>, D1) as TakeFragment>::Out,
                        List<(<(TensorShape<T>, D2) as Dimensioned>::Out, Empty)>,
                    ) as Mappend>::Out,
                    <(
                        TensorShape<
                            <(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out,
                        >,
                        <(D2, <(D1, U1) as Add>::Out) as Sub>::Out,
                    ) as TakeFragment>::Out,
                ) as Mappend>::Out,
                List<(<(TensorShape<T>, D1) as Dimensioned>::Out, Empty)>,
            ) as Mappend>::Out,
            <(TensorShape<T>, <(D2, U1) as Add>::Out) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::bool::True;
    use typosaurus::{
        assert_type_eq,
        num::consts::{U0, U1, U2, U3},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn valid() {
        type MyShape = shape![U3, U1, U2];
        assert_type_eq!(<(MyShape, U0, U2) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0, U2) as Compatible>::Out, shape![U2, U1, U3]);
        assert_type_eq!(<(MyShape, U0, U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0, U1) as Compatible>::Out, shape![U1, U3, U2]);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U2, B];

        assert_type_eq!(<(MyShape, U0, U1) as Compatible>::Out, shape![U2, U1, B]);
        assert_type_eq!(<(MyShape, U0, U2) as Compatible>::Out, shape![B, U2, U1]);
        assert_type_eq!(<(MyShape, U1, U2) as Compatible>::Out, shape![U1, B, U2]);
    }
}
