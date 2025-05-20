use typosaurus::{
    bool::And,
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    Dimensioned, IsFragEqual, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment, TakeFragment,
    TensorShape,
    cmp::{IsEqual, IsGreater},
    diagnostic::{self, Truthy},
    num::Add,
};

struct Cat;
impl diagnostic::Operation for Cat {}

/// Boolean type operator for `Cat` compatibility.
///
/// If shape `U` may be concatenated with shape `T` on dimension `I`, then
/// the `Out` associated type of this trait for `(T, U, I) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U, I> IsCompatible for (TensorShape<T>, TensorShape<U>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    U: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    (<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank): IsEqual,
    (I, U1): Add,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<U>, I): TakeFragment,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        <(TensorShape<U>, I) as TakeFragment>::Out,
    ): IsFragEqual,
    (TensorShape<T>, <(I, U1) as Add>::Out): SkipFragment,
    (TensorShape<U>, <(I, U1) as Add>::Out): SkipFragment,
    (
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
        <(TensorShape<U>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ): IsFragEqual,
    (
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank) as IsEqual>::Out,
    ): And,
    (
        <(
            <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
            <(<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank) as IsEqual>::Out,
        ) as And>::Out,
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            <(TensorShape<U>, I) as TakeFragment>::Out,
        ) as IsFragEqual>::Out,
    ): And,
    (
        <(
            <(
                <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
                <(<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank) as IsEqual>::Out,
            ) as And>::Out,
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                <(TensorShape<U>, I) as TakeFragment>::Out,
            ) as IsFragEqual>::Out,
        ) as And>::Out,
        <(
            <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
            <(TensorShape<U>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
        ) as IsFragEqual>::Out,
    ): And,
{
    type Out = <(
        <(
            <(
                <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
                <(<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank) as IsEqual>::Out,
            ) as And>::Out,
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                <(TensorShape<U>, I) as TakeFragment>::Out,
            ) as IsFragEqual>::Out,
        ) as And>::Out,
        <(
            <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
            <(TensorShape<U>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
        ) as IsFragEqual>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for concat-compatible shapes.
///
/// If shape `U` may be concatenated with shape `T` at dimension `I`,
/// then the `Out` assocatied type of this trait for `(T, U, I)` is
/// the resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, U, I> Compatible for (TensorShape<T>, TensorShape<U>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    U: ShapeFragment,
    (I, U1): Add,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <(I, U1) as Add>::Out): SkipFragment,
    (TensorShape<T>, TensorShape<U>, I): IsCompatible,
    <(TensorShape<T>, TensorShape<U>, I) as IsCompatible>::Out:
        Truthy<Cat, TensorShape<T>, TensorShape<U>>,
    (TensorShape<T>, I): Dimensioned,
    (TensorShape<U>, I): Dimensioned,
    (
        <(TensorShape<T>, I) as Dimensioned>::Out,
        <(TensorShape<U>, I) as Dimensioned>::Out,
    ): Add,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        List<(
            <(
                <(TensorShape<T>, I) as Dimensioned>::Out,
                <(TensorShape<U>, I) as Dimensioned>::Out,
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
                    <(TensorShape<U>, I) as Dimensioned>::Out,
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
                    <(TensorShape<U>, I) as Dimensioned>::Out,
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
                        <(TensorShape<U>, I) as Dimensioned>::Out,
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
        bool::True,
        num::consts::{U0, U1, U2, U3, U4, U6, U42, U44},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn valid() {
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, shape![U3, U2], U0) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, shape![U3, U2], U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, shape![U3, U42], U1) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, shape![U3, U2], U0) as Compatible>::Out,
            shape![U6, U2]
        );
        assert_type_eq!(
            <(MyShape, shape![U3, U2], U1) as Compatible>::Out,
            shape![U3, U4]
        );
        assert_type_eq!(
            <(MyShape, shape![U3, U42], U1) as Compatible>::Out,
            shape![U3, U44]
        );
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type ShapeA = shape![U1, U1, B];
        type ShapeB = shape![U1, U2, B];

        type OutShape = <(ShapeA, ShapeB, U1) as Compatible>::Out;
        assert_type_eq!(OutShape, shape![U1, U3, B]);
    }
}
