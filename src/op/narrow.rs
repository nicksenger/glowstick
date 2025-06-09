use typosaurus::{
    bool::And,
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    cmp::{IsGreater, IsGreaterOrEqual},
    diagnostic::{self, Truthy},
    num::Add,
    DecimalDiagnostic, Dimension, Dimensioned, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, Tensor, TensorShape, IDX,
};

pub struct Narrow;
impl diagnostic::Operation for Narrow {}

pub fn check<T, TS, I, S, L>(_t: &T)
where
    T: Tensor<Shape = TS>,
    TS: ShapeDiagnostic,
    I: DecimalDiagnostic,
    (TS, I, S, L): IsCompatible,
    <(TS, I, S, L) as IsCompatible>::Out:
        Truthy<Narrow, <TS as ShapeDiagnostic>::Out, <I as DecimalDiagnostic>::Out>,
{
}

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
impl<T, I, L, S> IsCompatible for (TensorShape<T>, I, S, L)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    (I, U1): Add,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <(I, U1) as Add>::Out): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (S, L): Add,
    (
        <(TensorShape<T>, I) as Dimensioned>::Out,
        <(S, L) as Add>::Out,
    ): IsGreaterOrEqual,
    (
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(
            <(TensorShape<T>, I) as Dimensioned>::Out,
            <(S, L) as Add>::Out,
        ) as IsGreaterOrEqual>::Out,
    ): And,
{
    type Out = <(
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(
            <(TensorShape<T>, I) as Dimensioned>::Out,
            <(S, L) as Add>::Out,
        ) as IsGreaterOrEqual>::Out,
    ) as And>::Out;
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
impl<T, I, S, L> Compatible for (TensorShape<T>, I, S, L)
where
    (TensorShape<T>, I, S, L): IsCompatible,
    <(TensorShape<T>, I, S, L) as IsCompatible>::Out: Truthy<
        Narrow,
        <TensorShape<T> as ShapeDiagnostic>::Out,
        IDX<<I as DecimalDiagnostic>::Out>,
    >,
    I: DecimalDiagnostic,
    L: Dimension,
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    (I, U1): Add,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <(I, U1) as Add>::Out): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (S, L): Add,
    (
        <(TensorShape<T>, I) as Dimensioned>::Out,
        <(S, L) as Add>::Out,
    ): IsGreaterOrEqual,
    (<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>): Mappend,
    (
        <(<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>) as Mappend>::Out,
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
        num::consts::{U0, U1, U2, U3, U42, U6},
    };

    use super::*;

    use crate::{dynamic::Any, shape, Dyn};

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U3, U1, U2];
        assert_type_eq!(<(MyShape, U0, U1, U2) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U0, U1, U2) as Compatible>::Out,
            shape![U2, U1, U2]
        );
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U42, B, U1];
        assert_type_eq!(<(MyShape, U1, U6, U6) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U1, U6, U6) as Compatible>::Out,
            shape![U1, U6, B, U1]
        );

        assert_type_eq!(<(MyShape, U2, U6, U2) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U2, U6, U2) as Compatible>::Out,
            shape![U1, U42, U2, U1]
        );
    }
}
