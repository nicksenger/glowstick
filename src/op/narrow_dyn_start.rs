use core::ops::Add;

use typosaurus::{
    bool::And,
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    cmp::{IsGreater, IsGreaterOrEqual},
    diagnostic::{self, Truthy},
    DecimalDiagnostic, Dimension, Dimensioned, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, TensorShape, IDX,
};

struct Narrow;
impl diagnostic::Operation for Narrow {}

/// Boolean type operator for `Narrow` compatibility.
///
/// If shape `T` may be narrowed at dim `I` to length `L` starting
/// from element 0, then the `Out` associated type of this trait for
/// `(T, I, L) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, I, L> IsCompatible for (TensorShape<T>, I, L)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (<(TensorShape<T>, I) as Dimensioned>::Out, L): IsGreaterOrEqual,
    (
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<(TensorShape<T>, I) as Dimensioned>::Out, L) as IsGreaterOrEqual>::Out,
    ): And,
{
    type Out = <(
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<(TensorShape<T>, I) as Dimensioned>::Out, L) as IsGreaterOrEqual>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Narrow`-compatible shapes.
///
/// If shape `T` may be narrowed on dim `I` to length `L` starting from element
/// 0, then the `Out` associated type of this trait for `(T, I, L)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, I, L> Compatible for (TensorShape<T>, I, L)
where
    (TensorShape<T>, I, L): IsCompatible,
    <(TensorShape<T>, I, L) as IsCompatible>::Out: Truthy<
        Narrow,
        <TensorShape<T> as ShapeDiagnostic>::Out,
        IDX<<I as DecimalDiagnostic>::Out>,
    >,
    I: DecimalDiagnostic,
    L: Dimension,
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (<(TensorShape<T>, I) as Dimensioned>::Out, L): IsGreaterOrEqual,
    (<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>): Mappend,
    (
        <(<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>) as Mappend>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>) as Mappend>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>) as Mappend>::Out,
            <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
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
        assert_type_eq!(<(MyShape, U0, U2) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0, U2) as Compatible>::Out, shape![U2, U1, U2]);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U42, B, U1];
        assert_type_eq!(<(MyShape, U1, U6) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U1, U6) as Compatible>::Out,
            shape![U1, U6, B, U1]
        );

        assert_type_eq!(<(MyShape, U2, U2) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U2, U2) as Compatible>::Out,
            shape![U1, U42, U2, U1]
        );
    }
}
