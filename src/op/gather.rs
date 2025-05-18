use core::ops::Add;

use typosaurus::{bool::And, num::consts::U1, traits::semigroup::Mappend};

use crate::{
    DecimalDiagnostic, Dimension, Dimensioned, IDX, MaxDim, Shape, ShapeDiagnostic, ShapeFragment,
    SkipFragment, TakeFragment, TensorShape,
    cmp::{IsGreater, IsGreaterOrEqual},
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
impl<T, I, U> IsCompatible for (TensorShape<T>, I, TensorShape<U>)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    TensorShape<U>: Shape + ShapeDiagnostic,
    U: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    TensorShape<U>: MaxDim,
    <TensorShape<U> as MaxDim>::Out: Dimension,
    (
        <(TensorShape<T>, I) as Dimensioned>::Out,
        <TensorShape<U> as MaxDim>::Out,
    ): IsGreaterOrEqual,
    (
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(
            <(TensorShape<T>, I) as Dimensioned>::Out,
            <TensorShape<U> as MaxDim>::Out,
        ) as IsGreaterOrEqual>::Out,
    ): And,
{
    type Out = <(
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(
            <(TensorShape<T>, I) as Dimensioned>::Out,
            <TensorShape<U> as MaxDim>::Out,
        ) as IsGreaterOrEqual>::Out,
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
impl<T, I, U> Compatible for (TensorShape<T>, I, TensorShape<U>)
where
    (TensorShape<T>, I, TensorShape<U>): IsCompatible,
    <(TensorShape<T>, I, TensorShape<U>) as IsCompatible>::Out: Truthy<Gather, <TensorShape<T> as ShapeDiagnostic>::Out, IDX<<I as DecimalDiagnostic>::Out>>,
    I: DecimalDiagnostic,
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    U: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    TensorShape<U>: MaxDim,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (<(TensorShape<T>, I) as TakeFragment>::Out, U): Mappend,
    (
        <(<(TensorShape<T>, I) as TakeFragment>::Out, U) as Mappend>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(<(TensorShape<T>, I) as TakeFragment>::Out, U) as Mappend>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(<(TensorShape<T>, I) as TakeFragment>::Out, U) as Mappend>::Out,
            <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
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
        type Another = shape![U6, U6];
        assert_type_eq!(<(MyShape, U1, Another) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U1, Another) as Compatible>::Out,
            shape![U3, U6, U6, U2]
        );
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U42, B, U1];
        type Another = shape![U6, U6];
        assert_type_eq!(<(MyShape, U2, Another) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U2, Another) as Compatible>::Out,
            shape![U1, U42, U6, U6, U1]
        );

        assert_type_eq!(<(MyShape, U1, Another) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U1, Another) as Compatible>::Out,
            shape![U1, U6, U6, B, U1]
        );

        type Invalid = shape![U420, U420];
        assert_type_eq!(<(MyShape, U1, Invalid) as IsCompatible>::Out, False);
    }
}
