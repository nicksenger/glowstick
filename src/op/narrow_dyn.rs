use core::ops::Add;

use typosaurus::{
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    DecimalDiagnostic, Dimensioned, IDX, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, TensorShape,
    cmp::IsGreater,
    diagnostic::{self, Truthy},
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
impl<T, I, DynDim> IsCompatible for (TensorShape<T>, I, DynDim)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
{
    type Out = <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out;
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
impl<T, I, DynDim> Compatible for (TensorShape<T>, I, DynDim)
where
    (TensorShape<T>, I, DynDim): IsCompatible,
    <(TensorShape<T>, I, DynDim) as IsCompatible>::Out: Truthy<Narrow, <TensorShape<T> as ShapeDiagnostic>::Out, IDX<<I as DecimalDiagnostic>::Out>>,
    I: DecimalDiagnostic,
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        List<(DynDim, Empty)>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(DynDim, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(DynDim, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                List<(DynDim, Empty)>,
            ) as Mappend>::Out,
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
        num::consts::{U0, U1, U2, U3},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U3, U1, U2];
        type D = Dyn<Any>;
        assert_type_eq!(<(MyShape, U0, Dyn<Any>) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U0, Dyn<Any>) as Compatible>::Out,
            shape![D, U1, U2]
        );
    }
}
