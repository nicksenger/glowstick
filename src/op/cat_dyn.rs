use typosaurus::{
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    Dimensioned, Dyn, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment, TakeFragment,
    TensorShape,
    cmp::IsGreater,
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
impl<T, I> IsCompatible for (TensorShape<T>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
{
    type Out = <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out;
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
impl<T, I, D> Compatible for (TensorShape<T>, I, Dyn<D>)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    (I, U1): Add,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <(I, U1) as Add>::Out): SkipFragment,
    <(TensorShape<T>, I) as IsCompatible>::Out: Truthy<Cat, TensorShape<T>, I>,
    (TensorShape<T>, I): Dimensioned,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        List<(Dyn<D>, Empty)>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(Dyn<D>, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(Dyn<D>, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                List<(Dyn<D>, Empty)>,
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
        num::consts::{U0, U1, U2, U3},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn valid() {
        type N = Dyn<Any>;
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, U0) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0, N) as Compatible>::Out, shape![N, U2]);
        assert_type_eq!(<(MyShape, U1, N) as Compatible>::Out, shape![U3, N]);
    }
}
