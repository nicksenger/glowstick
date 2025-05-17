use typosaurus::{
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    cmp::IsGreaterOrEqual,
    diagnostic::{self, Truthy},
    DecimalDiagnostic, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment, TakeFragment,
    TensorShape,
};

struct Unsqueeze;
impl diagnostic::Operation for Unsqueeze {}

/// Boolean type operator for `Unsqueeze` compatibility.
///
/// If shape `T` may be unsqueezed at dim `I`,
/// then the `Out` associated type of this trait for
/// `(T, I) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, I> IsCompatible for (TensorShape<T>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreaterOrEqual,
{
    type Out = <(<T as ShapeFragment>::Rank, I) as IsGreaterOrEqual>::Out;
    crate::private_impl!();
}

/// Type operator for `Unsqueeze`-compatible shapes.
///
/// If shape `T` may be unsqueezed at dim `I`, then the
/// `Out` associated type of this trait for `(T, I)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, I> Compatible for (TensorShape<T>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreaterOrEqual,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, I): SkipFragment,
    (TensorShape<T>, I): IsCompatible,
    I: DecimalDiagnostic,
    <(TensorShape<T>, I) as IsCompatible>::Out: Truthy<
        Unsqueeze,
        <TensorShape<T> as crate::ShapeDiagnostic>::Out,
        crate::IDX<<I as crate::DecimalDiagnostic>::Out>,
    >,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        List<(U1, Empty)>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(U1, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, I) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(U1, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, I) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                List<(U1, Empty)>,
            ) as Mappend>::Out,
            <(TensorShape<T>, I) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        num::consts::{U0, U1, U2, U3, U4, U6, U7, U8},
    };

    use super::*;

    use crate::{dynamic::Any, shape, Dyn};

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U3, U1, U2];
        assert_type_eq!(<(MyShape, U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U1) as Compatible>::Out, shape![U3, U1, U1, U2]);

        type Another = shape![U1, U2, U3, U2, U3, U2, U1];
        assert_type_eq!(<(Another, U0) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(Another, U0) as Compatible>::Out,
            shape![U1, U1, U2, U3, U2, U3, U2, U1]
        );
        assert_type_eq!(<(Another, U4) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(Another, U4) as Compatible>::Out,
            shape![U1, U2, U3, U2, U1, U3, U2, U1]
        );
        assert_type_eq!(<(Another, U6) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(Another, U6) as Compatible>::Out,
            shape![U1, U2, U3, U2, U3, U2, U1, U1]
        );
        assert_type_eq!(<(Another, U7) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(Another, U6) as Compatible>::Out,
            shape![U1, U2, U3, U2, U3, U2, U1, U1]
        );
        assert_type_eq!(<(Another, U8) as IsCompatible>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U1, B];
        assert_type_eq!(<(MyShape, U0) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0) as Compatible>::Out, shape![U1, U1, U1, B]);

        assert_type_eq!(<(MyShape, U3) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U3) as Compatible>::Out, shape![U1, U1, B, U1]);
    }
}
