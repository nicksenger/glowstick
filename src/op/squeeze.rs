use core::ops::Add;

use typosaurus::num::consts::U1;
use typosaurus::{bool::And, traits::semigroup::Mappend};

use crate::{
    DecimalDiagnostic, Dimensioned, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, TensorShape,
    cmp::{IsEqual, IsGreater},
    diagnostic::{self, Truthy},
};

struct Squeeze;
impl diagnostic::Operation for Squeeze {}

/// Boolean type operator for `Squeeze` compatibility.
///
/// If shape `T` may be squeezed at dim `I`,
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
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (<(TensorShape<T>, I) as Dimensioned>::Out, U1): IsEqual,
    (
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<(TensorShape<T>, I) as Dimensioned>::Out, U1) as IsEqual>::Out,
    ): And,
{
    type Out = <(
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<(TensorShape<T>, I) as Dimensioned>::Out, U1) as IsEqual>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Squeeze`-compatible shapes.
///
/// If shape `T` may be squeezed at dim `I`, then the
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
    I: Add<U1> + DecimalDiagnostic,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): IsCompatible,
    <(TensorShape<T>, I) as IsCompatible>::Out: Truthy<
            Squeeze,
            <TensorShape<T> as crate::ShapeDiagnostic>::Out,
            crate::IDX<<I as crate::DecimalDiagnostic>::Out>,
        >,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(TensorShape<T>, I) as TakeFragment>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
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
        num::consts::{U0, U1, U2, U3, U4, U5, U6, U64, U151, U936, U1000},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U3, U1, U2];
        assert_type_eq!(<(MyShape, U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U1) as Compatible>::Out, shape![U3, U2]);

        type Another = shape![U1, U2, U3, U2, U3, U2, U1];
        assert_type_eq!(<(Another, U0) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(Another, U0) as Compatible>::Out,
            shape![U2, U3, U2, U3, U2, U1]
        );
        assert_type_eq!(<(Another, U6) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(Another, U6) as Compatible>::Out,
            shape![U1, U2, U3, U2, U3, U2]
        );
        assert_type_eq!(<(Another, U1) as IsCompatible>::Out, False);
        assert_type_eq!(<(Another, U2) as IsCompatible>::Out, False);
        assert_type_eq!(<(Another, U3) as IsCompatible>::Out, False);
        assert_type_eq!(<(Another, U4) as IsCompatible>::Out, False);
        assert_type_eq!(<(Another, U5) as IsCompatible>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn logits() {
        type U151936 = <<U151 as core::ops::Mul<U1000>>::Output as Add<U936>>::Output;
        pub type Logits = shape![U1, U1, U151936];
        assert_type_eq!(<(Logits, U0) as IsCompatible>::Out, True);
        assert_type_eq!(<(Logits, U0) as Compatible>::Out, shape![U1, U151936]);

        type Test = shape![U1, U64, U1];
        assert_type_eq!(<(Test, U0) as IsCompatible>::Out, True);
        assert_type_eq!(<(Test, U0) as Compatible>::Out, shape![U64, U1]);
        assert_type_eq!(<(Test, U2) as IsCompatible>::Out, True);
        assert_type_eq!(<(Test, U2) as Compatible>::Out, shape![U1, U64]);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U1, B, U1];
        assert_type_eq!(<(MyShape, U0) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0) as Compatible>::Out, shape![U1, B, U1]);
        assert_type_eq!(<(MyShape, U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U1) as Compatible>::Out, shape![U1, B, U1]);
        assert_type_eq!(<(MyShape, U3) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U3) as Compatible>::Out, shape![U1, U1, B]);
    }
}
