use typosaurus::collections::Container;

use crate::{
    diagnostic::{self, Truthy},
    IsFragEqual, Shape, ShapeDiagnostic, ShapeFragment, TensorShape,
};

struct Transmute;
impl diagnostic::Operation for Transmute {}

/// Boolean type operator for `Transmute` compatibility.
///
/// If shape `T` may be transmuted to shape `U`,
/// then the `Out` associated type of this trait for
/// `(T, U) is `True`.
/// Otherwise, it is `False`.
///
/// Transmutation converts shape `T` to shape `U`, but requires
/// that they are equal. This is intended for converting to and
/// from `Dyn<_>` dimensions.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U> IsCompatible for (TensorShape<T>, TensorShape<U>)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment + Container,
    U: ShapeFragment + Container,
    (T, U): IsFragEqual,
{
    type Out = <(T, U) as IsFragEqual>::Out;
    crate::private_impl!();
}

/// Type operator for `Transmute` compatible shapes.
///
/// If shapes `T` may be transmuted to shape `U`, then the
/// `Out` associated type of this trait for `(T, U)` is the
/// resulting shape.
///
/// Transmutation converts shape `T` to shape `U`, and requires
/// that they are equal. This is intended for converting to and
/// from `Dyn<_>` dimensions.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, U> Compatible for (TensorShape<T>, TensorShape<U>)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment + Container,
    U: ShapeFragment + Container,
    (TensorShape<T>, TensorShape<U>): IsCompatible,
    <(TensorShape<T>, TensorShape<U>) as IsCompatible>::Out:
        Truthy<Transmute, TensorShape<T>, TensorShape<U>>,
{
    type Out = TensorShape<U>;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        num::consts::{U1, U2, U3, U7},
    };

    use super::*;

    use crate::{shape, Dyn};

    #[allow(unused)]
    #[test]
    fn valid() {
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, MyShape) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, MyShape) as Compatible>::Out, MyShape);
    }

    #[allow(unused)]
    #[test]
    fn invalid() {
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, shape![U1, U7]) as IsCompatible>::Out, False);
        assert_type_eq!(<(MyShape, shape![U3, U3]) as IsCompatible>::Out, False);
        assert_type_eq!(<(MyShape, shape![U2, U3, U2]) as IsCompatible>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        struct BatchSize;
        type B = Dyn<BatchSize>;
        struct SequenceLength;
        type L = Dyn<SequenceLength>;
        type MyShape = shape![U1, U1, B, U1];
        assert_type_eq!(
            <(shape![U1, U1, U7, U1], MyShape) as IsCompatible>::Out,
            True
        );
        assert_type_eq!(
            <(MyShape, shape![U1, U1, L, U1]) as IsCompatible>::Out,
            True
        );
    }
}
