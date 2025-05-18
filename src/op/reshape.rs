use typosaurus::{collections::Container, traits::fold::Foldable};

use crate::{
    Product, Shape, ShapeDiagnostic, ShapeFragment, TensorShape,
    cmp::IsEqual,
    diagnostic::{self, Truthy},
    num::monoid::Multiplication,
};

struct Reshape;
impl diagnostic::Operation for Reshape {}

/// Boolean type operator for `Reshape` compatibility.
///
/// If shape `T` may be reshaped to shape `U`, then the `Out`
/// associated type of this trait for `(T, U) is `True`.
/// Otherwise, it is `False`.
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
    T: Foldable<Multiplication>,
    U: Foldable<Multiplication>,
    (Product<T>, Product<U>): IsEqual,
{
    type Out = <(Product<T>, Product<U>) as IsEqual>::Out;
    crate::private_impl!();
}

/// Type operator for `Reshape`-compatible shapes.
///
/// If shape `T` may be reshaped to shape `U`, then the
/// `Out` associated type of this trait for `(T, U)` is the
/// resulting shape.
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
    T: Foldable<Multiplication>,
    U: Foldable<Multiplication>,
    (Product<T>, Product<U>): IsEqual,
    (TensorShape<T>, TensorShape<U>): IsCompatible,
    <(TensorShape<T>, TensorShape<U>) as IsCompatible>::Out:
        Truthy<Reshape, TensorShape<T>, TensorShape<U>>,
{
    type Out = TensorShape<U>;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        num::consts::{U1, U2, U3, U6, U7},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn valid() {
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, shape![U1, U6]) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, shape![U1, U6]) as Compatible>::Out,
            shape![U1, U6]
        );

        assert_type_eq!(<(MyShape, shape![U2, U3]) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, shape![U2, U3]) as Compatible>::Out,
            shape![U2, U3]
        );

        assert_type_eq!(<(MyShape, shape![U2, U3, U1]) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, shape![U2, U3, U1]) as Compatible>::Out,
            shape![U2, U3, U1]
        );
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
        type B = Dyn<Any>;
        type MyShape = shape![U1, U1, B, U1];
        assert_type_eq!(<(MyShape, shape![U1, U7]) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, shape![U1, U2, U3, U2, U7]) as IsCompatible>::Out,
            True
        );

        // TODO: this can be forbidden despite the wildness
        assert_type_eq!(<(MyShape, shape![U1]) as IsCompatible>::Out, True);
    }
}
