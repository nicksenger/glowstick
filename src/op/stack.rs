use typosaurus::{
    collections::list::{Empty, List},
    num::consts::U2,
    traits::semigroup::Mappend,
};

use crate::{
    diagnostic::{self, Truthy},
    fragment, IsFragEqual, Shape, ShapeDiagnostic, ShapeFragment, TensorShape,
};

struct Stack;
impl diagnostic::Operation for Stack {}

/// Boolean type operator for `Stack` compatibility.
///
/// If shapes `T` and `U` may be stacked,
/// then the `Out` associated type of this trait for
/// `(T, U) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U> IsCompatible for (TensorShape<T>, TensorShape<U>)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    U: ShapeFragment,
    (T, U): IsFragEqual,
{
    type Out = <(T, U) as IsFragEqual>::Out;
    crate::private_impl!();
}

/// Type operator for `Stack`-compatible shapes.
///
/// If shapes `T` and `U` may be stacked, then the
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
    T: ShapeFragment,
    U: ShapeFragment,
    (T, U): IsFragEqual,
    (List<(U2, Empty)>, T): Mappend,
    <(List<(U2, Empty)>, T) as Mappend>::Out: ShapeFragment,
    (TensorShape<T>, TensorShape<U>): IsCompatible,
    <(TensorShape<T>, TensorShape<U>) as IsCompatible>::Out:
        Truthy<Stack, TensorShape<T>, TensorShape<U>>,
{
    type Out = TensorShape<<(fragment![U2], T) as Mappend>::Out>;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::bool::{False, True};
    use typosaurus::{
        assert_type_eq,
        num::consts::{U1, U2, U3, U7},
    };

    use super::*;

    use crate::{dynamic::Any, shape, Dyn};

    #[allow(unused)]
    #[test]
    fn valid() {
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, shape![U3, U2]) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, shape![U3, U2]) as Compatible>::Out,
            shape![U2, U3, U2]
        );
    }

    #[allow(unused)]
    #[test]
    fn invalid() {
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, shape![U1, U7]) as IsCompatible>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U1, B];
        assert_type_eq!(<(MyShape, MyShape) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, MyShape) as Compatible>::Out,
            shape![U2, U1, U1, B]
        );
    }
}
