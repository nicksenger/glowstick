use typosaurus::{
    bool::{And, monoid::Both},
    collections::{
        Container,
        list::{IsUnique, Len, Ones},
    },
    num::Addition,
    traits::{fold::Foldable, functor::Map},
};

use crate::{
    AllLessThan, IsLessThan, PermutationOf, Shape, ShapeDiagnostic, ShapeFragment,
    cmp::IsEqual,
    diagnostic::{self, Truthy},
};

/// Boolean type operator for `Permute` compatibility.
///
/// If shape `T` may be permuted using dimension indices represented by `U`,
/// then the `Out` associated type of this trait for `(T, U)` is `True`.
/// Otherwise, it is `False`.
struct Permute;
impl diagnostic::Operation for Permute {}

pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U> IsCompatible for (T, U)
where
    (U, Ones): Map<<U as Container>::Content, Ones>,
    <(U, Ones) as Map<<U as Container>::Content, Ones>>::Out: Foldable<Addition>,
    T: Shape + ShapeDiagnostic,
    (<T as Shape>::Rank, Len<U>): IsEqual,
    U: IsUnique,
    U: Container,
    (U, IsLessThan<<T as Shape>::Rank>):
        Map<<U as Container>::Content, IsLessThan<<T as Shape>::Rank>>,
    <(U, IsLessThan<<T as Shape>::Rank>) as Map<
        <U as Container>::Content,
        IsLessThan<<T as Shape>::Rank>,
    >>::Out: Foldable<Both>,
    (
        <(<T as Shape>::Rank, Len<U>) as IsEqual>::Out,
        <U as IsUnique>::Out,
    ): And,
    (
        <(
            <(<T as Shape>::Rank, Len<U>) as IsEqual>::Out,
            <U as IsUnique>::Out,
        ) as And>::Out,
        AllLessThan<U, <T as Shape>::Rank>,
    ): And,
{
    type Out = <(
        <(
            <(<T as Shape>::Rank, Len<U>) as IsEqual>::Out,
            <U as IsUnique>::Out,
        ) as And>::Out,
        AllLessThan<U, <T as Shape>::Rank>,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Permute`-compatible shapes.
///
/// If shape `T` may be permuted by dimension indices `U`, then the
/// `Out` associated type of this trait for `(T, U)` is the
/// resulting shape.
pub trait Compatible {
    type Out: ShapeFragment;
    crate::private!();
}
impl<T, U> Compatible for (T, U)
where
    (U, Ones): Map<<U as Container>::Content, Ones>,
    <(U, Ones) as Map<<U as Container>::Content, Ones>>::Out: Foldable<Addition>,
    T: Shape + ShapeDiagnostic,
    (<T as Shape>::Rank, Len<U>): IsEqual,
    <(<T as Shape>::Rank, Len<U>) as IsEqual>::Out:
        Truthy<Permute, <T as crate::ShapeDiagnostic>::Out, ()>,
    U: IsUnique,
    <U as IsUnique>::Out: Truthy<Permute, <T as crate::ShapeDiagnostic>::Out, ()>,
    U: Container,
    (U, IsLessThan<<T as Shape>::Rank>):
        Map<<U as Container>::Content, IsLessThan<<T as Shape>::Rank>>,
    <(U, IsLessThan<<T as Shape>::Rank>) as Map<
        <U as Container>::Content,
        IsLessThan<<T as Shape>::Rank>,
    >>::Out: Foldable<Both>,
    AllLessThan<U, <T as Shape>::Rank>: Truthy<Permute, <T as crate::ShapeDiagnostic>::Out, ()>,
    (U, PermutationOf<T>): Map<<U as Container>::Content, PermutationOf<T>>,
    <(U, PermutationOf<T>) as Map<<U as Container>::Content, PermutationOf<T>>>::Out: ShapeFragment,
{
    type Out = <(U, PermutationOf<T>) as Map<<U as Container>::Content, PermutationOf<T>>>::Out;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        list,
        num::consts::{U0, U1, U2, U3, U4, U5, U42},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, fragment, shape};

    #[allow(unused)]
    #[test]
    fn permute() {
        type T = shape![U1, U1, U42];
        type I = list![U2, U0, U1];
        assert_type_eq!(<(T, I) as IsCompatible>::Out, True);
        assert_type_eq!(<(T, I) as Compatible>::Out, fragment![U42, U1, U1]);
    }

    #[allow(unused)]
    #[test]
    fn permute2() {
        type T = shape![U1, U2, U42, U1, U42, U2];
        type I1 = list![U2, U0, U1];
        assert_type_eq!(<(T, I1) as IsCompatible>::Out, False);

        type I2 = list![U5, U4, U3, U2, U1, U0];
        assert_type_eq!(<(T, I2) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(T, I2) as Compatible>::Out,
            fragment![U2, U42, U1, U42, U2, U1]
        );
    }

    #[allow(unused)]
    #[test]
    fn permute3() {
        type MyShape = shape![U2, U3, U4, U5];
        type Indices = list![U2, U3, U1, U0];
        assert_type_eq!(
            <(MyShape, Indices) as Compatible>::Out,
            fragment![U4, U5, U3, U2]
        );

        type Another = shape![U2, U3];
        type Idxs = list![U1, U0];
        assert_type_eq!(<(Another, Idxs) as Compatible>::Out, fragment![U3, U2]);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U1, B, U1];
        type Indices = list![U2, U3, U1, U0];
        assert_type_eq!(<(MyShape, Indices) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, Indices) as Compatible>::Out,
            fragment![B, U1, U1, U1]
        );
    }
}
