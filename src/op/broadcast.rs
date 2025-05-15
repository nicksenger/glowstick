use typosaurus::traits::semigroup::Mappend;

use crate::{
    cmp::IsGreaterOrEqual,
    diagnostic::{self, Truthy},
    num::Sub,
    IsFragEqualOrOne, MaxDims, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment, TakeFragment,
    TensorShape,
};

struct Broadcast;
impl diagnostic::Operation for Broadcast {}

/// Boolean type operator for `Broadcast` compatibility.
///
/// If shape `U` may be expanded to shape `T`, then the `Out`
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
    T: ShapeFragment,
    U: ShapeFragment,
    (
        <TensorShape<T> as Shape>::Rank,
        <TensorShape<U> as Shape>::Rank,
    ): IsGreaterOrEqual,
    <(
        <TensorShape<T> as Shape>::Rank,
        <TensorShape<U> as Shape>::Rank,
    ) as IsGreaterOrEqual>::Out: Truthy<
        Broadcast,
        <TensorShape<T> as crate::ShapeDiagnostic>::Out,
        <TensorShape<U> as crate::ShapeDiagnostic>::Out,
    >,
    (
        <TensorShape<T> as Shape>::Rank,
        <TensorShape<U> as Shape>::Rank,
    ): Sub,
    (
        TensorShape<T>,
        <(
            <TensorShape<T> as Shape>::Rank,
            <TensorShape<U> as Shape>::Rank,
        ) as Sub>::Out,
    ): TakeFragment,
    (
        TensorShape<T>,
        <(
            <TensorShape<T> as Shape>::Rank,
            <TensorShape<U> as Shape>::Rank,
        ) as Sub>::Out,
    ): SkipFragment,
    <(
        TensorShape<T>,
        <(
            <TensorShape<T> as Shape>::Rank,
            <TensorShape<U> as Shape>::Rank,
        ) as Sub>::Out,
    ) as SkipFragment>::Out: ShapeFragment,
    (
        <(
            TensorShape<T>,
            <(
                <TensorShape<T> as Shape>::Rank,
                <TensorShape<U> as Shape>::Rank,
            ) as Sub>::Out,
        ) as SkipFragment>::Out,
        U,
    ): IsFragEqualOrOne,
{
    type Out = <(
        <(
            TensorShape<T>,
            <(
                <TensorShape<T> as Shape>::Rank,
                <TensorShape<U> as Shape>::Rank,
            ) as Sub>::Out,
        ) as SkipFragment>::Out,
        U,
    ) as IsFragEqualOrOne>::Out;
    crate::private_impl!();
}

/// Type operator for broadcast-compatible shapes.
///
/// If shape `U` may be broadcast as shape `T`, then the
/// `Out` assocatied type of this trait for `(T, U)` is
/// the resulting shape.
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
    (
        <TensorShape<T> as Shape>::Rank,
        <TensorShape<U> as Shape>::Rank,
    ): IsGreaterOrEqual,
    <(
        <TensorShape<T> as Shape>::Rank,
        <TensorShape<U> as Shape>::Rank,
    ) as IsGreaterOrEqual>::Out: Truthy<
        Broadcast,
        <TensorShape<T> as crate::ShapeDiagnostic>::Out,
        <TensorShape<U> as crate::ShapeDiagnostic>::Out,
    >,
    (
        <TensorShape<T> as Shape>::Rank,
        <TensorShape<U> as Shape>::Rank,
    ): Sub,
    (
        TensorShape<T>,
        <(
            <TensorShape<T> as Shape>::Rank,
            <TensorShape<U> as Shape>::Rank,
        ) as Sub>::Out,
    ): TakeFragment,
    <(
        TensorShape<T>,
        <(
            <TensorShape<T> as Shape>::Rank,
            <TensorShape<U> as Shape>::Rank,
        ) as Sub>::Out,
    ) as TakeFragment>::Out: ShapeFragment,
    (
        TensorShape<T>,
        <(
            <TensorShape<T> as Shape>::Rank,
            <TensorShape<U> as Shape>::Rank,
        ) as Sub>::Out,
    ): SkipFragment,
    (
        <(
            TensorShape<T>,
            <(
                <TensorShape<T> as Shape>::Rank,
                <TensorShape<U> as Shape>::Rank,
            ) as Sub>::Out,
        ) as SkipFragment>::Out,
        U,
    ): IsFragEqualOrOne,
    (TensorShape<T>, TensorShape<U>): IsCompatible,
    <(TensorShape<T>, TensorShape<U>) as IsCompatible>::Out: Truthy<
        Broadcast,
        <TensorShape<T> as crate::ShapeDiagnostic>::Out,
        <TensorShape<U> as crate::ShapeDiagnostic>::Out,
    >,
    (
        <(
            TensorShape<T>,
            <(
                <TensorShape<T> as Shape>::Rank,
                <TensorShape<U> as Shape>::Rank,
            ) as Sub>::Out,
        ) as SkipFragment>::Out,
        U,
    ): MaxDims,
    (
        <(
            TensorShape<T>,
            <(
                <TensorShape<T> as Shape>::Rank,
                <TensorShape<U> as Shape>::Rank,
            ) as Sub>::Out,
        ) as TakeFragment>::Out,
        <(
            <(
                TensorShape<T>,
                <(
                    <TensorShape<T> as Shape>::Rank,
                    <TensorShape<U> as Shape>::Rank,
                ) as Sub>::Out,
            ) as SkipFragment>::Out,
            U,
        ) as MaxDims>::Out,
    ): Mappend,
    <(
        <(
            TensorShape<T>,
            <(
                <TensorShape<T> as Shape>::Rank,
                <TensorShape<U> as Shape>::Rank,
            ) as Sub>::Out,
        ) as TakeFragment>::Out,
        <(
            <(
                TensorShape<T>,
                <(
                    <TensorShape<T> as Shape>::Rank,
                    <TensorShape<U> as Shape>::Rank,
                ) as Sub>::Out,
            ) as SkipFragment>::Out,
            U,
        ) as MaxDims>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                TensorShape<T>,
                <(
                    <TensorShape<T> as Shape>::Rank,
                    <TensorShape<U> as Shape>::Rank,
                ) as Sub>::Out,
            ) as TakeFragment>::Out,
            <(
                <(
                    TensorShape<T>,
                    <(
                        <TensorShape<T> as Shape>::Rank,
                        <TensorShape<U> as Shape>::Rank,
                    ) as Sub>::Out,
                ) as SkipFragment>::Out,
                U,
            ) as MaxDims>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use core::marker::PhantomData;

    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        num::consts::{U1, U128, U16, U42, U422},
    };

    use super::*;

    use crate::{shape, Dyn};

    #[allow(unused)]
    #[test]
    fn single() {
        type Rhs = shape![U1];
        type Lhs = shape![U1];
        assert_type_eq!(<(Lhs, Rhs) as IsCompatible>::Out, True);
        assert_type_eq!(<(Lhs, Rhs) as Compatible>::Out, shape![U1]);
    }

    #[allow(unused)]
    #[test]
    fn ones() {
        type Rhs = shape![U1, U1, U1, U42];
        type Lhs = shape![U42, U42, U42, U1];
        assert_type_eq!(<(Lhs, Rhs) as IsCompatible>::Out, True);
        assert_type_eq!(<(Lhs, Rhs) as Compatible>::Out, shape![U42, U42, U42, U42]);
    }

    #[allow(unused)]
    #[test]
    fn smaller() {
        type Rhs = shape![U1, U1, U42];
        type Lhs = shape![U42, U42, U42, U42, U42, U42, U42, U42];
        assert_type_eq!(<(Lhs, Rhs) as IsCompatible>::Out, True);
    }

    #[allow(unused)]
    #[test]
    fn incompat() {
        type Rhs = shape![U1, U422, U42];
        type Lhs = shape![U42, U42, U42, U42, U42, U42, U42, U42];
        assert_type_eq!(<(Lhs, Rhs) as IsCompatible>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        struct BatchSize;
        type B = Dyn<BatchSize>;
        type Rhs = shape![U1, U1, U1, B];
        type Lhs = shape![U42, U42, U42, U1];

        type OutShape = <(Rhs, Lhs) as Compatible>::Out;
        assert_type_eq!(OutShape, shape![U42, U42, U42, B]);

        fn compat<A, B>() -> PhantomData<<(A, B) as Compatible>::Out>
        where
            (A, B): Compatible,
        {
            PhantomData
        }
        compat::<Lhs, Rhs>();
    }

    #[allow(unused)]
    #[test]
    fn op() {
        use core::marker::PhantomData;
        struct BatchSize;
        type B = Dyn<BatchSize>;
        {
            struct Op<'a, T>(&'a T);
            impl<S> Op<'_, PhantomData<S>>
            where
                S: crate::Shape,
                (crate::Shape4<U1, U16, B, U128>, S): Compatible,
            {
                #[allow(clippy::type_complexity)]
                pub fn broadcast_as(
                    &self,
                ) -> PhantomData<<(crate::Shape4<U1, U16, B, U128>, S) as Compatible>::Out>
                {
                    PhantomData
                }
            }
            Op(&PhantomData::<crate::Shape4<U1, U1, U1, U1>>).broadcast_as();
        }
    }
}
