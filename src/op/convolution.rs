use typosaurus::bool::And;
use typosaurus::bool::monoid::Both;
use typosaurus::collections::{
    Container,
    list::{Empty, List, Zippable},
};
use typosaurus::num::consts::{U0, U1, U2};
use typosaurus::traits::fold::Foldable;
use typosaurus::traits::functor::Map;
use typosaurus::traits::semigroup::Mappend;

use crate::cmp::{IsEqual, IsGreaterOrEqual};
use crate::num::{ZipAdd, ZipDivAddOne, ZipSub, ZipSubOneMul};
use crate::{
    AllGreaterThan, Dimension, Dimensioned, Shape, ShapeFragment, TensorShape,
    diagnostic::{self, Truthy},
    num::Sub,
};
use crate::{IsGreaterThan, ShapeDiagnostic, SkipFragment, TakeFragment};

pub trait Kernel<D>: Sized + Shape {
    type M: Dimension;
    type C: Dimension;
    type Sp: ShapeFragment;
    type DilateZipped;
    crate::private!();
}
impl<T, D> Kernel<D> for T
where
    T: Shape,
    D: ShapeFragment,
    (T, U2): SkipFragment,
    (T, U1): Dimensioned,
    (T, U0): Dimensioned,
    (D, <(T, U2) as SkipFragment>::Out): Zippable,
{
    type M = <(T, U0) as Dimensioned>::Out;
    type C = <(T, U1) as Dimensioned>::Out;
    type Sp = <(T, U2) as SkipFragment>::Out;
    type DilateZipped = <(D, <(T, U2) as SkipFragment>::Out) as Zippable>::Out;
    crate::private_impl!();
}

/// Boolean type operator for `Convolution` compatibility.
///
/// If shapes `T`, `K`, `P`, `S`, and `D` are compatible with convolution,
/// then the `Out` associated type of this trait for `(T, K, P, S, D)` is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, K, P1, P2, S, D> IsCompatible for (T, K, P1, P2, S, D)
where
    T: Shape,
    D: ShapeFragment + Container,
    (D, IsGreaterThan<U0>): Map<<D as Container>::Content, IsGreaterThan<U0>>,
    <(D, IsGreaterThan<U0>) as Map<<D as Container>::Content, IsGreaterThan<U0>>>::Out:
        Foldable<Both>,
    AllGreaterThan<D, U0>: typosaurus::bool::Truthy,
    S: Container,
    (S, IsGreaterThan<U0>): Map<<S as Container>::Content, IsGreaterThan<U0>>,
    <(S, IsGreaterThan<U0>) as Map<<S as Container>::Content, IsGreaterThan<U0>>>::Out:
        Foldable<Both>,
    AllGreaterThan<S, U0>: typosaurus::bool::Truthy,
    (T, U2): SkipFragment,
    (T, U1): Dimensioned,
    (T, U0): Dimensioned,
    (<(T, U2) as SkipFragment>::Out, D): Zippable,
    <(<(T, U2) as SkipFragment>::Out, D) as Zippable>::Out: Container,
    (
        <(<(T, U2) as SkipFragment>::Out, D) as Zippable>::Out,
        ZipSubOneMul,
    ): Map<
            <<(<(T, U2) as SkipFragment>::Out, D) as Zippable>::Out as Container>::Content,
            ZipSubOneMul,
        >,
    <(
        <(<(T, U2) as SkipFragment>::Out, D) as Zippable>::Out,
        ZipSubOneMul,
    ) as Map<
        <<(<(T, U2) as SkipFragment>::Out, D) as Zippable>::Out as Container>::Content,
        ZipSubOneMul,
    >>::Out: ShapeFragment,
    T: Shape,
    K: Kernel<D>,
    P1: ShapeFragment,
    P2: ShapeFragment,
    S: ShapeFragment,
    (<T as Shape>::Rank, <K as Shape>::Rank): IsGreaterOrEqual,
    (
        <T as Shape>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): Sub,
    (
        <(
            <T as Shape>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as Sub>::Out,
        U1,
    ): Sub,
    (
        T,
        <(
            <(
                <T as Shape>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as Sub>::Out,
            U1,
        ) as Sub>::Out,
    ): Dimensioned,
    (
        <(
            T,
            <(
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
                U1,
            ) as Sub>::Out,
        ) as Dimensioned>::Out,
        <K as Kernel<D>>::C,
    ): IsEqual,
    (
        <P1 as ShapeFragment>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): IsEqual,
    (
        <S as ShapeFragment>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): IsEqual,
    (
        <P2 as ShapeFragment>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): IsEqual,
    (
        <D as ShapeFragment>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): IsEqual,
    (
        <(<T as Shape>::Rank, <K as Shape>::Rank) as IsGreaterOrEqual>::Out,
        <(
            <P1 as ShapeFragment>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as IsEqual>::Out,
    ): And,
    (
        <(
            <(<T as Shape>::Rank, <K as Shape>::Rank) as IsGreaterOrEqual>::Out,
            <(
                <P1 as ShapeFragment>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as IsEqual>::Out,
        ) as And>::Out,
        <(
            <S as ShapeFragment>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as IsEqual>::Out,
    ): And,
    (
        <(
            <(
                <(<T as Shape>::Rank, <K as Shape>::Rank) as IsGreaterOrEqual>::Out,
                <(
                    <P1 as ShapeFragment>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as IsEqual>::Out,
            ) as And>::Out,
            <(
                <S as ShapeFragment>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as IsEqual>::Out,
        ) as And>::Out,
        <(
            <P2 as ShapeFragment>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as IsEqual>::Out,
    ): And,
    (
        <D as ShapeFragment>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): IsEqual,
    (
        <(
            <(
                <(
                    <(<T as Shape>::Rank, <K as Shape>::Rank) as IsGreaterOrEqual>::Out,
                    <(
                        <P1 as ShapeFragment>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as IsEqual>::Out,
                ) as And>::Out,
                <(
                    <S as ShapeFragment>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as IsEqual>::Out,
            ) as And>::Out,
            <(
                <P2 as ShapeFragment>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as IsEqual>::Out,
        ) as And>::Out,
        <(
            <(
                T,
                <(
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                    U1,
                ) as Sub>::Out,
            ) as Dimensioned>::Out,
            <K as Kernel<D>>::C,
        ) as IsEqual>::Out,
    ): And,
    (
        <(
            <D as ShapeFragment>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as IsEqual>::Out,
        <(
            <(
                <(
                    <(
                        <(<T as Shape>::Rank, <K as Shape>::Rank) as IsGreaterOrEqual>::Out,
                        <(
                            <P1 as ShapeFragment>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as IsEqual>::Out,
                    ) as And>::Out,
                    <(
                        <S as ShapeFragment>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as IsEqual>::Out,
                ) as And>::Out,
                <(
                    <P2 as ShapeFragment>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as IsEqual>::Out,
            ) as And>::Out,
            <(
                <(
                    T,
                    <(
                        <(
                            <T as Shape>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as Sub>::Out,
                        U1,
                    ) as Sub>::Out,
                ) as Dimensioned>::Out,
                <K as Kernel<D>>::C,
            ) as IsEqual>::Out,
        ) as And>::Out,
    ): And,
{
    type Out = <(
        <(
            <D as ShapeFragment>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as IsEqual>::Out,
        <(
            <(
                <(
                    <(
                        <(<T as Shape>::Rank, <K as Shape>::Rank) as IsGreaterOrEqual>::Out,
                        <(
                            <P1 as ShapeFragment>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as IsEqual>::Out,
                    ) as And>::Out,
                    <(
                        <S as ShapeFragment>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as IsEqual>::Out,
                ) as And>::Out,
                <(
                    <P2 as ShapeFragment>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as IsEqual>::Out,
            ) as And>::Out,
            <(
                <(
                    T,
                    <(
                        <(
                            <T as Shape>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as Sub>::Out,
                        U1,
                    ) as Sub>::Out,
                ) as Dimensioned>::Out,
                <K as Kernel<D>>::C,
            ) as IsEqual>::Out,
        ) as And>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Convolution`-compatibile shapes.
///
/// If shapes `T`, `K`, `P`, `S` and `D` may be used together for convolution,
/// then the `Out` associated type of this trait for `(T, K, P, S, D)` is the
/// resulting shape after convolution.
struct Convolution;
impl diagnostic::Operation for Convolution {}

pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, K, P1, P2, S, D> Compatible for (T, K, P1, P2, S, D)
where
    T: Shape + ShapeDiagnostic,
    D: ShapeFragment,
    K: Kernel<D> + ShapeDiagnostic,
    P1: ShapeFragment,
    P2: ShapeFragment,
    S: ShapeFragment,
    (T, K, P1, P2, S, D): IsCompatible,
    <(T, K, P1, P2, S, D) as IsCompatible>::Out:
        Truthy<Convolution, <T as ShapeDiagnostic>::Out, <K as ShapeDiagnostic>::Out>,
    (
        <T as Shape>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): Sub,
    (
        <(
            <T as Shape>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as Sub>::Out,
        U1,
    ): Sub,
    <K as Kernel<D>>::DilateZipped: Container,
    (<K as Kernel<D>>::DilateZipped, ZipSubOneMul):
        Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>,
    //
    (
        T,
        <(
            <T as Shape>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as Sub>::Out,
    ): SkipFragment,
    (
        <(
            T,
            <(
                <T as Shape>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as Sub>::Out,
        ) as SkipFragment>::Out,
        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
            <<K as Kernel<D>>::DilateZipped as Container>::Content,
            ZipSubOneMul,
        >>::Out,
    ): Zippable,
    <(
        <(
            T,
            <(
                <T as Shape>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as Sub>::Out,
        ) as SkipFragment>::Out,
        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
            <<K as Kernel<D>>::DilateZipped as Container>::Content,
            ZipSubOneMul,
        >>::Out,
    ) as Zippable>::Out: Container,
    (
        <(
            <(
                T,
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
            ) as SkipFragment>::Out,
            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                ZipSubOneMul,
            >>::Out,
        ) as Zippable>::Out,
        ZipSub,
    ): Map<
            <<(
                <(
                    T,
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                ) as SkipFragment>::Out,
                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                    ZipSubOneMul,
                >>::Out,
            ) as Zippable>::Out as Container>::Content,
            ZipSub,
        >,
    <(
        <(
            <(
                T,
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
            ) as SkipFragment>::Out,
            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                ZipSubOneMul,
            >>::Out,
        ) as Zippable>::Out,
        ZipSub,
    ) as Map<
        <<(
            <(
                T,
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
            ) as SkipFragment>::Out,
            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                ZipSubOneMul,
            >>::Out,
        ) as Zippable>::Out as Container>::Content,
        ZipSub,
    >>::Out: ShapeFragment,
    //
    (
        <(
            <(
                <(
                    T,
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                ) as SkipFragment>::Out,
                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                    ZipSubOneMul,
                >>::Out,
            ) as Zippable>::Out,
            ZipSub,
        ) as Map<
            <<(
                <(
                    T,
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                ) as SkipFragment>::Out,
                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                    ZipSubOneMul,
                >>::Out,
            ) as Zippable>::Out as Container>::Content,
            ZipSub,
        >>::Out,
        P1,
    ): Zippable,
    <(
        <(
            <(
                <(
                    T,
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                ) as SkipFragment>::Out,
                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                    ZipSubOneMul,
                >>::Out,
            ) as Zippable>::Out,
            ZipSub,
        ) as Map<
            <<(
                <(
                    T,
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                ) as SkipFragment>::Out,
                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                    ZipSubOneMul,
                >>::Out,
            ) as Zippable>::Out as Container>::Content,
            ZipSub,
        >>::Out,
        P1,
    ) as Zippable>::Out: Container,
    (
        <(
            <(
                <(
                    <(
                        T,
                        <(
                            <T as Shape>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as Sub>::Out,
                    ) as SkipFragment>::Out,
                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                        ZipSubOneMul,
                    >>::Out,
                ) as Zippable>::Out,
                ZipSub,
            ) as Map<
                <<(
                    <(
                        T,
                        <(
                            <T as Shape>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as Sub>::Out,
                    ) as SkipFragment>::Out,
                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                        ZipSubOneMul,
                    >>::Out,
                ) as Zippable>::Out as Container>::Content,
                ZipSub,
            >>::Out,
            P1,
        ) as Zippable>::Out,
        ZipAdd,
    ): Map<
            <<(
                <(
                    <(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out,
                    ZipSub,
                ) as Map<
                    <<(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out as Container>::Content,
                    ZipSub,
                >>::Out,
                P1,
            ) as Zippable>::Out as Container>::Content,
            ZipAdd,
        >,
    //
    (
        <(
            <(
                <(
                    <(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out,
                    ZipSub,
                ) as Map<
                    <<(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out as Container>::Content,
                    ZipSub,
                >>::Out,
                P1,
            ) as Zippable>::Out,
            ZipAdd,
        ) as Map<
            <<(
                <(
                    <(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out,
                    ZipSub,
                ) as Map<
                    <<(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out as Container>::Content,
                    ZipSub,
                >>::Out,
                P1,
            ) as Zippable>::Out as Container>::Content,
            ZipAdd,
        >>::Out,
        P2,
    ): Zippable,
    <(
        <(
            <(
                <(
                    <(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out,
                    ZipSub,
                ) as Map<
                    <<(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out as Container>::Content,
                    ZipSub,
                >>::Out,
                P1,
            ) as Zippable>::Out,
            ZipAdd,
        ) as Map<
            <<(
                <(
                    <(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out,
                    ZipSub,
                ) as Map<
                    <<(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out as Container>::Content,
                    ZipSub,
                >>::Out,
                P1,
            ) as Zippable>::Out as Container>::Content,
            ZipAdd,
        >>::Out,
        P2,
    ) as Zippable>::Out: Container,
    (
        <(
            <(
                <(
                    <(
                        <(
                            <(
                                T,
                                <(
                                    <T as Shape>::Rank,
                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                ) as Sub>::Out,
                            ) as SkipFragment>::Out,
                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                ZipSubOneMul,
                            >>::Out,
                        ) as Zippable>::Out,
                        ZipSub,
                    ) as Map<
                        <<(
                            <(
                                T,
                                <(
                                    <T as Shape>::Rank,
                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                ) as Sub>::Out,
                            ) as SkipFragment>::Out,
                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                ZipSubOneMul,
                            >>::Out,
                        ) as Zippable>::Out as Container>::Content,
                        ZipSub,
                    >>::Out,
                    P1,
                ) as Zippable>::Out,
                ZipAdd,
            ) as Map<
                <<(
                    <(
                        <(
                            <(
                                T,
                                <(
                                    <T as Shape>::Rank,
                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                ) as Sub>::Out,
                            ) as SkipFragment>::Out,
                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                ZipSubOneMul,
                            >>::Out,
                        ) as Zippable>::Out,
                        ZipSub,
                    ) as Map<
                        <<(
                            <(
                                T,
                                <(
                                    <T as Shape>::Rank,
                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                ) as Sub>::Out,
                            ) as SkipFragment>::Out,
                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                ZipSubOneMul,
                            >>::Out,
                        ) as Zippable>::Out as Container>::Content,
                        ZipSub,
                    >>::Out,
                    P1,
                ) as Zippable>::Out as Container>::Content,
                ZipAdd,
            >>::Out,
            P2,
        ) as Zippable>::Out,
        ZipAdd,
    ): Map<
            <<(
                <(
                    <(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                P2,
            ) as Zippable>::Out as Container>::Content,
            ZipAdd,
        >,
    //
    (
        <(
            <(
                <(
                    <(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                P2,
            ) as Zippable>::Out,
            ZipAdd,
        ) as Map<
            <<(
                <(
                    <(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                P2,
            ) as Zippable>::Out as Container>::Content,
            ZipAdd,
        >>::Out,
        S,
    ): Zippable,
    <(
        <(
            <(
                <(
                    <(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                P2,
            ) as Zippable>::Out,
            ZipAdd,
        ) as Map<
            <<(
                <(
                    <(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                P2,
            ) as Zippable>::Out as Container>::Content,
            ZipAdd,
        >>::Out,
        S,
    ) as Zippable>::Out: Container,
    (
        <(
            <(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out,
                                ZipSub,
                            ) as Map<
                                <<(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out as Container>::Content,
                                ZipSub,
                            >>::Out,
                            P1,
                        ) as Zippable>::Out,
                        ZipAdd,
                    ) as Map<
                        <<(
                            <(
                                <(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out,
                                ZipSub,
                            ) as Map<
                                <<(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out as Container>::Content,
                                ZipSub,
                            >>::Out,
                            P1,
                        ) as Zippable>::Out as Container>::Content,
                        ZipAdd,
                    >>::Out,
                    P2,
                ) as Zippable>::Out,
                ZipAdd,
            ) as Map<
                <<(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out,
                                ZipSub,
                            ) as Map<
                                <<(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out as Container>::Content,
                                ZipSub,
                            >>::Out,
                            P1,
                        ) as Zippable>::Out,
                        ZipAdd,
                    ) as Map<
                        <<(
                            <(
                                <(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out,
                                ZipSub,
                            ) as Map<
                                <<(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out as Container>::Content,
                                ZipSub,
                            >>::Out,
                            P1,
                        ) as Zippable>::Out as Container>::Content,
                        ZipAdd,
                    >>::Out,
                    P2,
                ) as Zippable>::Out as Container>::Content,
                ZipAdd,
            >>::Out,
            S,
        ) as Zippable>::Out,
        ZipDivAddOne,
    ): Map<
            <<(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                S,
            ) as Zippable>::Out as Container>::Content,
            ZipDivAddOne,
        >,
    //
    (
        <(
            <(
                T,
                <(
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                    U1,
                ) as Sub>::Out,
            ) as TakeFragment>::Out,
            List<(<K as Kernel<D>>::M, Empty)>,
        ) as Mappend>::Out,
        <(
            <(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                S,
            ) as Zippable>::Out,
            ZipDivAddOne,
        ) as Map<
            <<(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                S,
            ) as Zippable>::Out as Container>::Content,
            ZipDivAddOne,
        >>::Out,
    ): Mappend,
    //
    <(
        <(
            <(
                T,
                <(
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                    U1,
                ) as Sub>::Out,
            ) as TakeFragment>::Out,
            List<(<K as Kernel<D>>::M, Empty)>,
        ) as Mappend>::Out,
        <(
            <(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                S,
            ) as Zippable>::Out,
            ZipDivAddOne,
        ) as Map<
            <<(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                S,
            ) as Zippable>::Out as Container>::Content,
            ZipDivAddOne,
        >>::Out,
    ) as Mappend>::Out: ShapeFragment,
    (
        T,
        <(
            <(
                <T as Shape>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as Sub>::Out,
            U1,
        ) as Sub>::Out,
    ): TakeFragment,
    (
        <(
            T,
            <(
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
                U1,
            ) as Sub>::Out,
        ) as TakeFragment>::Out,
        List<(<K as Kernel<D>>::M, Empty)>,
    ): Mappend,
{
    type Out = TensorShape<
        <(
            <(
                <(
                    T,
                    <(
                        <(
                            <T as Shape>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as Sub>::Out,
                        U1,
                    ) as Sub>::Out,
                ) as TakeFragment>::Out,
                List<(<K as Kernel<D>>::M, Empty)>,
            ) as Mappend>::Out,
            <(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out,
                                ZipAdd,
                            ) as Map<
                                <<(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out as Container>::Content,
                                ZipAdd,
                            >>::Out,
                            P2,
                        ) as Zippable>::Out,
                        ZipAdd,
                    ) as Map<
                        <<(
                            <(
                                <(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out,
                                ZipAdd,
                            ) as Map<
                                <<(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out as Container>::Content,
                                ZipAdd,
                            >>::Out,
                            P2,
                        ) as Zippable>::Out as Container>::Content,
                        ZipAdd,
                    >>::Out,
                    S,
                ) as Zippable>::Out,
                ZipDivAddOne,
            ) as Map<
                <<(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out,
                                ZipAdd,
                            ) as Map<
                                <<(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out as Container>::Content,
                                ZipAdd,
                            >>::Out,
                            P2,
                        ) as Zippable>::Out,
                        ZipAdd,
                    ) as Map<
                        <<(
                            <(
                                <(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out,
                                ZipAdd,
                            ) as Map<
                                <<(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out as Container>::Content,
                                ZipAdd,
                            >>::Out,
                            P2,
                        ) as Zippable>::Out as Container>::Content,
                        ZipAdd,
                    >>::Out,
                    S,
                ) as Zippable>::Out as Container>::Content,
                ZipDivAddOne,
            >>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq, list,
        num::consts::{U1, U2, U3, U4, U5},
    };

    use super::*;
    use crate::shape;
    use typosaurus::bool::True;

    #[allow(unused)]
    #[test]
    fn compat() {
        type T = shape![U1, U2, U5, U5];
        type K = shape![U4, U2, U3, U3];
        type P1 = list![U1, U1];
        type P2 = list![U1, U1];
        type S = list![U1, U1];
        type D = list![U1, U1];

        assert_type_eq!(<(T, K, P1, P2, S, D) as IsCompatible>::Out, True);
        assert_type_eq!(
            shape![U1, U4, U5, U5],
            <(T, K, P1, P2, S, D) as Compatible>::Out
        );

        type ZeroPad = list![U0, U0];
        assert_type_eq!(<(T, K, ZeroPad, ZeroPad, S, D) as IsCompatible>::Out, True);
        assert_type_eq!(
            shape![U1, U4, U3, U3],
            <(T, K, ZeroPad, ZeroPad, S, D) as Compatible>::Out
        );
    }
}
