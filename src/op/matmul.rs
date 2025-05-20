use core::ops::Sub;

use typosaurus::bool::And;
use typosaurus::collections::list::{Empty, List as D};
use typosaurus::num::consts::{U1, U2};
use typosaurus::traits::semigroup::Mappend;

use crate::ShapeDiagnostic;
use crate::{
    Dimension, Dimensioned, IsDimEqual, IsFragEqual, IsRankEqual, Shape, ShapeFragment,
    TakeFragment,
    diagnostic::{self, Truthy},
};

pub trait Operand: Sized + Shape {
    type Pre: ShapeFragment;
    type LastDim: Dimension;
    type NextDim: Dimension;
    crate::private!();
}
impl<T> Operand for T
where
    T: Shape,
    <T as Shape>::Rank: Sub<U2> + Sub<U1>,
    (T, <<T as Shape>::Rank as Sub<U2>>::Output): Dimensioned + TakeFragment,
    (T, <<T as Shape>::Rank as Sub<U1>>::Output): Dimensioned,
{
    type Pre = <(T, <<T as Shape>::Rank as Sub<U2>>::Output) as TakeFragment>::Out;
    type LastDim = <(T, <<T as Shape>::Rank as Sub<U1>>::Output) as Dimensioned>::Out;
    type NextDim = <(T, <<T as Shape>::Rank as Sub<U2>>::Output) as Dimensioned>::Out;
    crate::private_impl!();
}

/// Boolean type operator for `Matmul` compatibility.
///
/// If shapes `T` and `U` are compatible with matrix multiplication,
/// then the `Out` associated type of this trait for `(T, U)` is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U> IsCompatible for (T, U)
where
    T: Operand,
    U: Operand,
    (<T as Shape>::Rank, <U as Shape>::Rank): IsRankEqual,
    (<T as Operand>::Pre, <U as Operand>::Pre): IsFragEqual,
    (<T as Operand>::LastDim, <U as Operand>::NextDim): IsDimEqual,
    (
        <(<T as Shape>::Rank, <U as Shape>::Rank) as IsRankEqual>::Out,
        <(<T as Operand>::Pre, <U as Operand>::Pre) as IsFragEqual>::Out,
    ): And,
    (
        <(
            <(<T as Shape>::Rank, <U as Shape>::Rank) as IsRankEqual>::Out,
            <(<T as Operand>::Pre, <U as Operand>::Pre) as IsFragEqual>::Out,
        ) as And>::Out,
        <(<T as Operand>::LastDim, <U as Operand>::NextDim) as IsDimEqual>::Out,
    ): And,
{
    type Out = <(
        <(
            <(<T as Shape>::Rank, <U as Shape>::Rank) as IsRankEqual>::Out,
            <(<T as Operand>::Pre, <U as Operand>::Pre) as IsFragEqual>::Out,
        ) as And>::Out,
        <(<T as Operand>::LastDim, <U as Operand>::NextDim) as IsDimEqual>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Matmul`-compatibile shapes.
///
/// If shapes `T` and `U` may be used together for matrix multiplication,
/// then the `Out` associated type of this trait for `(T, U)` is the
/// resulting shape.
struct MatrixMultiplication;
impl diagnostic::Operation for MatrixMultiplication {}

pub trait Compatible {
    type Out: ShapeFragment;
    crate::private!();
}
impl<T, U> Compatible for (T, U)
where
    T: Operand + ShapeDiagnostic,
    U: Operand + ShapeDiagnostic,
    (T, U): IsCompatible,
    <(T, U) as IsCompatible>::Out: Truthy<
            MatrixMultiplication,
            <T as crate::ShapeDiagnostic>::Out,
            <U as crate::ShapeDiagnostic>::Out,
        >,
    (
        D<(<T as Operand>::NextDim, Empty)>,
        D<(<U as Operand>::LastDim, Empty)>,
    ): Mappend,
    (
        <T as Operand>::Pre,
        <(
            D<(<T as Operand>::NextDim, Empty)>,
            D<(<U as Operand>::LastDim, Empty)>,
        ) as Mappend>::Out,
    ): Mappend,
    <(
        <T as Operand>::Pre,
        <(
            D<(<T as Operand>::NextDim, Empty)>,
            D<(<U as Operand>::LastDim, Empty)>,
        ) as Mappend>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = <(
        <T as Operand>::Pre,
        <(
            D<(<T as Operand>::NextDim, Empty)>,
            D<(<U as Operand>::LastDim, Empty)>,
        ) as Mappend>::Out,
    ) as Mappend>::Out;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        num::consts::{U0, U1, U2, U3, U100, U120, U200, U240, U360, U1024, U2048, U3600},
    };

    use super::*;
    use crate::{Dyn, dynamic::Any, fragment, shape};
    use typosaurus::bool::{False, True};
    use typosaurus::collections::list::Idx;

    #[allow(unused)]
    #[test]
    fn compat() {
        type ShapeA = shape![U1, U2, U3];
        type ShapeB = shape![U1, U3, U2];

        type OutShape = <(ShapeA, ShapeB) as Compatible>::Out;
        assert_type_eq!(Idx<OutShape, U0>, U1);
        assert_type_eq!(Idx<OutShape, U1>, U2);
        assert_type_eq!(Idx<OutShape, U2>, U2);
        assert_type_eq!(OutShape, fragment![U1, U2, U2]);
    }

    #[allow(unused)]
    #[test]
    fn compat2() {
        type ShapeA = shape![U100, U200, U240, U360];
        type ShapeB = shape![U100, U200, U360, U120];
        type PreA = <ShapeA as Operand>::Pre;
        type PreB = <ShapeB as Operand>::Pre;
        assert_type_eq!(PreA, PreB);
        assert_type_eq!(<(PreA, PreB) as IsFragEqual>::Out, True);
        assert_type_eq!(
            <(ShapeA, ShapeB) as Compatible>::Out,
            fragment![U100, U200, U240, U120]
        );
    }

    #[allow(unused)]
    #[test]
    fn compat3() {
        type ShapeA = shape![
            U100, U200, U100, U200, U100, U200, U100, U200, U100, U200, U2048, U3600
        ];
        type ShapeB = shape![
            U100, U200, U100, U200, U100, U200, U100, U200, U100, U200, U3600, U1024
        ];
        type PreA = <ShapeA as Operand>::Pre;
        type PreB = <ShapeB as Operand>::Pre;
        assert_type_eq!(PreA, PreB);
        assert_type_eq!(<(PreA, PreB) as IsFragEqual>::Out, True);
        assert_type_eq!(
            <(ShapeA, ShapeB) as Compatible>::Out,
            fragment![
                U100, U200, U100, U200, U100, U200, U100, U200, U100, U200, U2048, U1024
            ]
        );
    }

    #[allow(unused)]
    #[test]
    fn compat4() {
        type ShapeA = shape![U2, U3];
        type ShapeB = shape![U3, U2];

        type OutShape = <(ShapeA, ShapeB) as Compatible>::Out;
        assert_type_eq!(Idx<OutShape, U0>, U2);
        assert_type_eq!(Idx<OutShape, U1>, U2);
        assert_type_eq!(OutShape, fragment![U2, U2]);
    }

    #[allow(unused)]
    #[test]
    fn incompat() {
        type ShapeA = shape![U3, U3, U3, U3, U2, U3];
        type ShapeB = shape![U2, U2, U2, U2, U3, U2];
        type PreL = <ShapeA as Operand>::Pre;
        assert_type_eq!(PreL, fragment![U3, U3, U3, U3]);
        type PreR = <ShapeB as Operand>::Pre;
        assert_type_eq!(PreR, fragment![U2, U2, U2, U2]);
        assert_type_eq!(<(PreL, PreR) as IsFragEqual>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn incompat2() {
        type ShapeA = shape![U3, U3, U2, U3];
        type ShapeB = shape![U2, U2, U3, U3, U3, U2];
        type PreL = <ShapeA as Operand>::Pre;
        assert_type_eq!(PreL, fragment![U3, U3]);
        type PreR = <ShapeB as Operand>::Pre;
        assert_type_eq!(PreR, fragment![U2, U2, U3, U3]);
        assert_type_eq!(<(PreL, PreR) as IsFragEqual>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn incompat3() {
        type ShapeA = shape![U2, U2, U3, U3, U3, U2];
        type ShapeB = shape![U3, U3, U2, U3];
        type PreL = <ShapeA as Operand>::Pre;
        assert_type_eq!(PreL, fragment![U2, U2, U3, U3]);
        type PreR = <ShapeB as Operand>::Pre;
        assert_type_eq!(PreR, fragment![U3, U3]);
        assert_type_eq!(<(PreL, PreR) as IsFragEqual>::Out, False);

        //type Incomp = <(ShapeA, ShapeB) as Compatible>::Out;
        //assert_type_eq!(Incomp, fragment![U3, U3]);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type ShapeA = shape![U1, B, U3];
        type ShapeB = shape![U1, U3, B];

        type OutShape = <(ShapeA, ShapeB) as Compatible>::Out;
        assert_type_eq!(OutShape, fragment![U1, B, B]);
    }

    #[allow(unused)]
    #[test]
    fn wild2() {
        type B = Dyn<Any>;
        type ShapeA = shape![U1, U2, B];
        type ShapeB = shape![U1, U3, U2];

        type OutShape = <(ShapeA, ShapeB) as Compatible>::Out;
        assert_type_eq!(OutShape, fragment![U1, U2, U2]);
    }
}
