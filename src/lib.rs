use core::marker::PhantomData;

use dynamic::Dynamic;
use typosaurus::{
    bool::{And, Or},
    collections::{
        Container,
        list::{Rev, Reversible},
    },
    traits::{fold::Foldable, functor::Mapper},
};
use typosaurus::{
    collections::list::Zippable,
    num::{
        Bit, NonZero, UInt, Unsigned,
        consts::{U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10},
    },
};
use typosaurus::{
    collections::list::{All, Indexed, Skippable, Takeable},
    traits::functor::Map,
};

pub mod cmp;
pub mod diagnostic;
pub mod dynamic;
pub mod num;
pub mod op;
use cmp::{IsEqual, IsGreater, IsLess, Max};
use num::{Add, Div, Rem, Sub, monoid::Multiplication};
pub use typosaurus::assert_type_eq;
pub use typosaurus::bool::{False, True};
pub use typosaurus::collections::tuple;
pub use typosaurus::collections::value_list::List as ValueList;
pub use typosaurus::collections::{
    array::Arrayify,
    list::{Empty, List as Shp},
    tuple::Tuplify,
};
use typosaurus::traits::semigroup::Mappend;

#[macro_export]
macro_rules! shape {
    [$a:ident] => { $crate::TensorShape<$crate::Shp<(<$a as $crate::ValidDim>::Out, $crate::Empty)>> };
    [$a:ident,$($bs:ident),+] => { $crate::TensorShape<<($crate::Shp<(<$a as $crate::ValidDim>::Out, $crate::Empty)>, $crate::fragment![$($bs),+]) as $crate::Mappend>::Out> };
}
#[macro_export]
macro_rules! fragment {
    [$a:ident] => { $crate::Shp<(<$a as $crate::ValidDim>::Out, $crate::Empty)> };
    [$a:ident,$($bs:ident),+] => { <($crate::Shp<(<$a as $crate::ValidDim>::Out, $crate::Empty)>, $crate::fragment![$($bs),+]) as $crate::Mappend>::Out };
}
pub type Shape1<A> = shape![A];
pub type Shape2<A, B> = shape![A, B];
pub type Shape3<A, B, C> = shape![A, B, C];
pub type Shape4<A, B, C, D> = shape![A, B, C, D];
pub type Shape5<A, B, C, D, E> = shape![A, B, C, D, E];
pub type Shape6<A, B, C, D, E, F> = shape![A, B, C, D, E, F];
pub type Shape7<A, B, C, D, E, F, G> = shape![A, B, C, D, E, F, G];
pub type Shape8<A, B, C, D, E, F, G, H> = shape![A, B, C, D, E, F, G, H];
pub type Shape9<A, B, C, D, E, F, G, H, I> = shape![A, B, C, D, E, F, G, H, I];
pub type Shape10<A, B, C, D, E, F, G, H, I, J> = shape![A, B, C, D, E, F, G, H, I, J];
pub type Shape11<A, B, C, D, E, F, G, H, I, J, K> = shape![A, B, C, D, E, F, G, H, I, J, K];
pub type Shape12<A, B, C, D, E, F, G, H, I, J, K, L> = shape![A, B, C, D, E, F, G, H, I, J, K, L];

pub(crate) struct Private;
macro_rules! private {
    () => {
        /// The trait is sealed. It was made by those who authored the crate,
        /// and the authors keep it.
        #[doc(hidden)]
        #[allow(private_interfaces)]
        fn __glowstick_private__(&self) -> crate::Private;
    };
}
macro_rules! private_impl {
    () => {
        /// The trait is sealed.
        #[allow(private_interfaces)]
        fn __glowstick_private__(&self) -> crate::Private {
            crate::Private
        }
    };
}
pub(crate) use private;
pub(crate) use private_impl;

/// A dynamic dimension which cannot be checked at compile-time.
pub struct Dyn<Label>(PhantomData<Label>);
impl<T> Tuplify for Dyn<T> {
    type Out = Dyn<T>;
}
impl<T: tuple::Value> tuple::Value for Dyn<T> {
    type Out = <T as tuple::Value>::Out;
    fn value() -> <Self as tuple::Value>::Out {
        <T as tuple::Value>::value()
    }
}

type Product<T> = <T as Foldable<Multiplication>>::Out;

pub struct IsLessThan<M>(PhantomData<M>);
impl<N, M> Mapper<N> for IsLessThan<M>
where
    (N, M): IsLess,
{
    type Out = <(N, M) as IsLess>::Out;
}
pub struct IsGreaterThan<M>(PhantomData<M>);
impl<N, M> Mapper<N> for IsGreaterThan<M>
where
    (N, M): IsGreater,
{
    type Out = <(N, M) as IsGreater>::Out;
}
type LessThan<T, N> = <(T, IsLessThan<N>) as Map<<T as Container>::Content, IsLessThan<N>>>::Out;
type AllLessThan<T, N> = All<LessThan<T, N>>;
type GreaterThan<T, N> =
    <(T, IsGreaterThan<N>) as Map<<T as Container>::Content, IsGreaterThan<N>>>::Out;
pub type AllGreaterThan<T, N> = All<GreaterThan<T, N>>;

pub struct PermutationOf<T>(PhantomData<T>);
impl<T, N> Mapper<N> for PermutationOf<T>
where
    (T, N): Dimensioned,
{
    type Out = <(T, N) as Dimensioned>::Out;
}

pub trait Tensor {
    type Shape: Shape;
}

pub trait Shape {
    type Fragment: ShapeFragment;
    type Dim<N>: Dimension
    where
        (Self, N): Dimensioned;
    type Rank: Rank;
    const RANK: usize;

    fn iter() -> Box<dyn Iterator<Item = usize>>;
    private!();
}
pub trait ValidDim {
    type Out;
    private!();
}
impl<T> ValidDim for T
where
    T: NonZero,
{
    type Out = T;
    private_impl!();
}
impl<L> ValidDim for Dyn<L> {
    type Out = Dyn<L>;
    private_impl!();
}

pub struct TensorShape<T>(T)
where
    T: ShapeFragment;

impl<T> Shape for TensorShape<T>
where
    T: ShapeFragment,
{
    type Fragment = T;
    type Dim<N>
        = <(Self, N) as Dimensioned>::Out
    where
        (Self, N): Dimensioned;
    type Rank = <T as ShapeFragment>::Rank;
    const RANK: usize = <Self::Rank as Unsigned>::USIZE;

    fn iter() -> Box<dyn Iterator<Item = usize>> {
        let n = <<T as ShapeFragment>::Dim>::USIZE;
        match n {
            0 => Box::new(std::iter::empty()),
            d => Box::new(
                std::iter::once(d)
                    .chain(<TensorShape<<T as ShapeFragment>::Next> as Shape>::iter()),
            ),
        }
    }

    private_impl!();
}
impl<T> Tuplify for TensorShape<T>
where
    T: ShapeFragment + Tuplify,
{
    type Out = <T as Tuplify>::Out;
}
impl<T> tuple::Value for TensorShape<T>
where
    T: ShapeFragment + tuple::Value,
{
    type Out = <T as tuple::Value>::Out;
    fn value() -> <Self as tuple::Value>::Out {
        <T as tuple::Value>::value()
    }
}

pub trait Dimensioned {
    type Out: Dimension;
    private!();
}
impl<T, N> Dimensioned for (TensorShape<T>, N)
where
    T: ShapeFragment,
    (T, N): Indexed,
    <(T, N) as Indexed>::Out: Dimension,
{
    type Out = <(T, N) as Indexed>::Out;
    private_impl!();
}

pub trait SkipFragment {
    type Out: ShapeFragment;
    private!();
}
impl<T> SkipFragment for (TensorShape<T>, U0)
where
    T: ShapeFragment,
{
    type Out = T;
    private_impl!();
}
impl<T, U, B> SkipFragment for (TensorShape<T>, UInt<U, B>)
where
    T: ShapeFragment,
    (T, UInt<U, B>): Skippable,
    <(T, UInt<U, B>) as Skippable>::Out: ShapeFragment,
{
    type Out = <(T, UInt<U, B>) as Skippable>::Out;
    private_impl!();
}

pub trait ZipFragment {
    type Out;
    private!();
}
impl<T, U> ZipFragment for (TensorShape<T>, TensorShape<U>)
where
    T: ShapeFragment,
    U: ShapeFragment,
    (T, U): Zippable,
{
    type Out = <(T, U) as Zippable>::Out;
    private_impl!();
}

pub trait TakeFragment {
    type Out: ShapeFragment;
    private!();
}
impl<T, N> TakeFragment for (TensorShape<T>, N)
where
    T: ShapeFragment,
    (T, N): Takeable,
    <(T, N) as Takeable>::Out: ShapeFragment,
{
    type Out = <(T, N) as Takeable>::Out;
    private_impl!();
}

pub trait IsFragEqual {
    type Out;
}
impl IsFragEqual for (Empty, Empty) {
    type Out = True;
}
impl<T, U> IsFragEqual for (Shp<(T, U)>, Empty) {
    type Out = False;
}
impl<T, U> IsFragEqual for (Empty, Shp<(T, U)>) {
    type Out = False;
}
impl<T1, U1, T2, U2> IsFragEqual for (Shp<(T1, U1)>, Shp<(T2, U2)>)
where
    (T1, T2): IsDimEqual,
    (U1, U2): IsFragEqual,
    (
        <(T1, T2) as IsDimEqual>::Out,
        <(U1, U2) as IsFragEqual>::Out,
    ): And,
{
    type Out = <(
        <(T1, T2) as IsDimEqual>::Out,
        <(U1, U2) as IsFragEqual>::Out,
    ) as And>::Out;
}

pub trait MaxDim {
    type Out: Dimension;
    private!();
}
impl<T> MaxDim for TensorShape<T>
where
    T: ShapeFragment + MaxDim,
{
    type Out = <T as MaxDim>::Out;
    private_impl!();
}
impl MaxDim for Empty {
    type Out = U0;
    private_impl!();
}
impl<T, U> MaxDim for Shp<(T, U)>
where
    U: MaxDim,
    (T, <U as MaxDim>::Out): Max,
    <(T, <U as MaxDim>::Out) as Max>::Out: Dimension,
{
    type Out = <(T, <U as MaxDim>::Out) as Max>::Out;
    private_impl!();
}

pub trait MaxDims {
    type Out;
    private!();
}
impl<T, U> MaxDims for (TensorShape<T>, TensorShape<U>)
where
    T: ShapeFragment,
    U: ShapeFragment,
    (T, U): MaxDims,
{
    type Out = <(T, U) as MaxDims>::Out;
    private_impl!();
}
impl MaxDims for (Empty, Empty) {
    type Out = Empty;
    private_impl!();
}
impl<T1, T2, U1, U2> MaxDims for (Shp<(T1, T2)>, Shp<(U1, U2)>)
where
    (T1, U1): Max,
    (T2, U2): MaxDims,
{
    type Out = Shp<(<(T1, U1) as Max>::Out, <(T2, U2) as MaxDims>::Out)>;
    private_impl!();
}

pub trait IsFragEqualOrOne {
    type Out;
    private!();
}
impl<T, U> IsFragEqualOrOne for (TensorShape<T>, TensorShape<U>)
where
    T: ShapeFragment,
    U: ShapeFragment,
    (T, U): IsFragEqualOrOne,
{
    type Out = <(T, U) as IsFragEqualOrOne>::Out;
    private_impl!();
}
impl IsFragEqualOrOne for (Empty, Empty) {
    type Out = True;
    private_impl!();
}
impl<T, U> IsFragEqualOrOne for (Shp<(T, U)>, Empty) {
    type Out = False;
    private_impl!();
}
impl<T, U> IsFragEqualOrOne for (Empty, Shp<(T, U)>) {
    type Out = False;
    private_impl!();
}
impl<T1, U1, T2, U2> IsFragEqualOrOne for (Shp<(T1, U1)>, Shp<(T2, U2)>)
where
    (T1, T2): IsDimEqualOrOne,
    (U1, U2): IsFragEqualOrOne,
    (
        <(T1, T2) as IsDimEqualOrOne>::Out,
        <(U1, U2) as IsFragEqualOrOne>::Out,
    ): And,
{
    type Out = <(
        <(T1, T2) as IsDimEqualOrOne>::Out,
        <(U1, U2) as IsFragEqualOrOne>::Out,
    ) as And>::Out;
    private_impl!();
}

pub trait ShapeFragment: Sized {
    type Dim: Dimension;
    type Rank: Rank;
    type Next: ShapeFragment;
    private!();
}
impl ShapeFragment for Empty {
    type Dim = U0;
    type Rank = U0;
    type Next = Empty;
    private_impl!();
}
impl<Dim, T> ShapeFragment for Shp<(Dim, T)>
where
    Dim: Dimension,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, U1): AddRank,
    <(<T as ShapeFragment>::Rank, U1) as AddRank>::Out: Rank,
{
    type Dim = Dim;
    type Rank = <(<T as ShapeFragment>::Rank, U1) as AddRank>::Out;
    type Next = T;
    private_impl!();
}

pub trait Rank: Unsigned {
    private!();
}
impl<U, B> Rank for UInt<U, B>
where
    U: Unsigned,
    B: Bit,
{
    private_impl!();
}
impl Rank for U0 {
    private_impl!();
}

pub trait IsRankEqual {
    type Out;
    private!();
}
impl<T, U> IsRankEqual for (T, U)
where
    T: Rank,
    U: Rank,
    (T, U): IsEqual,
{
    type Out = <(T, U) as IsEqual>::Out;
    private_impl!();
}

pub trait AddRank {
    type Out: Rank;
    private!();
}
impl<R1> AddRank for (R1, U1)
where
    R1: Rank,
    (R1, U1): Add,
    <(R1, U1) as Add>::Out: Rank,
{
    type Out = <(R1, U1) as Add>::Out;
    private_impl!();
}

pub trait Dimension {
    const USIZE: usize;
    private!();
}
impl<U, B> Dimension for UInt<U, B>
where
    U: Unsigned,
    B: Bit,
{
    const USIZE: usize = <UInt<U, B> as Unsigned>::USIZE;
    private_impl!();
}
impl Dimension for U0 {
    const USIZE: usize = 0;
    private_impl!();
}
impl<L> Dimension for Dyn<L> {
    const USIZE: usize = 0; // TODO: what should this be?
    private_impl!();
}
pub trait IsDimEqual {
    type Out;
    private!();
}
impl<T, U> IsDimEqual for (T, U)
where
    T: Dimension,
    U: Dimension,
    (T, U): IsEqual,
{
    type Out = <(T, U) as IsEqual>::Out;
    private_impl!();
}

pub trait IsDimEqualOrOne {
    type Out;
    private!();
}
impl<T, U> IsDimEqualOrOne for (T, U)
where
    T: Dimension,
    U: Dimension,
    (T, U): IsEqual,
    (T, U1): IsEqual,
    (U, U1): IsEqual,
    (<(T, U) as IsEqual>::Out, <(U, U1) as IsEqual>::Out): Or,
    (
        <(<(T, U) as IsEqual>::Out, <(U, U1) as IsEqual>::Out) as Or>::Out,
        <(T, U1) as IsEqual>::Out,
    ): Or,
{
    type Out = <(
        <(<(T, U) as IsEqual>::Out, <(U, U1) as IsEqual>::Out) as Or>::Out,
        <(T, U1) as IsEqual>::Out,
    ) as Or>::Out;
    private_impl!();
}

// Diagnostic labels
pub struct IDX<T>(PhantomData<T>);
pub struct RANK<T>(PhantomData<T>);
pub struct _0;
pub struct _1;
pub struct _2;
pub struct _3;
pub struct _4;
pub struct _5;
pub struct _6;
pub struct _7;
pub struct _8;
pub struct _9;

impl Tuplify for _0 {
    type Out = _0;
}
impl Tuplify for _1 {
    type Out = _1;
}
impl Tuplify for _2 {
    type Out = _2;
}
impl Tuplify for _3 {
    type Out = _3;
}
impl Tuplify for _4 {
    type Out = _4;
}
impl Tuplify for _5 {
    type Out = _5;
}
impl Tuplify for _6 {
    type Out = _6;
}
impl Tuplify for _7 {
    type Out = _7;
}
impl Tuplify for _8 {
    type Out = _8;
}
impl Tuplify for _9 {
    type Out = _9;
}

pub trait DimensionDiagnostic {
    type Out;
    private!();
}
impl<T> DimensionDiagnostic for (T, U0) {
    type Out = Shp<()>;
    private_impl!();
}
impl<T, U, B> DimensionDiagnostic for (T, UInt<U, B>)
where
    T: ShapeFragment,
    (<T as ShapeFragment>::Dim, U10): Div,
    (<T as ShapeFragment>::Dim, U10): Rem,
    <(<T as ShapeFragment>::Dim, U10) as Rem>::Out: DecimalDiagnostic,
    (<(<T as ShapeFragment>::Dim, U10) as Div>::Out, U0): IsEqual,
    (
        Dec<<T as ShapeFragment>::Dim>,
        <(<(<T as ShapeFragment>::Dim, U10) as Div>::Out, U0) as IsEqual>::Out,
    ): DecimalDiagnostic,
    <(
        Dec<<T as ShapeFragment>::Dim>,
        <(<(<T as ShapeFragment>::Dim, U10) as Div>::Out, U0) as IsEqual>::Out,
    ) as DecimalDiagnostic>::Out: Reversible,
    (UInt<U, B>, U1): Sub,
    (NextFrag<T>, <(UInt<U, B>, U1) as Sub>::Out): DimensionDiagnostic,
{
    type Out = Shp<(
        DIM<
            Rev<
                <(
                    Dec<<T as ShapeFragment>::Dim>,
                    <(<(<T as ShapeFragment>::Dim, U10) as Div>::Out, U0) as IsEqual>::Out,
                ) as DecimalDiagnostic>::Out,
            >,
        >,
        <(NextFrag<T>, <(UInt<U, B>, U1) as Sub>::Out) as DimensionDiagnostic>::Out,
    )>;
    private_impl!();
}

type NextFrag<T> = <T as ShapeFragment>::Next;
pub trait ShapeDiagnostic {
    type Out;
    private!();
}
impl<T> ShapeDiagnostic for TensorShape<T>
where
    T: ShapeFragment,
    <T as ShapeFragment>::Rank: RankDiagnostic<T>,
{
    type Out = <<T as ShapeFragment>::Rank as RankDiagnostic<T>>::Out;
    private_impl!();
}
pub trait RankDiagnostic<T> {
    type Out;
    private!();
}

impl<T, N> RankDiagnostic<T> for N
where
    (N, U10): Div,
    (<(N, U10) as Div>::Out, U0): IsEqual,
    (Dec<N>, <(<(N, U10) as Div>::Out, U0) as IsEqual>::Out): DecimalDiagnostic,
    <(Dec<N>, <(<(N, U10) as Div>::Out, U0) as IsEqual>::Out) as DecimalDiagnostic>::Out: Tuplify,
    (T, N): DimensionDiagnostic,
    <(T, N) as DimensionDiagnostic>::Out: Tuplify,
{
    type Out = (
        RANK<<<(Dec<N>, <(<(N, U10) as Div>::Out, U0) as IsEqual>::Out) as DecimalDiagnostic>::Out as Tuplify>::Out>,
        <<(T, N) as DimensionDiagnostic>::Out as Tuplify>::Out,
    );
    private_impl!();
}

pub struct DIM<T>(PhantomData<T>);
impl<T: Tuplify> Tuplify for DIM<T> {
    type Out = DIM<<T as Tuplify>::Out>;
}
pub trait DecimalDiagnostic {
    type Out;
    private!();
}
macro_rules! decimpl {
    [($n:ident,$t:ident)] => {
        impl DecimalDiagnostic for $n {
            type Out = $t;
            private_impl!();
        }
    };
    [($n:ident,$t:ident),$(($ns:ident,$ts:ident)),+] => { decimpl![($n,$t)]; decimpl![$(($ns,$ts)),+]; };
}
decimpl![
    (U0, _0),
    (U1, _1),
    (U2, _2),
    (U3, _3),
    (U4, _4),
    (U5, _5),
    (U6, _6),
    (U7, _7),
    (U8, _8),
    (U9, _9)
];
pub struct Dec<T>(PhantomData<T>);
impl<L> DecimalDiagnostic for Dyn<L>
where
    L: Dynamic,
{
    type Out = Dyn<<L as Dynamic>::Label>;
    private_impl!();
}
impl<T> DecimalDiagnostic for (Dec<T>, True)
where
    (T, U10): Rem,
    <(T, U10) as Rem>::Out: DecimalDiagnostic,
{
    type Out = Shp<(<<(T, U10) as Rem>::Out as DecimalDiagnostic>::Out, Shp<()>)>;
    private_impl!();
}
impl<T> DecimalDiagnostic for (Dec<T>, False)
where
    (T, U10): Div,
    (T, U10): Rem,
    <(T, U10) as Rem>::Out: DecimalDiagnostic,
    (<(<(T, U10) as Div>::Out, U10) as Div>::Out, U0): IsEqual,
    (<(T, U10) as Div>::Out, U10): Div,
    (
        Dec<<(T, U10) as Div>::Out>,
        <(<(<(T, U10) as Div>::Out, U10) as Div>::Out, U0) as IsEqual>::Out,
    ): DecimalDiagnostic,
    (Dec<T>, True): DecimalDiagnostic,
{
    type Out = Shp<(
        <<(T, U10) as Rem>::Out as DecimalDiagnostic>::Out,
        <(
            Dec<<(T, U10) as Div>::Out>,
            <(<(<(T, U10) as Div>::Out, U10) as Div>::Out, U0) as IsEqual>::Out,
        ) as DecimalDiagnostic>::Out,
    )>;
    private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        num::consts::{U0, U2, U3},
    };

    use super::*;

    #[allow(unused)]
    #[test]
    fn dims() {
        type MyShape = shape![U1, U2, U3];
        assert_type_eq!(<MyShape as Shape>::Dim<U0>, U1);
        assert_type_eq!(<MyShape as Shape>::Dim<U1>, U2);
        assert_type_eq!(<MyShape as Shape>::Dim<U2>, U3);
    }

    #[allow(unused)]
    #[test]
    fn dim_eq() {
        type F = <(U0, U1) as IsDimEqual>::Out;
        assert_type_eq!(F, False);

        type T = <(U1, U1) as IsDimEqual>::Out;
        assert_type_eq!(T, True);
    }

    #[allow(unused)]
    #[test]
    fn frag_eq() {
        type La1 = fragment![U1, U1, U2];
        type Lb1 = fragment![U1, U1, U1];
        assert_type_eq!(<(La1, Lb1) as IsFragEqual>::Out, False);

        type La2 = fragment![U1, U1, U2];
        type Lb2 = fragment![U1, U1, U2];
        type Fe = <(La2, Lb2) as IsFragEqual>::Out;
        assert_type_eq!(Fe, True);

        assert_type_eq!(<(Empty, Empty) as IsFragEqual>::Out, True);
        assert_type_eq!(<(fragment![U1], Empty) as IsFragEqual>::Out, False);
        assert_type_eq!(<(Empty, fragment![U1]) as IsFragEqual>::Out, False);
        assert_type_eq!(<(fragment![U1], fragment![U1]) as IsFragEqual>::Out, True);
    }

    #[allow(unused)]
    #[test]
    fn dyn_diag() {
        struct BatchSize;
        impl Dynamic for BatchSize {
            type Label = Self;
        }
        type B = Dyn<BatchSize>;
        type DynShape = shape![U1, U1, B];
        type Diag = <DynShape as ShapeDiagnostic>::Out;
        assert_type_eq!(Diag, (RANK<_3>, (DIM<_1>, DIM<_1>, DIM<Dyn<BatchSize>>)));
    }
}

#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;
