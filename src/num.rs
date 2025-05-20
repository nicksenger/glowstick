use crate::Dyn;
use crate::dynamic::DynAdd;
use crate::dynamic::DynMul;
use typosaurus::collections::list::{Empty, List};
use typosaurus::num::UTerm;

pub use typosaurus::num::consts::*;
pub use typosaurus::num::{UInt, Unsigned};
use typosaurus::traits::functor::Mapper;

pub trait Div {
    type Out;
}
impl<U, B, L> Div for (UInt<U, B>, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<U, B, L> Div for (Dyn<L>, UInt<U, B>) {
    type Out = Dyn<L>;
}
impl<T, U> Div for (T, U)
where
    T: core::ops::Div<U>,
{
    type Out = <T as core::ops::Div<U>>::Output;
}

pub trait Rem {
    type Out;
}
impl<U, B, L> Rem for (UInt<U, B>, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<U, B, L> Rem for (Dyn<L>, UInt<U, B>) {
    type Out = Dyn<L>;
}
impl<T, U> Rem for (T, U)
where
    T: core::ops::Rem<U>,
{
    type Out = <T as core::ops::Rem<U>>::Output;
}

pub trait Add {
    type Out;
}
impl<L> Add for (UTerm, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<L> Add for (Dyn<L>, UTerm) {
    type Out = Dyn<L>;
}
impl<U, B, L> Add for (UInt<U, B>, Dyn<L>)
where
    L: DynAdd<UInt<U, B>>,
{
    type Out = Dyn<<L as DynAdd<UInt<U, B>>>::Out>;
}
impl<U, B, L> Add for (Dyn<L>, UInt<U, B>)
where
    L: DynAdd<UInt<U, B>>,
{
    type Out = Dyn<<L as DynAdd<UInt<U, B>>>::Out>;
}
impl<T, U> Add for (Dyn<T>, Dyn<U>)
where
    T: DynAdd<U>,
{
    type Out = Dyn<<T as DynAdd<U>>::Out>;
}
impl<T, U> Add for (T, U)
where
    T: core::ops::Add<U>,
{
    type Out = <T as core::ops::Add<U>>::Output;
}

pub trait Sub {
    type Out;
}
impl<L> Sub for (UTerm, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<L> Sub for (Dyn<L>, UTerm) {
    type Out = Dyn<L>;
}
impl<U, B, L> Sub for (UInt<U, B>, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<U, B, L> Sub for (Dyn<L>, UInt<U, B>) {
    type Out = Dyn<L>;
}
impl<T, U> Sub for (T, U)
where
    T: core::ops::Sub<U>,
{
    type Out = <T as core::ops::Sub<U>>::Output;
}

pub trait Mul {
    type Out;
}
impl<L> Mul for (UTerm, Dyn<L>) {
    type Out = UTerm;
}
impl<L> Mul for (Dyn<L>, UTerm) {
    type Out = UTerm;
}
impl<U, B, L> Mul for (UInt<U, B>, Dyn<L>)
where
    L: DynMul<UInt<U, B>>,
{
    type Out = Dyn<<L as DynMul<UInt<U, B>>>::Out>;
}
impl<U, B, L> Mul for (Dyn<L>, UInt<U, B>)
where
    L: DynMul<UInt<U, B>>,
{
    type Out = Dyn<<L as DynMul<UInt<U, B>>>::Out>;
}
impl<T, U> Mul for (Dyn<T>, Dyn<U>)
where
    T: DynMul<U>,
{
    type Out = Dyn<<T as DynMul<U>>::Out>;
}
impl<T, U> Mul for (T, U)
where
    T: core::ops::Mul<U>,
{
    type Out = <T as core::ops::Mul<U>>::Output;
}

pub struct ZipSub;
impl<T, U> Mapper<List<(T, List<(U, Empty)>)>> for ZipSub
where
    (T, U): Sub,
{
    type Out = <(T, U) as Sub>::Out;
}
pub struct ZipAdd;
impl<T, U> Mapper<List<(T, List<(U, Empty)>)>> for ZipAdd
where
    (T, U): Add,
{
    type Out = <(T, U) as Add>::Out;
}
pub struct ZipDiv;
impl<T, U> Mapper<List<(T, List<(U, Empty)>)>> for ZipDiv
where
    (T, U): Div,
{
    type Out = <(T, U) as Div>::Out;
}
pub struct ZipDivAddOne;
impl<T, U> Mapper<List<(T, List<(U, Empty)>)>> for ZipDivAddOne
where
    (T, U): Div,
    (<(T, U) as Div>::Out, U1): Add,
{
    type Out = <(<(T, U) as Div>::Out, U1) as Add>::Out;
}

// This is tailored to keff calc
pub struct ZipSubOneMul;
impl<T, U> Mapper<List<(T, List<(U, Empty)>)>> for ZipSubOneMul
where
    (T, U1): Sub,
    (U, U1): Sub,
    (<(T, U1) as Sub>::Out, <(U, U1) as Sub>::Out): Mul,
    (
        <(<(T, U1) as Sub>::Out, <(U, U1) as Sub>::Out) as Mul>::Out,
        U,
    ): Add,
{
    type Out = <(
        <(<(T, U1) as Sub>::Out, <(U, U1) as Sub>::Out) as Mul>::Out,
        U,
    ) as Add>::Out;
}

pub mod monoid {
    use typosaurus::num::consts::{U0, U1};
    use typosaurus::traits::{monoid::Mempty, semigroup::Semigroup};

    use super::*;

    pub struct Addition;
    pub struct Multiplication;

    impl<Lhs, Rhs> Semigroup<Lhs, Rhs> for Addition
    where
        (Lhs, Rhs): Add,
    {
        type Mappend = <(Lhs, Rhs) as Add>::Out;
    }
    impl Mempty for Addition {
        type Out = U0;
    }

    impl<Lhs, Rhs> Semigroup<Lhs, Rhs> for Multiplication
    where
        (Lhs, Rhs): Mul,
    {
        type Mappend = <(Lhs, Rhs) as Mul>::Out;
    }
    impl Mempty for Multiplication {
        type Out = U1;
    }
}
