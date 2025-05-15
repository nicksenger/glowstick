use crate::Dyn;
use typosaurus::num::{UInt, UTerm};

pub use typosaurus::num::Unsigned;
pub use typosaurus::num::consts::*;

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
impl<U, B, L> Add for (UInt<U, B>, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<U, B, L> Add for (Dyn<L>, UInt<U, B>) {
    type Out = Dyn<L>;
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
impl<U, B, L> Mul for (UInt<U, B>, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<U, B, L> Mul for (Dyn<L>, UInt<U, B>) {
    type Out = Dyn<L>;
}
impl<T, U> Mul for (T, U)
where
    T: core::ops::Mul<U>,
{
    type Out = <T as core::ops::Mul<U>>::Output;
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
