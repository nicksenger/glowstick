use std::marker::PhantomData;

use typosaurus::bool::{And, Bool, False, True};
use typosaurus::collections::tuple::{self, Tuplify};
use typosaurus::num::{Bit, Unsigned};

use crate::cmp::{IsEqual, Max, Min};
use crate::num::{Mul, UInt};
use crate::DecimalDiagnostic;

pub trait Dynamic {
    type Label;
}

pub trait IsDynEqual<Rhs> {
    type Out;
}
pub trait IsDynGreater<Rhs> {
    type Out;
}
pub trait IsDynLess<Rhs> {
    type Out;
}
pub trait DynMax<Rhs> {
    type Out;
}
pub trait DynMin<Rhs> {
    type Out;
}
pub trait DynAdd<Rhs> {
    type Out;
}
pub trait DynMul<Rhs> {
    type Out;
}

pub struct Any;
impl Dynamic for Any {
    type Label = Any;
}
impl Tuplify for Any {
    type Out = ();
}
impl tuple::Value for Any {
    type Out = ();
    fn value() {}
}
impl<U> IsDynEqual<U> for Any {
    type Out = True;
}
impl<U> IsDynGreater<U> for Any {
    type Out = True;
}
impl<U> IsDynLess<U> for Any {
    type Out = True;
}
impl<U> DynMax<U> for Any {
    type Out = Any;
}
impl<U> DynMin<U> for Any {
    type Out = Any;
}
impl<U> DynAdd<U> for Any {
    type Out = Any;
}
impl<U> DynMul<U> for Any {
    type Out = Any;
}
pub type Wild = Any;

pub struct Term<Coeff, Var>(PhantomData<Coeff>, PhantomData<Var>);
impl<Coeff1, Var1, Coeff2, Var2> IsDynEqual<Term<Coeff2, Var2>> for Term<Coeff1, Var1>
where
    (Coeff1, Coeff2): IsEqual,
    Var1: IsDynEqual<Var2>,
    (
        <(Coeff1, Coeff2) as IsEqual>::Out,
        <Var1 as IsDynEqual<Var2>>::Out,
    ): And,
{
    type Out = <(
        <(Coeff1, Coeff2) as IsEqual>::Out,
        <Var1 as IsDynEqual<Var2>>::Out,
    ) as And>::Out;
}
impl<Coeff1, Var1, Coeff2, Var2> DynMul<Term<Coeff2, Var2>> for Term<Coeff1, Var1>
where
    (Coeff1, Coeff2): Mul,
    Var1: DynMul<Var2>,
{
    type Out = Term<<(Coeff1, Coeff2) as Mul>::Out, <Var1 as DynMul<Var2>>::Out>;
}
impl<Coeff1, Var1, Coeff2, Var2> DynMax<Term<Coeff2, Var2>> for Term<Coeff1, Var1>
where
    (Coeff1, Coeff2): Max,
    Var1: DynMax<Var2>,
{
    type Out = Term<<(Coeff1, Coeff2) as Max>::Out, <Var1 as DynMax<Var2>>::Out>;
}
impl<Coeff, Var, U, B> IsDynEqual<UInt<U, B>> for Term<Coeff, Var> {
    type Out = True;
}
impl<Coeff, Var> IsDynEqual<Any> for Term<Coeff, Var> {
    type Out = True;
}
impl<Coeff, Var, T> IsDynGreater<T> for Term<Coeff, Var> {
    type Out = True;
}
impl<Coeff, Var, T> IsDynLess<T> for Term<Coeff, Var> {
    type Out = True;
}
impl<Coeff, Var, U, B> DynMul<UInt<U, B>> for Term<Coeff, Var>
where
    (Coeff, UInt<U, B>): Mul,
{
    type Out = Term<<(Coeff, UInt<U, B>) as Mul>::Out, Var>;
}
impl<Coeff, Var> Dynamic for Term<Coeff, Var>
where
    Coeff: DecimalDiagnostic,
{
    type Label = (<Coeff as DecimalDiagnostic>::Out, Var);
}
