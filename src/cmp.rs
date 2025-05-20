use typosaurus::num::{self, Bit, UInt, UTerm, Unsigned};

use crate::Dyn;
use crate::dynamic::{DynMax, DynMin, IsDynEqual, IsDynGreater, IsDynLess};

pub use typosaurus::bool::{And, Bool, False, Or, True};

pub trait Max {
    type Out;
}
impl<L1, L2> Max for (Dyn<L1>, Dyn<L2>)
where
    L1: DynMax<L2>,
{
    type Out = Dyn<<L1 as DynMax<L2>>::Out>;
}
impl<L> Max for (UTerm, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<L> Max for (Dyn<L>, UTerm) {
    type Out = Dyn<L>;
}
impl<U, B, L> Max for (UInt<U, B>, Dyn<L>)
where
    U: Unsigned,
    B: Bit,
    L: DynMax<UInt<U, B>>,
{
    type Out = Dyn<<L as DynMax<UInt<U, B>>>::Out>;
}
impl<U, B, L> Max for (Dyn<L>, UInt<U, B>)
where
    U: Unsigned,
    B: Bit,
    L: DynMax<UInt<U, B>>,
{
    type Out = Dyn<<L as DynMax<UInt<U, B>>>::Out>;
}
impl<T, U> Max for (T, U)
where
    T: num::Max<U>,
{
    type Out = <T as num::Max<U>>::Output;
}

pub trait Min {
    type Out;
}
impl<L1, L2> Min for (Dyn<L1>, Dyn<L2>)
where
    L1: DynMin<L2>,
{
    type Out = Dyn<<L1 as DynMin<L2>>::Out>;
}
impl<L> Min for (UTerm, Dyn<L>) {
    type Out = UTerm;
}
impl<L> Min for (Dyn<L>, UTerm) {
    type Out = UTerm;
}
impl<U, B, L> Min for (UInt<U, B>, Dyn<L>)
where
    U: Unsigned,
    B: Bit,
    L: DynMin<UInt<U, B>>,
{
    type Out = Dyn<<L as DynMin<UInt<U, B>>>::Out>;
}
impl<U, B, L> Min for (Dyn<L>, UInt<U, B>)
where
    U: Unsigned,
    B: Bit,
    L: DynMin<UInt<U, B>>,
{
    type Out = Dyn<<L as DynMin<UInt<U, B>>>::Out>;
}
impl<T, U> Min for (T, U)
where
    T: num::Min<U>,
{
    type Out = <T as num::Min<U>>::Output;
}

pub trait Equal {
    crate::private!();
}
impl<T, U> Equal for (T, U)
where
    (T, U): IsEqual,
    <(T, U) as IsEqual>::Out: typosaurus::bool::Truthy,
{
    crate::private_impl!();
}

pub trait IsEqual {
    type Out;
}
impl<A, B> IsEqual for (Dyn<A>, Dyn<B>)
where
    A: IsDynEqual<B>,
{
    type Out = <A as IsDynEqual<B>>::Out;
}
impl<L> IsEqual for (UTerm, Dyn<L>) {
    type Out = True;
}
impl<L> IsEqual for (Dyn<L>, UTerm) {
    type Out = True;
}
impl<U, B, L> IsEqual for (UInt<U, B>, Dyn<L>)
where
    L: IsDynEqual<UInt<U, B>>,
{
    type Out = <L as IsDynEqual<UInt<U, B>>>::Out;
}
impl<U, B, L> IsEqual for (Dyn<L>, UInt<U, B>)
where
    L: IsDynEqual<UInt<U, B>>,
{
    type Out = <L as IsDynEqual<UInt<U, B>>>::Out;
}
impl<T, U> IsEqual for (T, U)
where
    T: num::IsEqual<U>,
    <T as num::IsEqual<U>>::Output: Bool,
{
    type Out = <<T as num::IsEqual<U>>::Output as Bool>::Out;
}

pub trait IsLess {
    type Out;
}
impl<L1, L2> IsLess for (Dyn<L1>, Dyn<L2>)
where
    L1: IsDynLess<L2>,
{
    type Out = <L1 as IsDynLess<L2>>::Out;
}
impl<L> IsLess for (UTerm, Dyn<L>) {
    type Out = True;
}
impl<L> IsLess for (Dyn<L>, UTerm) {
    type Out = True;
}
impl<U, B, L> IsLess for (UInt<U, B>, Dyn<L>)
where
    L: IsDynLess<UInt<U, B>>,
{
    type Out = <L as IsDynLess<UInt<U, B>>>::Out;
}
impl<U, B, L> IsLess for (Dyn<L>, UInt<U, B>)
where
    L: IsDynLess<UInt<U, B>>,
{
    type Out = <L as IsDynLess<UInt<U, B>>>::Out;
}
impl<T, U> IsLess for (T, U)
where
    T: num::IsLess<U>,
    <T as num::IsLess<U>>::Output: Bool,
{
    type Out = <<T as num::IsLess<U>>::Output as Bool>::Out;
}

pub trait Greater {
    crate::private!();
}
impl<T, U> Greater for (T, U)
where
    (T, U): IsGreater,
    <(T, U) as IsGreater>::Out: typosaurus::bool::Truthy,
{
    crate::private_impl!();
}

pub trait IsGreater {
    type Out;
}
impl<L1, L2> IsGreater for (Dyn<L1>, Dyn<L2>)
where
    L1: IsDynGreater<L2>,
{
    type Out = <L1 as IsDynGreater<L2>>::Out;
}
impl<L> IsGreater for (UTerm, Dyn<L>) {
    type Out = True;
}
impl<L> IsGreater for (Dyn<L>, UTerm) {
    type Out = True;
}
impl<U, B, L> IsGreater for (UInt<U, B>, Dyn<L>)
where
    L: IsDynGreater<UInt<U, B>>,
{
    type Out = <L as IsDynGreater<UInt<U, B>>>::Out;
}
impl<U, B, L> IsGreater for (Dyn<L>, UInt<U, B>)
where
    L: IsDynGreater<UInt<U, B>>,
{
    type Out = <L as IsDynGreater<UInt<U, B>>>::Out;
}
impl<T, U> IsGreater for (T, U)
where
    T: num::IsGreater<U>,
    <T as num::IsGreater<U>>::Output: Bool,
{
    type Out = <<T as num::IsGreater<U>>::Output as Bool>::Out;
}

pub trait IsGreaterOrEqual {
    type Out;
}
impl<T, U> IsGreaterOrEqual for (T, U)
where
    (T, U): IsGreater,
    (T, U): IsEqual,
    (<(T, U) as IsGreater>::Out, <(T, U) as IsEqual>::Out): Or,
{
    type Out = <(<(T, U) as IsGreater>::Out, <(T, U) as IsEqual>::Out) as Or>::Out;
}

pub trait IsLessOrEqual {
    type Out;
}
impl<T, U> IsLessOrEqual for (T, U)
where
    (T, U): IsLess,
    (T, U): IsEqual,
    (<(T, U) as IsLess>::Out, <(T, U) as IsEqual>::Out): Or,
{
    type Out = <(<(T, U) as IsLess>::Out, <(T, U) as IsEqual>::Out) as Or>::Out;
}
