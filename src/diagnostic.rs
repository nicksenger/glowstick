use crate::{False, True};

pub trait Operation {}

#[diagnostic::on_unimplemented(
    message = "Incompatible dimensions for operation \"{Op}\": {A}",
    label = "Shape Mismatch"
)]
pub trait Truthy1<Op: Operation, A> {}
#[diagnostic::on_unimplemented(
    message = "Incompatible dimensions for operation \"{Op}\": {A}, {B}",
    label = "Shape Mismatch"
)]
pub trait Truthy<Op: Operation, A, B> {}
#[diagnostic::on_unimplemented(
    message = "Incompatible dimensions for operation \"{Op}\": {A}, {B}, {C}",
    label = "Shape Mismatch"
)]
pub trait Truthy3<Op: Operation, A, B, C> {}
#[diagnostic::on_unimplemented(
    message = "Incompatible dimensions for operation \"{Op}\": {A}, {B}, {C}, {D}",
    label = "Shape Mismatch"
)]
pub trait Truthy4<Op: Operation, A, B, C, D> {}
impl<Op: Operation, A> Truthy1<Op, A> for True {}
impl<Op: Operation, A, B> Truthy<Op, A, B> for True {}
impl<Op: Operation, A, B, C> Truthy3<Op, A, B, C> for True {}
impl<Op: Operation, A, B, C, D> Truthy4<Op, A, B, C, D> for True {}

#[allow(unused)]
#[diagnostic::on_unimplemented(
    message = "Incompatible dimension for operation \"{Op}\": {Lhs}, {Rhs}",
    label = "Shape Mismatch"
)]
pub(crate) trait Falsy<Op: Operation, Lhs, Rhs> {}
impl<Op: Operation, Lhs, Rhs> Falsy<Op, Lhs, Rhs> for False {}

#[allow(unused)]
#[diagnostic::on_unimplemented(
    message = "[glowstick shape]: {T}",
    label = "glowstick::debug_tensor!()",
    note = "This error is due to a `debug_tensor!()` macro invocation."
)]
#[allow(private_bounds)]
pub trait Diagnostic<T>: DebugTensorInvocation {
    crate::private!();
}
trait DebugTensorInvocation {
    crate::private!();
}

#[macro_export]
macro_rules! dbg_shape {
    ($t:ty) => {
        diagnostic_msg::<$t>();
    };
}
#[macro_export]
macro_rules! debug_tensor {
    ($t:ident) => {
        fn diagnostic_msg<T>(t: &T)
        where
            T: $crate::Tensor,
            T: $crate::diagnostic::Diagnostic<
                <<T as $crate::Tensor>::Shape as $crate::ShapeDiagnostic>::Out,
            >,
        {
        }
        diagnostic_msg::<_>(&$t);
    };
}
