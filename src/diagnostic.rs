use crate::{False, True};

pub(crate) trait Operation {}

#[diagnostic::on_unimplemented(
    message = "Incompatible dimensions for operation \"{Op}\": {Lhs}, {Rhs}",
    label = "Shape Mismatch"
)]
pub(crate) trait Truthy<Op: Operation, Lhs, Rhs> {}
impl<Op: Operation, Lhs, Rhs> Truthy<Op, Lhs, Rhs> for True {}

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
