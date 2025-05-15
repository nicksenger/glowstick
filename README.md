# glowstick

This is a crate for checking tensor shapes at compile time in stable Rust. It consists of types and traits which may be used to gain additional type safety when working with tensors in a framework-agnostic manner.

As an example, consider a rank 3 tensor shape `ExampleShape` with dimensions [1, 2, 3]. Glowstick allows for type-level expression of this shape and associated operations as follows:

```rust
use glowstick::{Shape2, Shape3, Shape4};
use glowstick::num::{U0, U1, U2, U3};

type ExampleShape = Shape3<U1, U2, U3>;

// Unsqueeze
type UnsqueezedExampleShape = <(ExampleShape, U0) as glowstick::op::unsqueeze::Compatible>::Out;
glowstick::assert_type_eq!(UnsqueezedExampleShape, Shape4<U1, U1, U2, U3>);

// Broadcast
type Expanded = <(ExampleShape, Shape2<U2, U1>) as glowstick::op::broadcast::Compatible>::Out;
glowstick::assert_type_eq!(ExampleShape, Expanded);
```

Shape mismatches will not compile:

```rust
use glowstick::{Shape2, Shape3, Shape4, True, False};
use glowstick::num::{U0, U1, U2, U3};

type ExampleShape = Shape3<U1, U2, U3>;

glowstick::assert_type_eq!(<(ExampleShape, Shape3<U1, U1, U1>) as glowstick::op::matmul::IsCompatible>::Out, False);
// type Output = <(ExampleShape, Shape3<U1, U1, U1>) as glowstick::op::matmul::Compatible>::Out; // Does not compile
```

For examples of more extensive usage and integration with popular Rust ML frameworks like [candle](https://github.com/huggingface/candle) and [Burn](https://github.com/tracel-ai/burn), check out the examples directory.

The project is currently pre-1.0: breaking changes will be made!

## Features

- [x] Express tensor shapes as types
- [x] Basic support for dynamic dimensions (gradual typing)
- [x] Human-readable error messages
- [x] Manually check type-level shapes (use `debug_tensor!(_)`)
- [x] Support for common operations:
    - [x] broadcast (expand)
    - [x] flatten
    - [x] matmul
    - [x] narrow
    - [x] permute
    - [x] reshape
    - [x] squeeze
    - [x] stack
    - [x] transpose (swap)
    - [x] unsqueeze

