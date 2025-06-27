# glowstick

This crate makes working with tensors in Rust safe, **easy**, and _fun_ by tracking their shapes in the type system!

Example usage with candle:

```rust
use candle::{DType, Device};  
use glowstick::{Shape2, num::{U1, U2}, debug_tensor};
use glowstick_candle::{Tensor, matmul};

let a: Tensor<Shape2<U2, U1>> = Tensor::zeros(DType::F32, &Device::Cpu).expect("tensor A");
let b: Tensor<Shape2<U1, U2>> = Tensor::zeros(DType::F32, &Device::Cpu).expect("tensor B");

let c = matmul!(a, b).expect("matmul");
//debug_tensor!(c); // [glowstick shape]: (RANK<_2>, (DIM<_2>, DIM<_2>))
```

Several operations are available:

```rust
use candle::{DType, Device};  
use glowstick::{num::{U0, U1, U2, U4, U3, U64, U5, U8}, Shape2, Shape4};
use glowstick_candle::{Tensor, conv2d, squeeze, unsqueeze, narrow, reshape, transpose, flatten, broadcast_add};

#[allow(unused)]
use glowstick::debug_tensor;

let my_tensor: Tensor<Shape2<U8, U8>> = Tensor::zeros(DType::F32, &Device::Cpu).expect("tensor");
//debug_tensor!(my_tensor); // [glowstick shape]: (RANK<_2>, (DIM<_8>, DIM<_8>))

let reshaped = reshape!(my_tensor, [U64]).expect("reshape"); 
//debug_tensor!(reshaped); // [glowstick shape]: (RANK<_1>, (DIM<_64>))

let unsqueezed = unsqueeze!(reshaped, U0, U2).expect("unsqueeze");
//debug_tensor!(unsqueezed); // [glowstick shape]: (RANK<_3>, (DIM<_1>, DIM<_64>, DIM<_1>))

let squeezed = squeeze!(unsqueezed, U0, U2).expect("squeeze");
//debug_tensor!(squeezed); // [glowstick shape]: (RANK<_1>, (DIM<_64>))

let narrowed = narrow!(squeezed, U0: [U8, U5]).expect("narrow");
//debug_tensor!(narrowed); // [glowstick shape]: (RANK<_1>, (DIM<_5>))

let expanded = broadcast_add!(Tensor::<Shape4<U2, U5, U2, U1>>::zeros(DType::F32, &Device::Cpu).unwrap(), narrowed).expect("add");
//debug_tensor!(expanded); // [glowstick shape]: (RANK<_4>, (DIM<_2>, DIM<_5>, DIM<_2>, DIM<_5>))

let swapped = transpose!(expanded, U1: U2).expect("swap");
//debug_tensor!(swapped); // [glowstick shape]: (RANK<_2>, (DIM<_2>, DIM<_5>, DIM<_5>))

let kernel: Tensor<Shape4<U4, U2, U3, U3>> = Tensor::zeros(DType::F32, &Device::Cpu).expect("kernel");
let conv = conv2d!(swapped, kernel, U0, U1, U1, 1).expect("conv2d");
//debug_tensor!(conv); // [glowstick shape]: (RANK<_4>, (DIM<_2>, DIM<_4>, DIM<_3>, DIM<_3>))

let flattened = flatten!(conv, [U1, U2]).expect("flatten");
//debug_tensor!(swapped); // [glowstick shape]: (RANK<_3>, (DIM<_2>, DIM<_12>, DIM<_3>))

assert_eq!(flattened.inner().dims(), [2, 12, 3]);
```

For examples of more extensive usage and integration with popular Rust ML frameworks like [candle](https://github.com/huggingface/candle) and [Burn](https://github.com/tracel-ai/burn), check out the examples directory.

The project is currently pre-1.0: breaking changes will be made!

## Features

- [x] Express tensor shapes as types
- [x] Support for dynamic dimensions (gradual typing)
- [x] Human-readable error messages (sort of)
- [x] Manually check type-level shapes (`debug_tensor!(_)`)
- [ ] Support for all ONNX operations

