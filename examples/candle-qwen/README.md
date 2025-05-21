## candle-qwen

This example implements Alibaba Cloud's Qwen 2 and 3 using the [candle](https://github.com/huggingface/candle) framework, leveraging glowstick where possible for compile-time tensor shapes. It was largely copied from the corresponding [candle qwen example](https://github.com/huggingface/candle/tree/main/candle-examples/examples/qwen).

Use the following command to generate some code:

`cargo run --release -- --model 2.5-0.5b --prompt 'fn print_prime(n: u32) -> bool {'`

Note that most of the typed shape usage can be found in the model implementations (`src/qwen3.rs` & `src/qwen2.rs`).

If you're looking for additional depth, the burn-whisper example is a bit more complex.
