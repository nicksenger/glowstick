## burn-whisper

This example implements Alibaba's Qwen3 0.6B using the [candle](https://github.com/huggingface/candle) framework, leveraging glowstick where possible for compile-time tensor shapes. It was largely copied from the corresponding [candle qwen example](https://github.com/huggingface/candle/tree/main/candle-examples/examples/qwen).

Use the following command to transcribe the included WAV file:

`cargo run --release --features=metal -- --model 3-0.6b --prompt 'fn print_prime(n: u32) -> bool {'`

Note that most of the typed shape usage can be found in the model implementation (`src/qwen3.rs`), while `src/tensor.rs` provides a minimal integration of glowstick and candle.

If you're looking for additional depth, the burn-whisper example is a bit more complex.
