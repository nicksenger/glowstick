## candle-llama

This example implements Meta's Llama 3.2 architecture using the [candle](https://github.com/huggingface/candle) framework, leveraging glowstick where possible for compile-time tensor shapes. It was largely copied from the corresponding [candle llama example](https://github.com/huggingface/candle/tree/main/candle-examples/examples/llama).

Use the following command to test using SmolLM2 135M:

`cargo run --release`

Note that most of the typed shape usage can be found in the model implementation (`src/llama.rs`).

