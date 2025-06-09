## burn-llama

This example implements Meta's Llama 3.2 architecture using the [burn](https://github.com/tracel-ai/burn) framework, leveraging glowstick where possible for compile-time tensor shapes. It was largely copied from the corresponding [burn llama example](https://github.com/tracel-ai/models/tree/main/llama-burn).

Use the following command to test using Llama 3.2 1B:

`cargo run --release`

Note that most of the typed shape usage can be found in the model implementation (`src/transformer.rs`).

