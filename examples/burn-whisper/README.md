## burn-whisper

This example implements OpenAI's Whisper Tiny using the [Burn](https://github.com/tracel-ai/burn) framework, leveraging glowstick where possible for compile-time tensor shapes. Support for dynamic batch-size and beam-width demonstrate the use of `Dyn<_>` dimensions. Much of the code was copied from [Gadersd/whisper-burn](https://github.com/Gadersd/whisper-burn).

Use the following command to transcribe the included WAV file:

`cargo run --bin transcribe --release -- transcribe --inputs=./jfk.wav`

Note that most of the typed shape usage can be found in the model implementation (`src/model/mod.rs`), while `src/tensor.rs` provides a minimal integration of glowstick and burn.

If you're just looking to learn how the crate works, the candle-qwen example is a bit less complex and I'd recommend starting there.
