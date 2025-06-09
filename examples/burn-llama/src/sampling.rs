use burn::tensor::backend::Backend as BurnBackend;
use glowstick::num::{U0, U1};
use glowstick_burn::{argmax, narrow, sort_descending_with_indices};
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    rngs::StdRng,
    SeedableRng,
};

use crate::shape::*;

pub enum Sampler {
    TopP(TopP),
    Argmax,
}

impl Sampler {
    pub fn sample<Backend: BurnBackend>(
        &mut self,
        logits: Rank2Tensor<B, C, Backend>,
    ) -> Rank2IntTensor<B, U1, Backend> {
        match self {
            Self::TopP(s) => s.sample(logits),
            Self::Argmax => {
                argmax!(logits, U1)
            }
        }
    }
}

pub trait Sampling {
    fn sample<Backend: BurnBackend>(
        &mut self,
        logits: Rank2Tensor<B, C, Backend>,
    ) -> Rank2IntTensor<B, U1, Backend>;
}

/// Top-p sampling (nucleus sampling) selects the smallest set of tokens whose cumulative
/// probability mass exceed the threshold p.
pub struct TopP {
    /// Probability threshold for sampling.
    p: f64,
    /// RNG.
    rng: StdRng,
}

impl TopP {
    pub fn new(p: f64, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self { p, rng }
    }
}

impl Sampling for TopP {
    fn sample<Backend: BurnBackend>(
        &mut self,
        probs: Rank2Tensor<B, C, Backend>,
    ) -> Rank2IntTensor<B, U1, Backend> {
        assert_eq!(
            probs.dims()[0],
            1,
            "Naive top-p sampling only supports single-batch tensors"
        );
        let (probs_sort, probs_idx) = sort_descending_with_indices!(probs, U1);

        // TODO: cumsum + Distribution::Multinomial support

        let mut probs_sort = probs_sort.to_data().iter::<f64>().collect::<Vec<_>>();

        let mut cumsum = 0.;
        probs_sort.iter_mut().for_each(|x| {
            if cumsum >= self.p {
                *x = 0.0;
            } else {
                cumsum += *x;
            }
        });

        let next_token_idx = WeightedIndex::new(probs_sort)
            .unwrap()
            .sample(&mut self.rng);

        narrow!(probs_idx, U0: [{ 0 }, { 1 }] => B, U1: [{ next_token_idx }, { 1 }] => U1)
    }
}
