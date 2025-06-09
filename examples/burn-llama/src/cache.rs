use burn::tensor::{backend::Backend as BurnBackend, Device, Tensor as BurnTensor};

use crate::shape::*;

pub(crate) struct AutoregressiveCache<Backend: BurnBackend> {
    /// Tensor cache with shape `[batch_size, num_heads, seq_len, d_model]`
    cache: BurnTensor<Backend, 4>,
    pub(crate) max_seq_len: usize,
    cur_seq_len: usize,
}

impl<Backend: BurnBackend> AutoregressiveCache<Backend> {
    /// Creates a new empty cache.
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        d_model: usize,
        device: &Device<Backend>,
    ) -> Self {
        Self {
            cache: BurnTensor::empty([max_batch_size, num_heads, max_seq_len, d_model], device),
            max_seq_len,
            cur_seq_len: 0,
        }
    }

    /// Reset the cache state.
    pub fn reset(&mut self) {
        self.cache = BurnTensor::empty(self.cache.shape(), &self.cache.device());
        self.cur_seq_len = 0;
    }

    pub fn forward(
        &mut self,
        tensor: Rank4Tensor<B, K, N, H, Backend>,
    ) -> Rank4Tensor<B, K, N, H, Backend> {
        let [batch_size, num_heads, seq_len, d_model] = tensor.dims();
        let mut new_seq_len = self.cur_seq_len + seq_len;

        if new_seq_len > self.max_seq_len {
            self.cur_seq_len = self.max_seq_len - seq_len;
            let prev_slice = self.cache.clone().slice([
                0..batch_size,
                0..num_heads,
                seq_len..self.max_seq_len,
                0..d_model,
            ]);
            self.cache = self.cache.clone().slice_assign(
                [0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..d_model],
                prev_slice,
            );
            new_seq_len = self.max_seq_len;
        }

        self.cache = self.cache.clone().slice_assign(
            [
                0..batch_size,
                0..num_heads,
                self.cur_seq_len..new_seq_len,
                0..d_model,
            ],
            tensor.into_inner(),
        );

        self.cur_seq_len += seq_len;

        self.cache
            .clone()
            .slice([0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..d_model])
            .try_into()
            .unwrap()
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }
}
