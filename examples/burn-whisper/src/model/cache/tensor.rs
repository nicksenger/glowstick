use burn::prelude::*;

use super::Cache;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Inner cache error: {0}")]
    Cache(#[from] super::Error),
}

#[derive(Debug, Clone)]
pub struct TensorCache<B: Backend, const D: usize> {
    cache: Cache<B, D>,
    concat_dim: usize,
    max_seq_len: usize,
}

impl<B: Backend, const D: usize> TensorCache<B, D> {
    pub fn new(concat_dim: usize, max_seq_len: usize) -> Self {
        Self {
            cache: Cache::new(concat_dim, 8),
            concat_dim,
            max_seq_len,
        }
    }

    pub fn cache(&self) -> &Cache<B, D> {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut Cache<B, D> {
        &mut self.cache
    }

    pub fn all_data(&self) -> Option<&Tensor<B, D>> {
        self.cache.all_data()
    }

    pub fn take_all_data(&mut self) -> Option<Tensor<B, D>> {
        self.cache.take_all_data()
    }

    pub fn reset(&mut self) {
        self.cache.reset()
    }

    pub fn append(&mut self, v: Tensor<B, D>) -> Result<(), Error> {
        let dims = v.dims();
        let seq_len = dims[self.concat_dim];
        let current_allocated_size = self.cache.max_seq_len();
        let size_required_for_append = self.cache.current_seq_len() + seq_len;

        if size_required_for_append > current_allocated_size {
            let next_power_of_two = size_required_for_append.next_power_of_two();
            let new_cache_max_seq_len = next_power_of_two.min(self.max_seq_len);

            let mut new_cache = Cache::new(self.concat_dim, new_cache_max_seq_len);
            if let Some(v) = self.cache.take_all_data() {
                new_cache.append(v)?;
            }

            self.cache = new_cache;
        }

        self.cache.append(v).map_err(Into::into)
    }
}
