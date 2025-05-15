use std::ops::Range;

use burn::prelude::*;

pub mod kv;
pub mod tensor;

pub use kv::KvCache;
pub use tensor::TensorCache;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Cache: maximum sequence length exceeded")]
    AboveMaxSequenceLength,
    #[error("Invalid size for input tensor")]
    SizeMismatch,
}

#[derive(Debug, Clone)]
pub struct Cache<B: Backend, const D: usize> {
    all_data: Option<Tensor<B, D>>,
    dim: usize,
    current_seq_len: usize,
    max_seq_len: usize,
}

impl<B: Backend, const D: usize> Cache<B, D> {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            max_seq_len,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> Option<&Tensor<B, D>> {
        self.all_data.as_ref()
    }

    pub fn take_all_data(&mut self) -> Option<Tensor<B, D>> {
        self.all_data.take()
    }

    pub fn current_data(&self) -> Result<Option<Tensor<B, D>>, Error> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => Some(d.clone().narrow(self.dim, 0, self.current_seq_len)),
        };
        Ok(data)
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.all_data = None;
    }

    pub fn append(&mut self, src: Tensor<B, D>) -> Result<(), Error> {
        let mut src_dims = src.dims();
        let src_seq_len = src_dims[self.dim];

        let nd = if self.all_data.is_none() {
            src_dims[self.dim] = self.max_seq_len;
            let ad = Tensor::zeros(src_dims, &src.device());
            Some(ad)
        } else {
            None
        };
        if self.current_seq_len + src_seq_len > self.max_seq_len {
            return Err(Error::AboveMaxSequenceLength);
        }
        // ad.slice_set(src, self.dim, self.current_seq_len)?;
        let ranges = (0..src.dims().len())
            .map(|i| {
                if i == self.dim {
                    self.current_seq_len..(self.current_seq_len + src_seq_len)
                } else {
                    0..src_dims[i]
                }
            })
            .collect::<Vec<_>>();
        let ranges: [Range<usize>; D] = ranges.try_into().map_err(|_| Error::SizeMismatch)?;

        let ad = match nd {
            None => self.all_data().cloned(),
            Some(data) => Some(data),
        }
        .unwrap();
        let new = ad.slice_assign::<D>(ranges, src);
        self.current_seq_len += src_seq_len;
        self.all_data = Some(new);
        Ok(())
    }
}
