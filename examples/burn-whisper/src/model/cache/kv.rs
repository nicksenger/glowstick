use burn::prelude::*;

use super::Cache;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Inner cache error: {0}")]
    Cache(#[from] super::Error),
}

#[derive(Debug, Clone)]
pub struct KvCache<Bk: Backend, Bv: Backend, const DK: usize, const DV: usize> {
    k: Cache<Bk, DK>,
    v: Cache<Bv, DV>,
}

impl<Bk: Backend, Bv: Backend, const DK: usize, const DV: usize> KvCache<Bk, Bv, DK, DV> {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let k = Cache::new(dim, max_seq_len);
        let v = Cache::new(dim, max_seq_len);
        Self { k, v }
    }

    pub fn k_cache(&self) -> &Cache<Bk, DK> {
        &self.k
    }

    pub fn v_cache(&self) -> &Cache<Bv, DV> {
        &self.v
    }

    pub fn k_cache_mut(&mut self) -> &mut Cache<Bk, DK> {
        &mut self.k
    }

    pub fn v_cache_mut(&mut self) -> &mut Cache<Bv, DV> {
        &mut self.v
    }

    pub fn k(&self) -> Result<Option<Tensor<Bk, DK>>, Error> {
        self.k.current_data().map_err(Into::into)
    }

    pub fn v(&self) -> Result<Option<Tensor<Bv, DV>>, Error> {
        self.v.current_data().map_err(Into::into)
    }

    pub fn append(
        &mut self,
        k: Tensor<Bk, DK>,
        v: Tensor<Bv, DV>,
    ) -> Result<(Tensor<Bk, DK>, Tensor<Bv, DV>), Error> {
        let (mut k_dims, mut v_dims) = (k.dims(), v.dims());
        let (k_device, v_device) = (k.device(), v.device());
        self.k.append(k)?;
        self.v.append(v)?;
        let out_k = self.k.current_data()?;
        let out_v = self.v.current_data()?;
        let k = match out_k {
            None => {
                k_dims[self.k.dim] = 0;
                Tensor::zeros(k_dims, &k_device)
            }
            Some(k) => k,
        };
        let v = match out_v {
            None => {
                v_dims[self.k.dim] = 0;
                Tensor::zeros(v_dims, &v_device)
            }
            Some(v) => v,
        };
        Ok((k, v))
    }

    pub fn current_seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }
}
