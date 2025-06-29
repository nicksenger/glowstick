use std::marker::PhantomData;

use glowstick::{num::*, Empty, ShapeFragment};
use glowstick::{Dimension, Mappend, Shape, Shape1, Shp, TensorShape};

pub trait Content {}
impl Content for f32 {}
impl<T: Arr> Content for T
where
    T: Arr,
    <T as Arr>::Content: Content,
{
}

pub trait Arr {
    type Content: Content;
    type Size: Dimension;

    fn get(&self, idx: usize) -> &Self::Content;
    fn get_mut(&mut self, idx: usize) -> &mut Self::Content;
    fn swap(&mut self, i: usize, j: usize);
}

#[derive(Debug, PartialEq, Eq)]
pub struct TVec<T, N: Dimension>(Vec<T>, PhantomData<N>);

impl<T, N: Dimension> Clone for TVec<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<T> TVec<T, U0> {
    pub fn new() -> Self {
        Self(vec![], PhantomData)
    }
}

impl<T> Default for TVec<T, U0> {
    fn default() -> Self {
        TVec(vec![], PhantomData)
    }
}

impl<T: Clone, N: Dimension> TVec<T, N> {
    pub fn with(element: T) -> Self {
        Self(vec![element; <N as Dimension>::USIZE], PhantomData)
    }
}

impl<N: Dimension> TVec<f32, N> {
    pub fn zeros() -> Self {
        Self::with(0.)
    }
}

impl<T, N: Dimension> TVec<T, N>
where
    (N, U1): Add,
    <(N, U1) as Add>::Out: Dimension,
{
    pub fn push(mut self, item: T) -> TVec<T, <(N, U1) as Add>::Out> {
        self.0.push(item);
        TVec(self.0, PhantomData)
    }
}

impl<T, N: Dimension> TVec<T, N>
where
    (N, U1): Sub,
    <(N, U1) as Sub>::Out: Dimension,
{
    pub fn pop(mut self) -> TVec<T, <(N, U1) as Sub>::Out> {
        let _ = self.0.pop();
        TVec(self.0, PhantomData)
    }
}

impl<C: Content, N: Dimension> Arr for TVec<C, N> {
    type Content = C;
    type Size = N;

    fn get(&self, idx: usize) -> &Self::Content {
        &self.0[idx]
    }

    fn get_mut(&mut self, idx: usize) -> &mut Self::Content {
        &mut self.0[idx]
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j);
    }
}

pub trait Tensor: Arr {
    type Shape: Shape;

    /// Iterate over the tensor element-wise
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32>;
}

impl<N: Dimension> Tensor for TVec<f32, N>
where
    Self: Arr,
    <Self as Arr>::Size: NonZero,
{
    type Shape = Shape1<<Self as Arr>::Size>;

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.0.iter_mut()
    }
}

impl<T, N: Dimension> Tensor for TVec<T, N>
where
    T: Tensor,
    Self: Arr,
    <Self as Arr>::Size: NonZero,
    (
        Shp<(<Self as Arr>::Size, Empty)>,
        <<T as Tensor>::Shape as Shape>::Fragment,
    ): Mappend,
    <(
        Shp<(<Self as Arr>::Size, Empty)>,
        <<T as Tensor>::Shape as Shape>::Fragment,
    ) as Mappend>::Out: ShapeFragment,
{
    type Shape = TensorShape<
        <(
            Shp<(<Self as Arr>::Size, Empty)>,
            <<T as Tensor>::Shape as Shape>::Fragment,
        ) as Mappend>::Out,
    >;

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.0.iter_mut().flat_map(|content| content.iter_mut())
    }
}

impl<T, N: Dimension> glowstick::Tensor for TVec<T, N>
where
    TVec<T, N>: Tensor,
{
    type Shape = <Self as Tensor>::Shape;
}

#[macro_export]
macro_rules! t {
    [$($x:expr),+] => {{
        let tv = $crate::tensor::TVec::new();
        $(
            let tv = tv.push($x);
        )+
        tv
    }}
}
