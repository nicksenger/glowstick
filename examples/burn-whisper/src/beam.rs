use std::usize;

use burn::prelude::*;

use crate::model::cache::TensorCache;
use crate::transcribe::Error;

#[derive(Clone, Debug)]
pub struct BeamNode<T: Clone + std::fmt::Debug> {
    pub seq: Vec<T>,
    pub log_prob: f64,
    pub is_finished: bool,
}

pub fn beam_search<B, T, F, G>(
    batch_beams: Vec<Vec<BeamNode<T>>>,
    next: F,
    is_finished: G,
    beam_width: usize,
    max_depth: usize,
    tensor_cache: &mut [TensorCache<B, 4>],
) -> Result<Vec<(usize, Vec<T>)>, Error>
where
    B: Backend,
    T: Clone + std::fmt::Debug,
    F: FnMut(
        usize,
        &[Vec<BeamNode<T>>],
        &mut [TensorCache<B, 4>],
    ) -> Result<Vec<Vec<Vec<(T, f64)>>>, Error>,
    G: Fn(&BeamNode<T>) -> bool + Clone,
{
    let mut next = next;
    let mut batch_beams = batch_beams;
    for i in 0..max_depth {
        if batch_beams
            .iter()
            .all(|batch| batch.iter().all(|beam| beam.is_finished))
        {
            break;
        }

        (batch_beams, next) = beam_search_step(
            i,
            batch_beams,
            next,
            is_finished.clone(),
            beam_width,
            tensor_cache,
        )?;
    }

    Ok(batch_beams
        .into_iter()
        .enumerate()
        .map(|(i, beams)| {
            beams
                .into_iter()
                .max_by(|a, b| {
                    (a.log_prob / a.seq.len() as f64)
                        .partial_cmp(&(b.log_prob / b.seq.len() as f64))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|x| (i, x.seq))
                .unwrap_or_else(|| (usize::MAX, vec![]))
        })
        .collect())
}

pub fn beam_search_step<B: Backend, T, F, G>(
    i: usize,
    batch_beams: Vec<Vec<BeamNode<T>>>,
    mut next: F,
    is_finished: G,
    beam_size: usize,
    tensor_cache: &mut [TensorCache<B, 4>],
) -> Result<(Vec<Vec<BeamNode<T>>>, F), Error>
where
    T: Clone + std::fmt::Debug,
    F: FnMut(
        usize,
        &[Vec<BeamNode<T>>],
        &mut [TensorCache<B, 4>],
    ) -> Result<Vec<Vec<Vec<(T, f64)>>>, Error>,
    G: Fn(&BeamNode<T>) -> bool,
{
    let batch_continuations = next(i, &batch_beams, tensor_cache)?;
    let mut out_batch_beams = vec![];
    for (beams, continuations) in batch_beams.into_iter().zip(batch_continuations) {
        let mut finished_beams = Vec::with_capacity(beam_size);
        let mut new_beams = Vec::with_capacity(beam_size);

        for (beam_node, continuations) in beams.into_iter().zip(continuations) {
            if is_finished(&beam_node) {
                finished_beams.push(BeamNode {
                    is_finished: true,
                    ..beam_node
                });
            } else {
                let top_new_beams =
                    get_top_elements(&continuations, |(_, log_prob)| *log_prob, beam_size)
                        .into_iter()
                        .map(|(tok, log_prob)| {
                            let mut node = BeamNode {
                                seq: beam_node
                                    .seq
                                    .iter()
                                    .cloned()
                                    .chain(std::iter::once(tok.clone()))
                                    .collect(),
                                log_prob: *log_prob,
                                is_finished: false,
                            };
                            node.is_finished = is_finished(&node);
                            node
                        });

                new_beams.push(top_new_beams.collect::<Vec<_>>());
            }
        }

        if finished_beams.len() == beam_size {
            out_batch_beams.push(finished_beams);
            continue;
        }

        let new_beams = if i == 0 {
            new_beams.into_iter().take(1).flatten().collect::<Vec<_>>()
        } else {
            new_beams.into_iter().flatten().collect::<Vec<_>>()
        };

        let all_beams = new_beams
            .into_iter()
            .chain(finished_beams)
            .collect::<Vec<_>>();
        out_batch_beams.push(
            get_top_elements(
                &all_beams,
                |beam| beam.log_prob / beam.seq.len() as f64,
                beam_size,
            )
            .into_iter()
            .cloned()
            .collect::<Vec<_>>(),
        )
    }

    Ok((out_batch_beams, next))
}

fn get_top_elements<T: std::fmt::Debug>(
    elems: &[T],
    score: impl Fn(&T) -> f64,
    num: usize,
) -> Vec<&T> {
    let mut top_elems = Vec::with_capacity(num);
    let mut scores = Vec::with_capacity(num);

    for elem in elems {
        let score = score(elem);

        if top_elems.len() == num && score < scores[0] {
            continue;
        }

        if let Some((idx, _)) = scores.iter().enumerate().find(|(_, s)| **s >= score) {
            top_elems.insert(idx, elem);
            scores.insert(idx, score);
        } else {
            top_elems.push(elem);
            scores.push(score);
        }

        if top_elems.len() > num {
            top_elems.remove(0);
            scores.remove(0);
        }
    }

    top_elems
}
