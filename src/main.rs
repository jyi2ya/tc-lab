#![feature(slice_split_once)]

use bitvec::prelude::*;
use std::cmp::Ordering;
use std::fs::File;
use std::sync::Mutex;
use std::time;

use bitvec::bitvec;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

struct ScopeTimer {
    label: String,
    start: time::Instant,
}

#[allow(dead_code)]
impl ScopeTimer {
    pub fn with_label(label: impl ToString) -> Self {
        Self {
            label: label.to_string(),
            start: time::Instant::now(),
        }
    }

    pub fn stop(self) {}
}

impl Drop for ScopeTimer {
    fn drop(&mut self) {
        let end = time::Instant::now();
        println!(
            "[ScopeTimer] {} seconds | {}",
            (end - self.start).as_secs_f64(),
            self.label,
        );
    }
}

type NodeId = u32;
type FatNodeId = u64;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
struct Edge(NodeId, NodeId);

impl From<Edge> for FatNodeId {
    #[inline(always)]
    fn from(value: Edge) -> Self {
        ((value.0 as FatNodeId) << 32) | (value.1 as FatNodeId)
    }
}

impl From<FatNodeId> for Edge {
    #[inline(always)]
    fn from(value: FatNodeId) -> Self {
        Self((value >> 32) as NodeId, value as NodeId)
    }
}

impl From<&FatNodeId> for Edge {
    #[inline(always)]
    fn from(value: &FatNodeId) -> Self {
        Self::from(*value)
    }
}

fn parallel_merge(result: &mut [u64], mut parts: Vec<&mut [u64]>) -> Option<()> {
    const CUTOFF: usize = 5000;
    let total_len = parts.iter().map(|x| x.len()).sum::<usize>();
    if total_len < CUTOFF || parts.len() == 1 {
        match parts.len() {
            0 => {}
            1 => result.copy_from_slice(parts[0]),
            2 => itertools::merge(parts[0].iter(), parts[1].iter())
                .zip(result)
                .for_each(|(src, dst)| *dst = *src),
            _ => itertools::kmerge(parts)
                .zip(result)
                .for_each(|(src, dst)| *dst = *src),
        }
    } else {
        parts.sort_by_key(|part| part.len());
        let position = parts.iter().position(|x| !x.is_empty())?;
        let (_, parts) = parts.split_at_mut(position);
        let mid_pos = parts.last()?.len() / 2;
        let mid_val = parts.last()?[mid_pos];
        let (left, right): (Vec<_>, Vec<_>) = parts
            .iter_mut()
            .map(|part| {
                let split_pos = part
                    .binary_search_by(|element| match element.cmp(&mid_val) {
                        Ordering::Equal => Ordering::Greater,
                        ord => ord,
                    })
                    .unwrap_err();
                part.split_at_mut(split_pos)
            })
            .unzip();
        let left_size = left.iter().map(|x| x.len()).sum::<usize>();
        let (left_result, right_result) = result.split_at_mut(left_size);
        rayon::scope(|s| {
            s.spawn(|_| {
                parallel_merge(left_result, left);
            });
            parallel_merge(right_result, right);
        });
    }
    Some(())
}

fn main() {
    let _timer = ScopeTimer::with_label("totals");

    let edges_group_by_src = {
        let _timer = ScopeTimer::with_label("read and build");

        let args: Vec<_> = std::env::args().collect();
        let filename = &args[1];
        let file = File::open(filename).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let content = &mmap[..];

        let edges: Vec<_> = {
            let _timer = ScopeTimer::with_label("scan and parse");
            let n_threads = rayon::current_num_threads();
            let partition_size = content.len() / n_threads;
            let result = Mutex::new(Vec::new());
            rayon::scope(|s| {
                let mut content = content;
                while !content.is_empty() {
                    let cap = if content.len() < partition_size {
                        content.len()
                    } else {
                        content[partition_size..]
                            .iter()
                            .position(|&x| x == b'\n')
                            .map(|idx| idx + partition_size)
                            .unwrap_or(content.len())
                    };
                    let (current, res) = content.split_at(cap);
                    content = res;

                    s.spawn(|_| {
                        let mut local = Vec::with_capacity(partition_size / 20);
                        let extend = current
                            .split(|&x| x == b'\n')
                            .map(|line| line.strip_suffix(&[b'\r']).unwrap_or(line))
                            .filter_map(|line| line.split_once(|x| x.is_ascii_whitespace()))
                            .filter_map(|(src, dst)| {
                                let src: NodeId = atoi_simd::parse(src).ok()?;
                                let dst: NodeId = atoi_simd::parse(dst).ok()?;
                                match NodeId::cmp(&src, &dst) {
                                    Ordering::Greater => Some(FatNodeId::from(Edge(src, dst))),
                                    Ordering::Equal => None,
                                    Ordering::Less => Some(FatNodeId::from(Edge(dst, src))),
                                }
                            });
                        local.extend(extend);
                        local.as_mut_slice().par_sort_unstable();
                        result.lock().unwrap().push(local);
                    });
                }
            });
            result.into_inner().unwrap()
        };

        let edges = {
            let _timer = ScopeTimer::with_label("concat");
            let mut edges = edges;
            let sizes = edges.iter().map(Vec::len).collect::<Vec<_>>();
            let total_len = sizes.iter().sum::<usize>();
            let edges = edges
                .iter_mut()
                .map(|x| x.as_mut_slice())
                .collect::<Vec<_>>();

            let mut result: Vec<FatNodeId> = Vec::with_capacity(total_len);
            let spare = result.spare_capacity_mut();
            let spare: &mut [FatNodeId] = unsafe { std::mem::transmute(spare) };
            parallel_merge(spare, edges);
            unsafe { result.set_len(total_len) };
            result
        };

        {
            let _timer = ScopeTimer::with_label("sort and dedup");
            let mut result = edges;

            result.dedup();
            result
        }
    };

    let max_node_id = edges_group_by_src.last().map(|x| x >> 32).unwrap() as usize;

    let (src_list, dst_list): (Vec<_>, Vec<_>) = {
        let _tiemr = ScopeTimer::with_label("unzip");
        edges_group_by_src
            .into_par_iter()
            .map(|x| {
                let Edge(src, dst) = Edge::from(x);
                (src, dst)
            })
            .unzip()
    };

    let lowers = {
        let _timer = ScopeTimer::with_label("counting neighbors");
        let mut lowers: Vec<&[NodeId]> = Vec::new();
        lowers.resize_with(max_node_id + 1, Default::default);

        let mut src_list = &src_list[..];
        let mut dst_list = &dst_list[..];

        while !src_list.is_empty() {
            let src = src_list[0];
            let split_pos = src_list
                .iter()
                .cloned()
                .chain(std::iter::once(src + 1))
                .position(|x| x != src)
                .unwrap();
            let (dst_current, dst_res) = dst_list.split_at(split_pos);
            let (_, src_res) = src_list.split_at(split_pos);
            lowers[src as usize] = dst_current;
            (src_list, dst_list) = (src_res, dst_res);
        }

        lowers.into_boxed_slice()
    };

    const TRUNK_SIZE_RATIO: usize = 32;

    let result: usize = {
        let _timer = ScopeTimer::with_label("computation");
        lowers
            .par_chunks(lowers.len() / (TRUNK_SIZE_RATIO * rayon::current_num_threads()))
            .map(|data| {
                let mut bitmap = bitvec![];
                bitmap.resize(max_node_id + 1, false);
                let mut bitmap = bitmap.into_boxed_bitslice();

                data.iter()
                    .filter_map(|&first| {
                        let min = first.first()?;
                        let max = first.last()?;
                        first.iter().for_each(|&idx| bitmap.set(idx as usize, true));

                        let result = first
                            .iter()
                            .map(|&idx| lowers[idx as usize])
                            .filter(|&second| match (second.first(), second.last()) {
                                (Some(first), Some(last)) => first <= max && last >= min,
                                _ => false,
                            })
                            .flatten()
                            .filter(|&&idx| bitmap[idx as usize])
                            .count();

                        first
                            .iter()
                            .for_each(|&idx| bitmap.set(idx as usize, false));

                        Some(result)
                    })
                    .sum::<usize>()
            })
            .sum()
    };

    println!("{result}");
}
