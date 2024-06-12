#![feature(slice_split_once)]

use bitvec::prelude::*;
use std::cmp::Ordering;
use std::fs::File;
use std::sync::Mutex;
use std::time;
use voracious_radix_sort::RadixSort;

use bitvec::bitvec;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

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

fn par_merge_helper(a: &[FatNodeId], b: &[FatNodeId], result: &mut [FatNodeId]) {
    const CUTOFF: usize = 2000;
    let (small, big) = if a.len() < b.len() { (a, b) } else { (b, a) };
    if small.is_empty() {
        result.copy_from_slice(big);
    } else if small.len() + big.len() <= CUTOFF {
        itertools::merge(a, b)
            .zip(result)
            .for_each(|(src, dst)| *dst = *src);
    } else {
        let big_mid = big.len() / 2;
        let (big_left, big_right) = big.split_at(big_mid);
        let small_mid = match small.binary_search(&big[big_mid]) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };
        let (small_left, small_right) = small.split_at(small_mid);
        let (result_left, result_right) = result.split_at_mut(big_mid + small_mid);
        rayon::scope(|s| {
            s.spawn(|_| par_merge_helper(small_left, big_left, result_left));
            par_merge_helper(small_right, big_right, result_right);
        });
    }
}

fn merge(
    elements: &[Vec<FatNodeId>],
    data: &mut [FatNodeId],
    aux: &mut [FatNodeId],
    expect_result_in_data: bool,
) {
    match elements.len() {
        0 => {}
        1 => {
            if expect_result_in_data {
                data.copy_from_slice(&elements[0])
            } else {
                aux.copy_from_slice(&elements[0])
            }
        }
        2 => {
            if expect_result_in_data {
                par_merge_helper(&elements[0], &elements[1], data);
            } else {
                par_merge_helper(&elements[0], &elements[1], aux);
            }
        }
        _ => {
            let mid = elements.len() / 2;
            let (left, right) = elements.split_at(mid);
            let left_len = left.iter().map(Vec::len).sum::<usize>();
            let (data_left, data_right) = data.split_at_mut(left_len);
            let (aux_left, aux_right) = aux.split_at_mut(left_len);
            rayon::scope(|s| {
                s.spawn(|_| {
                    merge(left, data_left, aux_left, !expect_result_in_data);
                });
                merge(right, data_right, aux_right, !expect_result_in_data);
            });
            if expect_result_in_data {
                par_merge_helper(aux_left, aux_right, data);
            } else {
                par_merge_helper(data_left, data_right, aux);
            }
        }
    };
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
                        let mut local = current
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
                            })
                            .collect::<Vec<_>>();
                        // local.voracious_sort();
                        result.lock().unwrap().push(local);
                    });
                }
            });
            result.into_inner().unwrap()
        };

        let edges = {
            let _timer = ScopeTimer::with_label("sort and dedup");
            // let total_len = edges.iter().map(Vec::len).sum::<usize>();
            // let mut aux = Vec::with_capacity(total_len);
            // aux.resize(total_len, FatNodeId::default());
            // let mut result = Vec::with_capacity(total_len);
            // result.resize(total_len, FatNodeId::default());
            // merge(&edges, &mut result, &mut aux, true);
            let mut result = edges.concat();
            result.voracious_mt_sort(rayon::current_num_threads());
            result.dedup();
            result
        };

        edges
    };

    let max_node_id = edges_group_by_src.last().map(|x| x >> 32).unwrap() as usize;

    let (src_list, dst_list): (Vec<_>, Vec<_>) = edges_group_by_src
        .into_par_iter()
        .map(|x| {
            let Edge(src, dst) = Edge::from(x);
            (src, dst)
        })
        .unzip();

    let lowers = {
        let _timer = ScopeTimer::with_label("counting neighbors");
        let mut lowers: Vec<&[NodeId]> = Vec::new();
        lowers.resize_with(max_node_id + 1, Default::default);

        let mut src_list = &src_list[..];
        let mut dst_list = &dst_list[..];

        while !src_list.is_empty() {
            let src = src_list[0];
            let split_pos = src_list.partition_point(|&x| x == src);
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

                let mut result = 0_usize;
                for &first in data {
                    for neigh in first {
                        bitmap.set(*neigh as usize, true);
                    }

                    result += first
                        .iter()
                        .flat_map(|second| lowers[*second as usize].iter())
                        .map(|third| bitmap[*third as usize] as usize)
                        .sum::<usize>();

                    for neigh in first {
                        bitmap.set(*neigh as usize, false);
                    }
                }

                result
            })
            .sum()
    };

    println!("{result}");
}
