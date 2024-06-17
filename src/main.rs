#![feature(slice_split_once)]
#![feature(slice_partition_dedup)]

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

fn main() {
    let compute_chunk_size_ratio = std::env::var("COMPUTE_CHUNK_SIZE_RATIO")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(16);
    println!("compute chunk size ratio: {compute_chunk_size_ratio}");

    let scan_chunk_size_ratio = std::env::var("SCAN_CHUNK_SIZE_RATIO")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(1);
    println!("scan chunk size ratio: {scan_chunk_size_ratio}");

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
            let n_threads = scan_chunk_size_ratio * rayon::current_num_threads();
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
            let _timer = ScopeTimer::with_label("merge");
            let mut edges = edges;
            let sizes = edges.iter().map(Vec::len).collect::<Vec<_>>();
            let total_len = sizes.iter().sum::<usize>();
            let edges = edges.iter_mut().map(|x| x.as_slice()).collect::<Vec<_>>();

            let mut result: Vec<FatNodeId> = Vec::with_capacity(total_len);
            let spare = result.spare_capacity_mut();
            let spare: &mut [FatNodeId] = unsafe { std::mem::transmute(spare) };
            rayon_k_way_merge::merge(spare, edges);
            unsafe { result.set_len(total_len) };
            result
        };

        {
            let _timer = ScopeTimer::with_label("dedup");

            let mut edges = edges;
            let chunk_size = edges.len() / rayon::current_num_threads() + 1;
            let mut parts = edges
                .par_chunks_mut(chunk_size)
                .map(|chunk| chunk.partition_dedup().0)
                .map(|x| &x[..])
                .collect::<Vec<_>>();

            let mut previous = parts[0].last().unwrap();
            for part in &mut parts[1..] {
                if let Some(pos) = part.iter().position(|x| x != previous) {
                    let (_, current) = part.split_at(pos);
                    *part = current;
                    previous = current.last().unwrap();
                } else {
                    *part = &mut [];
                }
            }
            let total_len = parts.iter().map(|x| x.len()).sum::<usize>();

            let mut result: Vec<FatNodeId> = Vec::with_capacity(total_len);
            let spare = result.spare_capacity_mut();
            let spare: &mut [FatNodeId] = unsafe { std::mem::transmute(spare) };
            rayon_k_way_merge::merge(spare, parts);
            unsafe { result.set_len(total_len) };
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

    let result: usize = {
        let _timer = ScopeTimer::with_label("computation");
        lowers
            .par_chunks(lowers.len() / (compute_chunk_size_ratio * rayon::current_num_threads()))
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
